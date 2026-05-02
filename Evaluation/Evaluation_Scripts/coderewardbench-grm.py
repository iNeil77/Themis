# Invoked as:
# CUDA_VISIBLE_DEVICES=0 python coderewardbench-seqcls.py "YOUR_ORG/YOUR_MODEL_NAME" --output "YOUR_OUTPUT_DIR" --use-system-prompts --use-aspect-prompts --batch-size 32 --max-length 4096

"""
This script evaluates a scalar reward model on the Code RewardBench dataset,
calculating accuracy across different aspects and programming languages.
"""

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import argparse
import json
import os

# Modified from Huggingface trl package AutoModelForCausalLMWithValueHead class
# Enabling better customization for generalizable reward modeling
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from trl import PreTrainedModelWrapper


class ValueHead(nn.Module):
    """Linear projection head mapping hidden states to a scalar reward."""

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        # get vhead config
        if hasattr(config, "vhead_layer_type"): # config from json first
            self.layer_type = config.vhead_layer_type
        else:
            self.layer_type = kwargs.pop("vhead_layer_type", 'mlp')
        if hasattr(config, 'vhead_num_neurons'):
            num_neurons = config.vhead_num_neurons
        else:
            num_neurons = kwargs.pop("vhead_num_neurons", 1024)
        if hasattr(config, 'vhead_num_layers'):
            num_layers = config.vhead_num_layers
        else:
            num_layers = kwargs.pop("vhead_num_layers", 1)

        if self.layer_type == 'linear':
            self.summary = nn.Linear(hidden_size, 1)
        else:
            module_lis = []
            input_neurons = hidden_size
            for i in range(num_layers):
                module_lis.extend([nn.Linear(input_neurons, num_neurons), nn.ReLU()])
                input_neurons = num_neurons
                
            module_lis.append(nn.Linear(num_neurons, 1))
            self.summary = nn.Sequential(*module_lis)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if (self.layer_type == 'linear' and output.dtype != self.summary.weight.dtype):
            output = output.to(self.summary.weight.dtype)
        elif (self.layer_type != 'linear' and output.dtype != self.summary[0].weight.dtype):
            output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output)
        return output


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
        "layer_type",
        'num_neurons',
        'num_layers',
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. 
        """
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if (hasattr(self.v_head.summary, 'weight') and last_hidden_state.device != self.v_head.summary.weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        elif not hasattr(self.v_head.summary, 'weight') and (last_hidden_state.device != self.v_head.summary[0].weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.summary[0].weight.device)
        
        # use the last token value as reward
        if torch.any(attention_mask[:, 0] == 0):
            # left padding
            last_index = attention_mask.shape[-1] - 1
        else:
            # right padding
            last_index = attention_mask.sum(dim=-1) - 1
        value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    

    def post_init(self, state_dict):
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True
    
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


# System prompts for different coding aspects
SYSTEM_PROMPT_MAP = {

    'Full': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Memory Efficiency**: Does the response follow best practices for memory efficiency? Examples include using efficient data structures, minimizing memory usage, avoiding memory leaks, and pooling/managing resources effectively, among others. This is a tenet that applies to evaluating code responses. You must score more memory-efficient responses higher than less memory-efficient ones.

4. **Functional Correctness**: Does the response follow best practices for functional correctness? Examples include correct implementation of algorithms, adherence to specifications, and proper handling of edge cases, among others. This is a tenet that applies to evaluating code responses. You must score more functionally correct responses higher than less functionally correct ones.

5. **Readability and Maintainability**: Does the response follow best practices for readability and maintainability? Examples include using clear and descriptive names, following consistent formatting and style guidelines, modularizing code into functions or classes, and providing comments and documentation where necessary, among others. This is a tenet that applies to evaluating code responses. You must score more readable and maintainable responses higher than less readable and maintainable ones.

6. **Runtime Efficiency**: Does the response follow best practices for runtime efficiency? Examples include using efficient algorithms and data structures, minimizing time complexity, avoiding unnecessary computations, caching results, and leveraging parallel processing or asynchronous programming techniques where appropriate, among others. This is a tenet that applies to evaluating code responses. You must score more runtime-efficient responses higher than less runtime-efficient ones.

7. **Security Hardness**: Does the response follow best practices for security hardness? Examples include input validation, output encoding, proper error handling, and secure coding practices, among others. This is a tenet that applies to evaluating code responses. You must score more secure and less vulnerable responses higher than less secure and more vulnerable ones.
""",

    'Functional_Correctness': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Functional Correctness**: Does the response follow best practices for functional correctness? Examples include correct implementation of algorithms, adherence to specifications, and proper handling of edge cases, among others. This is a tenet that applies to evaluating code responses. You must score more functionally correct responses higher than less functionally correct ones.""",

    'Memory_Efficiency': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Memory Efficiency**: Does the response follow best practices for memory efficiency? Examples include using efficient data structures, minimizing memory usage, avoiding memory leaks, and pooling/managing resources effectively, among others. This is a tenet that applies to evaluating code responses. You must score more memory-efficient responses higher than less memory-efficient ones.""",

    'Readability_Maintainability': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Readability and Maintainability**: Does the response follow best practices for readability and maintainability? Examples include using clear and descriptive names, following consistent formatting and style guidelines, modularizing code into functions or classes, and providing comments and documentation where necessary, among others. This is a tenet that applies to evaluating code responses. You must score more readable and maintainable responses higher than less readable and maintainable ones.""",

    'Runtime_Efficiency': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Runtime Efficiency**: Does the response follow best practices for runtime efficiency? Examples include using efficient algorithms and data structures, minimizing time complexity, avoiding unnecessary computations, caching results, and leveraging parallel processing or asynchronous programming techniques where appropriate, among others. This is a tenet that applies to evaluating code responses. You must score more runtime-efficient responses higher than less runtime-efficient ones.""",

    'Security_Hardness': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Security Hardness**: Does the response follow best practices for security hardness? Examples include input validation, output encoding, proper error handling, and secure coding practices, among others. This is a tenet that applies to evaluating code responses. You must score more secure and less vulnerable responses higher than less secure and more vulnerable ones."""
}


class RewardModelEvaluator:

    def __init__(self,
                model_path,
                max_length,
                use_system_prompts=False,
                use_aspect_prompts=False):
        """
        Initialize the reward model evaluator.
        
        Args:
            model_path (str): Path to the reward model on HuggingFace or local path
            use_system_prompts (bool): Whether to use aspect-specific system prompts
        """
        print(f"System prompts enabled: {use_system_prompts}")
        self.use_system_prompts = use_system_prompts
        print(f"Aspect prompts enabled: {use_aspect_prompts}")
        self.use_aspect_prompts = use_aspect_prompts

        # Load tokenizer and model
        print(f"Loading model and tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA detected: Using {device_count} GPUs for Inference")
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                device_map="auto",
            )
            self.input_device = next(iter(self.model.parameters())).device.type
        else:
            print("CUDA not detected: Using CPU Inference")
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            self.input_device = 'cpu'
        self.model.eval()
        
        # Set max sequence length
        if self.tokenizer.model_max_length is not None and self.tokenizer.model_max_length < max_length:
            self.max_length = self.tokenizer.model_max_length
            print(
                f"Warning: Tokenizer max length {self.tokenizer.model_max_length} is less than specified max_length {max_length}. Using {self.max_length} instead."
            )
        else:
            self.max_length = max_length
            print(f"Using max sequence length: {self.max_length}")
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def format_conversation(
            self, 
            prompt, 
            response, 
            aspect=None
        ):
        """
        Format prompt and response using the tokenizer's chat template.
        
        Args:
            prompt (str): The input prompt (user message)
            response (str): The model response (assistant message)
            aspect (str): The coding aspect for system prompt selection
            
        Returns:
            str: Formatted conversation text
        """
        # Create conversation in the standard chat format
        conversation = []

        # Add system prompt if enabled
        if self.use_system_prompts and aspect and aspect in SYSTEM_PROMPT_MAP:
            if self.use_aspect_prompts:
                conversation.append({
                    "role": "system",
                    "content": SYSTEM_PROMPT_MAP[aspect]
                })
            else:
                conversation.append({
                    "role": "system",
                    "content": SYSTEM_PROMPT_MAP['Full']
                })

        # Add user and assistant messages
        conversation.extend([{
            "role": "user",
            "content": prompt
        }, {
            "role": "assistant",
            "content": response
        }])

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                formatted_text = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False)
                return formatted_text
            except Exception as e:
                print(f"Warning: Failed to apply chat template ({e}), falling back to simple concatenation")

        # Fallback to simple concatenation if chat template is not available
        if self.use_system_prompts:
            if self.use_aspect_prompts and aspect and aspect in SYSTEM_PROMPT_MAP:
                return f"System:\n{SYSTEM_PROMPT_MAP[aspect]}\n\nUser:\n{prompt}\n\nAssistant:\n{response}"
            else:
                return f"System:\n{SYSTEM_PROMPT_MAP['Full']}\n\nUser:\n{prompt}\n\nAssistant:\n{response}"
        else:
            return f"User: {prompt}\nAssistant: {response}"


    @torch.inference_mode()
    def get_reward_scores_batch(
            self, 
            conversations, 
            batch_size=8
        ):
        """
        Get reward scores for a batch of conversations.
        
        Args:
            conversations (list): List of formatted conversation strings
            batch_size (int): Number of conversations to process at once
            
        Returns:
            list: List of reward scores
        """
        all_scores = []

        # Process in batches
        for i in range(0, len(conversations), batch_size):
            batch_conversations = conversations[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_conversations,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True).to(self.input_device)

            # Get model output
            _, _, reward_tensor = self.model(**inputs)
            # Extract scalar scores and detach from computation graph
            scores = reward_tensor.cpu().detach()

            # Handle single example case
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

            # Move to CPU and convert to list
            batch_scores = scores.cpu().tolist()
            all_scores.extend(batch_scores)

            # Clean up to reduce memory usage
            del inputs, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_scores


    def evaluate_batch(
            self,
            prompts,
            chosen_responses,
            rejected_responses,
            aspects,
            batch_size=8
        ):
        """
        Evaluate a batch of prompt-response pairs.
        
        Args:
            prompts (list): List of input prompts
            chosen_responses (list): List of preferred responses
            rejected_responses (list): List of dispreferred responses
            aspects (list): List of coding aspects for each example
            batch_size (int): Batch size for model inference
            
        Returns:
            list: List of evaluation results
        """
        # Format all conversations
        chosen_conversations = [
            self.format_conversation(prompt, chosen,aspect) 
            for prompt, chosen, aspect in zip(prompts, chosen_responses, aspects)
        ]
        rejected_conversations = [
            self.format_conversation(prompt, rejected, aspect) 
            for prompt, rejected, aspect in zip(prompts, rejected_responses, aspects)
        ]

        # Get scores for all conversations
        chosen_scores = self.get_reward_scores_batch(chosen_conversations, batch_size)
        rejected_scores = self.get_reward_scores_batch(rejected_conversations, batch_size)

        # Compile results
        results = []
        for chosen_score, rejected_score in zip(chosen_scores, rejected_scores):
            results.append({
                'chosen_score': chosen_score,
                'rejected_score': rejected_score,
                'correct': chosen_score > rejected_score,
                'score_diff': chosen_score - rejected_score
            })

        return results


    def evaluate_dataset(
            self,
            dataset_name="project-themis/Themis-CodeRewardBench",
            config=None,
            split=None,
            batch_size=8
        ):
        """
        Evaluate the reward model on the CRB dataset.
        
        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            config (str): Configuration name for the dataset
            subset (str): Optional subset to evaluate on
            batch_size (int): Batch size for model inference
            
        Returns:
            dict: Evaluation results
        """
        print(f"Loading dataset: {dataset_name}")
        if config:
            print(f"Using config: {config} and split: {split}")

        # Load dataset with optional config
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, "Full", split="Full")

        print(f"Evaluating on {len(dataset)} examples with batch size {batch_size}.")

        results = []
        detailed_scores = []  # For storing individual scores with IDs

        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size),
                      desc="Processing batches"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]

            try:
                # Extract batch data
                prompts = batch['prompt'] if isinstance(
                    batch['prompt'], list) else [batch['prompt']]
                chosen_responses = batch['chosen'] if isinstance(
                    batch['chosen'], list) else [batch['chosen']]
                rejected_responses = batch['rejected'] if isinstance(
                    batch['rejected'], list) else [batch['rejected']]

                # Handle metadata
                if isinstance(batch['id'], list):
                    ids = batch['id']
                    languages = batch['language']
                    aspects = batch['aspect']
                    subsets = batch['subset']
                else:
                    ids = [batch['id']]
                    languages = [batch['language']]
                    aspects = [batch['aspect']]
                    subsets = [batch['subset']]

                # Evaluate batch
                batch_results = self.evaluate_batch(prompts, chosen_responses,
                                                    rejected_responses,
                                                    aspects, batch_size)

                # Add metadata to each result and store detailed scores
                for j, result in enumerate(batch_results):
                    result.update({
                        'id': ids[j],
                        'language': languages[j],
                        'aspect': aspects[j],
                        'subset': subsets[j]
                    })
                    # Store detailed scores for parquet export
                    detailed_scores.append({
                        'id': ids[j],
                        'chosen_score': result['chosen_score'],
                        'rejected_score': result['rejected_score'],
                        'correct': result['correct'],
                        'score_diff': result['score_diff'],
                        'language': languages[j],
                        'aspect': aspects[j],
                        'subset': subsets[j]
                    })

                results.extend(batch_results)

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue

        compiled_results = self.compile_results(results)
        compiled_results['detailed_scores'] = detailed_scores

        return compiled_results


    def compile_results(
            self, 
            results
        ):
        """
        Compile evaluation results into a comprehensive report.
        
        Args:
            results (list): List of individual evaluation results
            
        Returns:
            dict: Compiled results with overall and breakdown statistics
        """
        df = pd.DataFrame(results)

        # Overall accuracy
        overall_accuracy = df['correct'].mean()
        total_examples = len(df)

        # Accuracy by aspect
        aspect_accuracy = df.groupby('aspect')['correct'].agg(['mean', 'count']).round(4)
        aspect_accuracy.columns = ['accuracy', 'count']

        # Accuracy by language
        language_accuracy = df.groupby('language')['correct'].agg(['mean', 'count']).round(4)
        language_accuracy.columns = ['accuracy', 'count']

        # Accuracy by subset
        subset_accuracy = df.groupby('subset')['correct'].agg(['mean', 'count']).round(4)
        subset_accuracy.columns = ['accuracy', 'count']

        # Additional statistics
        mean_score_diff = df['score_diff'].mean()
        median_score_diff = df['score_diff'].median()
        std_score_diff = df['score_diff'].std()

        compiled_results = {
            'overall': {
                'accuracy': round(overall_accuracy, 4),
                'total_examples': total_examples,
                'correct_predictions': int(df['correct'].sum()),
                'mean_score_difference': round(mean_score_diff, 4),
                'median_score_difference': round(median_score_diff, 4),
                'std_score_difference': round(std_score_diff, 4)
            },
            'by_aspect': aspect_accuracy.to_dict('index'),
            'by_language': language_accuracy.to_dict('index'),
            'by_subset': subset_accuracy.to_dict('index'),
            'raw_results': results
        }

        return compiled_results


    def save_results(
            self, 
            results, 
            output_dir
        ):
        """
        Save evaluation results to the specified output directory.
        
        Args:
            results (dict): Compiled results from evaluate_dataset
            output_dir (str): Output directory path
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed scores as parquet
        scores_df = pd.DataFrame(results['detailed_scores'])

        # Set ID as index
        scores_df.set_index('id', inplace=True)
        scores_path = os.path.join(output_dir, 'scores.parquet')
        scores_df.to_parquet(scores_path, index=True)
        print(f"Detailed scores saved to: {scores_path}")

        # Save results as JSON (excluding raw_results and detailed_scores for cleaner output)
        results_to_save = {
            k: v
            for k, v in results.items()
            if k not in ['raw_results', 'detailed_scores']
        }
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Results saved to: {results_path}")


    def print_results(
            self, 
            results
        ):
        """
        Print formatted evaluation results.
        
        Args:
            results (dict): Compiled results from evaluate_dataset
        """
        print("\n" + "=" * 60)
        print("REWARD MODEL EVALUATION RESULTS")
        print("=" * 60)

        # Overall results
        overall = results['overall']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Examples: {overall['total_examples']}")
        print(f"  Correct Predictions: {overall['correct_predictions']}")
        print(f"  Accuracy: {overall['accuracy']:.2%}")
        print(f"  Mean Score Difference: {overall['mean_score_difference']:.4f}")
        print(f"  Median Score Difference: {overall['median_score_difference']:.4f}")
        print(f"  Score Difference Std: {overall['std_score_difference']:.4f}")

        # Accuracy by aspect
        print(f"\nACCURACY BY ASPECT:")
        for aspect, stats in results['by_aspect'].items():
            print(f"  {aspect:<25}: {stats['accuracy']:.2%} ({stats['count']} examples)")
        # Accuracy by language
        print(f"\nACCURACY BY LANGUAGE:")
        for language, stats in results['by_language'].items():
            print(f"  {language:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")
        # Accuracy by subset
        print(f"\nACCURACY BY SUBSET:")
        for subset, stats in results['by_subset'].items():
            print(f"  {subset:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on CRB dataset")
    parser.add_argument(
        "model_path",
        help="Path to the reward model"
    )
    parser.add_argument(
        "--dataset",
        default="project-themis/Themis-CodeRewardBench",
        help="Dataset name"
    )
    parser.add_argument(
        "--config",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--split",
        help="Specific subset to evaluate"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory name to save results"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Batch size for evaluation"
    )    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--use-system-prompts",
        action="store_true",
        default=False,
        help="Use aspect-specific system prompts in conversations"
    )
    parser.add_argument(
        "--use-aspect-prompts",
        action="store_true",
        default=False,
        help="Use aspect-specific system prompts in conversations"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RewardModelEvaluator(
        args.model_path,
        args.max_length,
        args.use_system_prompts,
        args.use_aspect_prompts
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.dataset,
        args.config,
        args.split,
        args.batch_size
    )

    # Print results
    evaluator.print_results(results)

    # Save results to output directory
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
