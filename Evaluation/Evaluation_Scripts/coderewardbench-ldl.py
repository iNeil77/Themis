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
import traceback


import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Gemma2PreTrainedModel, Gemma2Model
from transformers.utils import ModelOutput
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from huggingface_hub import snapshot_download


class MultiOutputNN(nn.Module):
    """Multi-layer regression network producing per-objective reward predictions."""

    def __init__(self, input_dim, output_dim, hidden_dims=[4096, 4096]):
        super(MultiOutputNN, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU())
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.network(x)
        return self.softmax(x.view(x.size(0), -1, 10))


class GatingNN(nn.Module):
    """Gating MLP with BatchNorm, dropout, and softmax for objective weighting."""

    def __init__(self, input_dim, output_dim, hidden_dim=4096, num_layers=2, temperature=1.0, dropout_prob=0.0, softmax=False):
        super(GatingNN, self).__init__()
        self.temperature = temperature
        self.softmax = softmax
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout_prob))

        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        if self.softmax:
            x = F.softmax(x / self.temperature, dim=1)
        return x

@dataclass
class CustomOutput(ModelOutput):
    """Named tuple-like container for LDL model forward pass outputs."""

    rewards: torch.FloatTensor = None
    hidden_state: Optional[Tuple[torch.FloatTensor, ...]] = None
    score: Optional[torch.FloatTensor] = None
    total_reward_distribution: Optional[torch.FloatTensor] = None
    weights: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LDLRewardModel27B(Gemma2PreTrainedModel):
    """Gemma2 backbone with label-distribution-learning reward heads and gating (LDL architecture)."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        config_dict = config.to_dict()
        self.num_objectives = config_dict.get("num_objectives", 220)
        self.regression_layer = MultiOutputNN(config.hidden_size, self.num_objectives)
        self.gating_layer = GatingNN(
            config.hidden_size,
            self.num_objectives // 10,
            temperature=config_dict.get("temperature", 1.0),
            softmax=config_dict.get("softmax", False),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CustomOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        tokens_hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(tokens_hidden_states.device)
            else:
                sequence_lengths = -1

        dummy_iterator = torch.arange(batch_size, device=tokens_hidden_states.device)
        hidden_states = tokens_hidden_states[dummy_iterator, sequence_lengths]
        assert hidden_states.shape == (batch_size, self.config.hidden_size)
        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            rewards = self.regression_layer(hidden_states)
            weights = self.gating_layer(hidden_states)
            weights = weights.unsqueeze(1)
            total_reward_distribution = torch.bmm(weights, rewards).squeeze(1)
            score = (
                total_reward_distribution
                * torch.linspace(0, 1, total_reward_distribution.size(-1)).to(tokens_hidden_states.device)
            ).sum(-1)
        return CustomOutput(
            rewards=rewards,
            weights=weights,
            hidden_state=hidden_states,
            total_reward_distribution=total_reward_distribution,
            score=score,
            logits=score,
        )

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory, dtype=torch.bfloat16)
        torch.save(self.regression_layer.state_dict(), os.path.join(save_directory, "regression_layer.pt"))
        torch.save(self.gating_layer.state_dict(), os.path.join(save_directory, "gating_layer.pt"))
        self.config.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, device_map=None, *model_args, **kwargs):
        if not os.path.exists(load_directory):
            cached_dir = snapshot_download(repo_id=load_directory)
        else:
            cached_dir = load_directory
        model = super(LDLRewardModel27B, cls).from_pretrained(
            cached_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )

        model.regression_layer = model.regression_layer.float()
        regression_layer_path = os.path.join(cached_dir, "regression_layer.pt")
        regression_layer_state_dict = torch.load(regression_layer_path, map_location="cpu")
        model.regression_layer.load_state_dict(regression_layer_state_dict)

        model.gating_layer = model.gating_layer.float()
        gating_layer_path = os.path.join(cached_dir, "gating_layer.pt")
        gating_layer_state_dict = torch.load(gating_layer_path, map_location="cpu")
        model.gating_layer.load_state_dict(gating_layer_state_dict)

        if device_map == "auto" or device_map == "balanced":
            max_memory = get_balanced_memory(model, no_split_module_classes=["Gemma2DecoderLayer", "Gemma2RMSNorm"])
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=["Gemma2DecoderLayer", "Gemma2RMSNorm"],
                max_memory=max_memory,
            )
            model = dispatch_model(model, device_map=device_map)
        elif device_map is not None:
            raise NotImplementedError("Write your own device map")

        return model



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
            self.model = LDLRewardModel27B.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                #attn_implementation="sdpa",
                device_map="auto",
                trust_remote_code=False
            )
            self.input_device = next(iter(self.model.parameters())).device.type
        else:
            print("CUDA not detected: Using CPU Inference")
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = LDLRewardModel27B.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                #attn_implementation="sdpa",
                device_map="cpu",
                trust_remote_code=False
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
                padding=True
            ).to(self.input_device)

            # Get model output
            outputs = self.model(**inputs)
            # Extract scalar scores and detach from computation graph
            scores = outputs.logits[0].squeeze().detach()

            # Handle single example case
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)

            # Move to CPU and convert to list
            batch_scores = scores.cpu().tolist()
            all_scores.extend(batch_scores)

            # Clean up to reduce memory usage
            del inputs, outputs, scores
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
                traceback.print(e.__traceback__)
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
