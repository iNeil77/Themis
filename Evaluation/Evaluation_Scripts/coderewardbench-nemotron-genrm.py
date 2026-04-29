# Invoked as:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python coderewardbench-nemotron-genrm.py "nvidia/Qwen3-Nemotron-32B-GenRM-Principle" --output "GenRM-Principle-vLLM" --use-aspect-prompts --max-model-len 32768 --tensor-parallel-size 4

"""
This script evaluates a generative reward model (GenRM) on the Code RewardBench dataset
using vLLM for high-throughput inference with continuous batching.

All chosen and rejected conversations are formatted upfront and submitted to vLLM
in a single `llm.generate()` call, leveraging continuous batching for maximum throughput.

The reward is derived from the logprobs of " Yes" vs " No" tokens at the
penultimate generation step.
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import argparse
import json
import os


# Principle prompts mapped from coding aspects
# These are used as the 'principle' role content in the chat template
PRINCIPLE_MAP = {
    'Full': (
        "helpfulness, harmlessness, functional correctness, memory efficiency, "
        "readability and maintainability, runtime efficiency, security hardness"
    ),
    'Functional_Correctness': "functional correctness",
    'Memory_Efficiency': "memory efficiency",
    'Readability_Maintainability': "readability and maintainability",
    'Runtime_Efficiency': "runtime efficiency",
    'Security_Hardness': "security hardness",
}

# Token IDs for " Yes" and " No" in the Qwen tokenizer
TOKEN_ID_YES = 7414
TOKEN_ID_NO = 2308

# Default logprob clamp value (matches original HF implementation)
# Applied when a token is not found in the top-k logprobs
LOGPROB_CLAMP = -50.0


class GenRMEvaluator:
    """CRB evaluator for Nemotron GenRM: principle-based Yes/No generative evaluation via vLLM."""

    def __init__(
            self,
            model_path,
            max_model_len=32768,
            use_aspect_prompts=False,
            max_new_tokens=16000,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            top_logprobs=50,
            trust_remote_code=True):
        """
        Initialize the generative reward model evaluator with vLLM.

        Args:
            model_path (str): Path to the GenRM model on HuggingFace or local path
            max_model_len (int): Maximum model context length for vLLM
            use_aspect_prompts (bool): Whether to use aspect-specific principle prompts
            max_new_tokens (int): Maximum new tokens for generation
            tensor_parallel_size (int): Number of GPUs for tensor parallelism
            gpu_memory_utilization (float): Fraction of GPU memory to use
            top_logprobs (int): Number of top logprobs to return per token.
                Must be large enough to capture " Yes" and " No" tokens.
                If these tokens fall outside top-k, their logprob is clamped
                to LOGPROB_CLAMP (-50). Use a higher value for more accuracy
                at the cost of some overhead.
            trust_remote_code (bool): Whether to trust remote code in model
        """
        self.use_aspect_prompts = use_aspect_prompts
        self.max_new_tokens = max_new_tokens
        self.top_logprobs = top_logprobs

        # Load tokenizer separately for chat template formatting
        print(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        # Initialize vLLM engine
        print(f"Initializing vLLM engine with tensor_parallel_size={tensor_parallel_size}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            dtype="bfloat16",
        )

        # Configure sampling parameters for reward extraction
        # We need logprobs at each generated token position to extract
        # the penultimate position's " Yes"/" No" logprobs
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,  # Greedy decoding for deterministic output
            logprobs=top_logprobs,
        )

        print(f"Aspect prompts enabled: {use_aspect_prompts}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Top logprobs per position: {top_logprobs}")

    def _get_principle(self, aspect=None):
        """Get the principle string for the given aspect."""
        if self.use_aspect_prompts and aspect and aspect in PRINCIPLE_MAP:
            return PRINCIPLE_MAP[aspect]
        return PRINCIPLE_MAP['Full']

    def format_conversation(self, prompt, response, aspect=None):
        """
        Format a prompt-response pair into a tokenized string using the
        GenRM chat template with the principle role.

        Args:
            prompt (str): The input prompt (user message)
            response (str): The model response (assistant message)
            aspect (str): The coding aspect for principle selection

        Returns:
            str: Formatted conversation string ready for vLLM
        """
        principle = self._get_principle(aspect)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
            {"role": "principle", "content": principle},
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted

    @staticmethod
    def extract_reward_from_output(output, top_logprobs_k):
        """
        Extract the reward score from a single vLLM CompletionOutput.

        The reward is computed from the penultimate token's logprobs:
          reward = logprob(" Yes") - logprob(" No")

        If " Yes" or " No" is not in the top-k logprobs for that position,
        its logprob is clamped to LOGPROB_CLAMP (matching the original HF
        implementation's behavior when tokens have very low probability).

        Args:
            output: A vLLM RequestOutput object
            top_logprobs_k: Number of top logprobs requested (for diagnostics)

        Returns:
            float: Reward score (score_yes - score_no)
        """
        completion = output.outputs[0]
        logprobs_list = completion.logprobs  # list[dict[int, Logprob]]

        if logprobs_list is None or len(logprobs_list) < 2:
            # Not enough generated tokens to extract penultimate logprobs
            print(
                f"Warning: Request {output.request_id} generated "
                f"{len(logprobs_list) if logprobs_list else 0} tokens, "
                f"need at least 2. Returning reward=0.0"
            )
            return 0.0

        # Get the penultimate position's logprobs (index -2)
        penultimate_logprobs = logprobs_list[-2]

        # Extract logprobs for " Yes" and " No" tokens
        # vLLM logprobs are already log-softmax values (normalized),
        # so they correspond directly to the original code's
        # (raw_logit - max_logit) computation
        if TOKEN_ID_YES in penultimate_logprobs:
            score_yes = penultimate_logprobs[TOKEN_ID_YES].logprob
        else:
            score_yes = LOGPROB_CLAMP

        if TOKEN_ID_NO in penultimate_logprobs:
            score_no = penultimate_logprobs[TOKEN_ID_NO].logprob
        else:
            score_no = LOGPROB_CLAMP

        reward = score_yes - score_no
        return reward

    def evaluate_dataset(
            self,
            dataset_name="iNeil77/CRB",
            config=None,
            split=None):
        """
        Evaluate the reward model on the CRB dataset.

        All conversations (chosen + rejected) are formatted upfront and
        submitted to vLLM in a single generate() call for maximum throughput
        via continuous batching.

        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            config (str): Configuration name for the dataset
            split (str): Optional split to evaluate on

        Returns:
            dict: Evaluation results
        """
        # --- Load dataset ---
        print(f"Loading dataset: {dataset_name}")
        if config:
            print(f"Using config: {config} and split: {split}")
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, "Full", split="Full")

        num_examples = len(dataset)
        print(f"Dataset loaded: {num_examples} examples")

        # --- Format all conversations upfront ---
        print("Formatting all conversations...")
        chosen_conversations = []
        rejected_conversations = []
        metadata = []  # Store metadata aligned with dataset indices

        for i in range(num_examples):
            example = dataset[i]
            prompt = example['prompt']
            chosen = example['chosen']
            rejected = example['rejected']
            aspect = example['aspect']

            chosen_conversations.append(
                self.format_conversation(prompt, chosen, aspect)
            )
            rejected_conversations.append(
                self.format_conversation(prompt, rejected, aspect)
            )
            metadata.append({
                'id': example['id'],
                'language': example['language'],
                'aspect': aspect,
                'subset': example['subset'],
            })

        # --- Submit all conversations to vLLM in one call ---
        # Interleave chosen and rejected so each pair is adjacent:
        #   [chosen_0, rejected_0, chosen_1, rejected_1, ...]
        # This makes it easy to pair results back together.
        all_conversations = []
        for chosen, rejected in zip(chosen_conversations, rejected_conversations):
            all_conversations.append(chosen)
            all_conversations.append(rejected)

        total_requests = len(all_conversations)
        print(
            f"Submitting {total_requests} requests to vLLM "
            f"({num_examples} chosen + {num_examples} rejected)..."
        )

        # Single generate() call — vLLM handles continuous batching internally
        all_outputs = self.llm.generate(all_conversations, self.sampling_params)

        print(f"Generation complete. Processing {len(all_outputs)} outputs...")

        # --- Extract rewards and compile results ---
        results = []
        detailed_scores = []

        for i in range(num_examples):
            chosen_output = all_outputs[2 * i]
            rejected_output = all_outputs[2 * i + 1]

            chosen_score = self.extract_reward_from_output(
                chosen_output, self.top_logprobs
            )
            rejected_score = self.extract_reward_from_output(
                rejected_output, self.top_logprobs
            )

            meta = metadata[i]
            result = {
                'chosen_score': chosen_score,
                'rejected_score': rejected_score,
                'correct': chosen_score > rejected_score,
                'score_diff': chosen_score - rejected_score,
                'id': meta['id'],
                'language': meta['language'],
                'aspect': meta['aspect'],
                'subset': meta['subset'],
            }
            results.append(result)
            detailed_scores.append({
                'id': meta['id'],
                'chosen_score': chosen_score,
                'rejected_score': rejected_score,
                'correct': chosen_score > rejected_score,
                'score_diff': chosen_score - rejected_score,
                'language': meta['language'],
                'aspect': meta['aspect'],
                'subset': meta['subset'],
            })

        compiled_results = self.compile_results(results)
        compiled_results['detailed_scores'] = detailed_scores

        return compiled_results

    def compile_results(self, results):
        """
        Compile evaluation results into a comprehensive report.

        Args:
            results (list): List of individual evaluation results

        Returns:
            dict: Compiled results with overall and breakdown statistics
        """
        df = pd.DataFrame(results)

        overall_accuracy = df['correct'].mean()
        total_examples = len(df)

        aspect_accuracy = df.groupby('aspect')['correct'].agg(
            ['mean', 'count']).round(4)
        aspect_accuracy.columns = ['accuracy', 'count']

        language_accuracy = df.groupby('language')['correct'].agg(
            ['mean', 'count']).round(4)
        language_accuracy.columns = ['accuracy', 'count']

        subset_accuracy = df.groupby('subset')['correct'].agg(
            ['mean', 'count']).round(4)
        subset_accuracy.columns = ['accuracy', 'count']

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

    def save_results(self, results, output_dir):
        """
        Save evaluation results to the specified output directory.

        Args:
            results (dict): Compiled results from evaluate_dataset
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed scores as parquet
        scores_df = pd.DataFrame(results['detailed_scores'])
        scores_df.set_index('id', inplace=True)
        scores_path = os.path.join(output_dir, 'scores.parquet')
        scores_df.to_parquet(scores_path, index=True)
        print(f"Detailed scores saved to: {scores_path}")

        # Save results as JSON (excluding raw_results and detailed_scores)
        results_to_save = {
            k: v for k, v in results.items()
            if k not in ['raw_results', 'detailed_scores']
        }
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Results saved to: {results_path}")

    def print_results(self, results):
        """
        Print formatted evaluation results.

        Args:
            results (dict): Compiled results from evaluate_dataset
        """
        print("\n" + "=" * 60)
        print("GENERATIVE REWARD MODEL EVALUATION RESULTS (vLLM)")
        print("=" * 60)

        overall = results['overall']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Examples: {overall['total_examples']}")
        print(f"  Correct Predictions: {overall['correct_predictions']}")
        print(f"  Accuracy: {overall['accuracy']:.2%}")
        print(f"  Mean Score Difference: {overall['mean_score_difference']:.4f}")
        print(f"  Median Score Difference: {overall['median_score_difference']:.4f}")
        print(f"  Score Difference Std: {overall['std_score_difference']:.4f}")

        print(f"\nACCURACY BY ASPECT:")
        for aspect, stats in results['by_aspect'].items():
            print(f"  {aspect:<25}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print(f"\nACCURACY BY LANGUAGE:")
        for language, stats in results['by_language'].items():
            print(f"  {language:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print(f"\nACCURACY BY SUBSET:")
        for subset, stats in results['by_subset'].items():
            print(f"  {subset:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a generative reward model (GenRM) on CRB dataset using vLLM"
    )
    parser.add_argument(
        "model_path",
        help="Path to the generative reward model"
    )
    parser.add_argument(
        "--dataset",
        default="iNeil77/code-reward-bench",
        help="Dataset name"
    )
    parser.add_argument(
        "--config",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--split",
        help="Specific split to evaluate"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory name to save results"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length for vLLM"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16000,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.0-1.0)"
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help=(
            "Number of top logprobs to return per generated token. "
            "Higher values ensure ' Yes' and ' No' are captured but "
            "increase overhead. If tokens fall outside top-k, their "
            "logprob is clamped to -50."
        )
    )
    parser.add_argument(
        "--use-aspect-prompts",
        action="store_true",
        default=False,
        help="Use aspect-specific principle prompts instead of the full principle"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading model"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = GenRMEvaluator(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        use_aspect_prompts=args.use_aspect_prompts,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        top_logprobs=args.top_logprobs,
        trust_remote_code=args.trust_remote_code,
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.dataset,
        args.config,
        args.split,
    )

    # Print results
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()