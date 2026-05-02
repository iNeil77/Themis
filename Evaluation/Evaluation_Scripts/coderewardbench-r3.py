# Invoked as:
# CUDA_VISIBLE_DEVICES=6,7 python coderewardbench-r3.py "rubricreward/R3-Qwen3-14B-14k" --output "R3-Qwen3-14B-14k" --use-system-prompts --use-aspect-prompts --batch-size 32 --max-model-len 10000 --tensor-parallel-size 2

"""
This script evaluates a generative reward model (R3-style) on the Code RewardBench
dataset, calculating accuracy across different aspects and programming languages.

Unlike scalar reward models, R3 models generate textual evaluations with reasoning
and scores. This script uses vLLM for efficient batched inference and parses the
generated output to extract numerical scores for pairwise comparison.
"""

import re
import json
import os
import argparse
from typing import Optional

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ──────────────────────────────────────────────────────────────────────────────
# Rubric templates for different coding aspects (used as evaluation criteria)
# ──────────────────────────────────────────────────────────────────────────────

RUBRIC_MAP = {

    'Full': (
        "1. **Helpfulness**: How well does the response address the prompt? "
        "Does it provide a complete and thorough answer?\n"
        "2. **Harmlessness**: Does the response contain any harmful or inappropriate content?\n"
        "3. **Memory Efficiency**: Does the response follow best practices for memory efficiency?\n"
        "4. **Functional Correctness**: Does the response correctly implement algorithms, "
        "adhere to specifications, and handle edge cases?\n"
        "5. **Readability and Maintainability**: Does the response use clear naming, "
        "consistent formatting, modular code, and appropriate comments?\n"
        "6. **Runtime Efficiency**: Does the response use efficient algorithms and data structures, "
        "minimize time complexity, and avoid unnecessary computations?\n"
        "7. **Security Hardness**: Does the response follow best practices for security, "
        "including input validation, error handling, and secure coding practices?"
    ),

    'Functional_Correctness': (
        "1. **Helpfulness**: How well does the response address the prompt?\n"
        "2. **Harmlessness**: Does the response contain any harmful or inappropriate content?\n"
        "3. **Functional Correctness**: Does the response correctly implement algorithms, "
        "adhere to specifications, and properly handle edge cases?"
    ),

    'Memory_Efficiency': (
        "1. **Helpfulness**: How well does the response address the prompt?\n"
        "2. **Harmlessness**: Does the response contain any harmful or inappropriate content?\n"
        "3. **Memory Efficiency**: Does the response follow best practices for memory efficiency, "
        "using efficient data structures, minimizing memory usage, and avoiding memory leaks?"
    ),

    'Readability_Maintainability': (
        "1. **Helpfulness**: How well does the response address the prompt?\n"
        "2. **Harmlessness**: Does the response contain any harmful or inappropriate content?\n"
        "3. **Readability and Maintainability**: Does the response use clear naming, "
        "consistent formatting, modular code, and appropriate documentation?"
    ),

    'Runtime_Efficiency': (
        "1. **Helpfulness**: How well does the response address the prompt?\n"
        "2. **Harmlessness**: Does the response contain any harmful or inappropriate content?\n"
        "3. **Runtime Efficiency**: Does the response use efficient algorithms and data structures, "
        "minimize time complexity, and avoid unnecessary computations?"
    ),

    'Security_Hardness': (
        "1. **Helpfulness**: How well does the response address the prompt?\n"
        "2. **Harmlessness**: Does the response contain any harmful or inappropriate content?\n"
        "3. **Security Hardness**: Does the response follow best practices for security, "
        "including input validation, output encoding, error handling, and secure coding?"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction for R3-style evaluation
# ──────────────────────────────────────────────────────────────────────────────

def build_pointwise_prompt(
    prompt: str,
    response: str,
    aspect: Optional[str] = None,
    use_system_prompts: bool = False,
    use_aspect_prompts: bool = False,
) -> str:
    """
    Build an R3-style point-wise evaluation prompt.

    The R3 model expects a structured prompt with:
    - TASK description
    - INPUT (the original coding prompt)
    - RESPONSE (the code response to evaluate)
    - EVALUATION RUBRIC (scoring criteria)
    - OUTPUT FORMAT

    Returns:
        str: The formatted evaluation prompt for the R3 model.
    """
    # Select the appropriate rubric
    if use_system_prompts and aspect and aspect in RUBRIC_MAP:
        if use_aspect_prompts:
            rubric_text = RUBRIC_MAP[aspect]
        else:
            rubric_text = RUBRIC_MAP['Full']
    else:
        rubric_text = RUBRIC_MAP['Full']

    eval_prompt = (
        "Evaluate the response based on the given task, input, response, and "
        "evaluation rubric. Provide a fair and detailed assessment following "
        "the rubric.\n\n"
        "### TASK\n"
        "Evaluate the quality of the following code response to a programming task. "
        "Consider the evaluation criteria specified in the rubric.\n\n"
        f"### INPUT\n{prompt}\n\n"
        f"### RESPONSE\n{response}\n\n"
        f"### EVALUATION RUBRIC\n{rubric_text}\n\n"
        "### OUTPUT FORMAT\n"
        "Return a JSON response in the following format:\n"
        '{\n'
        '  "explanation": "Explanation of why the response received this score",\n'
        '  "score": <integer score from 1 to 5>\n'
        '}\n\n'
        "### EVALUATION"
    )

    return eval_prompt


def extract_score_from_output(text: str) -> Optional[float]:
    """
    Extract a numerical score from the R3 model's generated output.

    The model may output:
    1. A JSON object with a "score" field
    2. A score mentioned in the text (e.g., "Score: 4")
    3. A number in various formats

    Returns:
        float or None: The extracted score, or None if parsing failed.
    """
    # Strip thinking tokens if present (Qwen3 <think>...</think>)
    # The content after </think> is the actual response
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Strategy 1: Try to parse as JSON
    # Look for JSON blocks (possibly wrapped in ```json ... ```)
    json_patterns = [
        r'```json\s*(\{[^`]*?\})\s*```',
        r'```\s*(\{[^`]*?\})\s*```',
        r'(\{[^{}]*"score"[^{}]*\})',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if 'score' in data:
                    score_val = data['score']
                    if isinstance(score_val, (int, float)):
                        return float(score_val)
                    # Score might be a string like "4" or "4/5"
                    if isinstance(score_val, str):
                        num_match = re.search(r'(\d+(?:\.\d+)?)', score_val)
                        if num_match:
                            return float(num_match.group(1))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    # Strategy 2: Look for "score": <number> pattern (even in malformed JSON)
    score_json_pattern = re.search(
        r'"score"\s*:\s*(\d+(?:\.\d+)?)', text
    )
    if score_json_pattern:
        return float(score_json_pattern.group(1))

    # Strategy 3: Look for common score patterns in text
    score_patterns = [
        r'\bscore\s*(?:is|:)\s*(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?',
        r'\b(\d+(?:\.\d+)?)\s*/\s*5\b',
        r'\[RESULT\]\s*(\d+(?:\.\d+)?)',
        r'(?:final|overall)\s+score\s*(?:is|:)\s*(\d+(?:\.\d+)?)',
        r'Rating:\s*(\d+(?:\.\d+)?)',
    ]
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Strategy 4: Last resort - find the last standalone number
    # (often the score appears at the very end)
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if numbers:
        last_num = float(numbers[-1])
        if 1 <= last_num <= 5:
            return last_num

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluator class
# ──────────────────────────────────────────────────────────────────────────────

class R3RewardModelEvaluator:
    """CRB evaluator for R3: rubric-based pointwise generative scoring via vLLM."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 10000,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.9,
        enable_thinking: bool = True,
        use_system_prompts: bool = False,
        use_aspect_prompts: bool = False,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 8192,
    ):
        """
        Initialize the R3 generative reward model evaluator.

        Args:
            model_path: HuggingFace model path or local path
            max_model_len: Maximum model context length for vLLM
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            enable_thinking: Whether to enable Qwen3 thinking mode
            use_system_prompts: Whether to use aspect-specific rubrics
            use_aspect_prompts: Whether to use per-aspect rubrics (vs Full)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            min_p: Min-p sampling parameter
            max_tokens: Max tokens to generate
        """
        self.model_path = model_path
        self.use_system_prompts = use_system_prompts
        self.use_aspect_prompts = use_aspect_prompts
        self.enable_thinking = enable_thinking

        print(f"System prompts enabled: {use_system_prompts}")
        print(f"Aspect prompts enabled: {use_aspect_prompts}")
        print(f"Thinking mode enabled: {enable_thinking}")

        # Load tokenizer
        print(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize vLLM engine
        print(f"Loading vLLM model from: {model_path}")
        print(f"  tensor_parallel_size={tensor_parallel_size}")
        print(f"  max_model_len={max_model_len}")
        print(f"  gpu_memory_utilization={gpu_memory_utilization}")

        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens,
        )

        print("Model loaded successfully.")

    def format_messages(
        self,
        prompt: str,
        response: str,
        aspect: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """
        Format a single evaluation as a chat message list.

        Returns:
            list: Chat messages for the tokenizer's chat template.
        """
        eval_prompt = build_pointwise_prompt(
            prompt=prompt,
            response=response,
            aspect=aspect,
            use_system_prompts=self.use_system_prompts,
            use_aspect_prompts=self.use_aspect_prompts,
        )

        messages = [{"role": "user", "content": eval_prompt}]
        return messages

    def prepare_prompts(
        self,
        prompts: list[str],
        responses: list[str],
        aspects: list[str],
    ) -> list[str]:
        """
        Prepare a batch of formatted prompt strings for vLLM generation.

        Returns:
            list[str]: Tokenizer-formatted prompt strings.
        """
        formatted_texts = []
        for prompt, response, aspect in zip(prompts, responses, aspects):
            messages = self.format_messages(prompt, response, aspect)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            formatted_texts.append(text)
        return formatted_texts

    def get_scores_batch(
        self,
        formatted_texts: list[str],
    ) -> tuple[list[Optional[float]], list[str]]:
        """
        Generate evaluations for a batch of formatted prompts and extract scores.

        Args:
            formatted_texts: List of tokenizer-formatted prompt strings.

        Returns:
            tuple: (list of extracted scores, list of raw generated texts)
        """
        # vLLM handles batching internally and efficiently
        outputs = self.llm.generate(formatted_texts, self.sampling_params)

        scores = []
        raw_texts = []
        for output in outputs:
            generated_text = output.outputs[0].text
            raw_texts.append(generated_text)
            score = extract_score_from_output(generated_text)
            scores.append(score)

        return scores, raw_texts

    def evaluate_batch(
        self,
        prompts: list[str],
        chosen_responses: list[str],
        rejected_responses: list[str],
        aspects: list[str],
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Evaluate a batch of prompt-response pairs by scoring chosen and
        rejected responses independently, then comparing.

        Args:
            prompts: List of input prompts
            chosen_responses: List of preferred responses
            rejected_responses: List of dispreferred responses
            aspects: List of coding aspects for each example
            batch_size: Batch size for vLLM (used for sub-batching if needed)

        Returns:
            list: List of evaluation result dicts.
        """
        # Prepare all prompts (chosen + rejected interleaved for efficiency)
        chosen_texts = self.prepare_prompts(prompts, chosen_responses, aspects)
        rejected_texts = self.prepare_prompts(prompts, rejected_responses, aspects)

        # Combine all texts for a single vLLM call (more efficient)
        all_texts = chosen_texts + rejected_texts
        n = len(chosen_texts)

        # Generate all scores in one batch
        all_scores, all_raw = self.get_scores_batch(all_texts)

        chosen_scores = all_scores[:n]
        rejected_scores = all_scores[n:]
        chosen_raw = all_raw[:n]
        rejected_raw = all_raw[n:]

        # Compile results
        results = []
        for i in range(n):
            c_score = chosen_scores[i]
            r_score = rejected_scores[i]

            parse_failed = (c_score is None) or (r_score is None)

            if parse_failed:
                # If either score could not be parsed, mark as failure
                correct = False
                score_diff = 0.0
            else:
                correct = c_score > r_score
                score_diff = c_score - r_score

            results.append({
                'chosen_score': c_score,
                'rejected_score': r_score,
                'correct': correct,
                'score_diff': score_diff,
                'parse_failed': parse_failed,
                'chosen_parse_failed': c_score is None,
                'rejected_parse_failed': r_score is None,
            })

        return results

    def evaluate_dataset(
        self,
        dataset_name: str = "project-themis/Themis-CodeRewardBench",
        config: Optional[str] = None,
        split: Optional[str] = None,
        batch_size: int = 32,
    ) -> dict:
        """
        Evaluate the R3 reward model on the CRB dataset.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            config: Configuration name for the dataset
            split: Optional split to evaluate on
            batch_size: Batch size for generation

        Returns:
            dict: Evaluation results
        """
        print(f"Loading dataset: {dataset_name}")
        if config:
            print(f"Using config: {config} and split: {split}")

        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, "Full", split="Full")

        print(f"Evaluating on {len(dataset)} examples with batch size {batch_size}.")

        results = []
        detailed_scores = []
        parse_failures = 0

        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]

            try:
                # Extract batch data
                prompts = batch['prompt'] if isinstance(batch['prompt'], list) else [batch['prompt']]
                chosen_responses = batch['chosen'] if isinstance(batch['chosen'], list) else [batch['chosen']]
                rejected_responses = batch['rejected'] if isinstance(batch['rejected'], list) else [batch['rejected']]

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

                batch_results = self.evaluate_batch(
                    prompts, chosen_responses, rejected_responses,
                    aspects, batch_size
                )

                for j, result in enumerate(batch_results):
                    result.update({
                        'id': ids[j],
                        'language': languages[j],
                        'aspect': aspects[j],
                        'subset': subsets[j],
                    })

                    if result['parse_failed']:
                        parse_failures += 1

                    detailed_scores.append({
                        'id': ids[j],
                        'chosen_score': result['chosen_score'],
                        'rejected_score': result['rejected_score'],
                        'correct': result['correct'],
                        'score_diff': result['score_diff'],
                        'parse_failed': result['parse_failed'],
                        'chosen_parse_failed': result['chosen_parse_failed'],
                        'rejected_parse_failed': result['rejected_parse_failed'],
                        'language': languages[j],
                        'aspect': aspects[j],
                        'subset': subsets[j],
                    })

                results.extend(batch_results)

            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nTotal parse failures: {parse_failures} / {len(results)} "
              f"({parse_failures / max(len(results), 1) * 100:.1f}%)")

        compiled_results = self.compile_results(results)
        compiled_results['detailed_scores'] = detailed_scores
        compiled_results['parse_failures'] = parse_failures

        return compiled_results

    def compile_results(self, results: list[dict]) -> dict:
        """Compile evaluation results into a comprehensive report."""
        df = pd.DataFrame(results)

        overall_accuracy = df['correct'].mean()
        total_examples = len(df)

        aspect_accuracy = df.groupby('aspect')['correct'].agg(['mean', 'count']).round(4)
        aspect_accuracy.columns = ['accuracy', 'count']

        language_accuracy = df.groupby('language')['correct'].agg(['mean', 'count']).round(4)
        language_accuracy.columns = ['accuracy', 'count']

        subset_accuracy = df.groupby('subset')['correct'].agg(['mean', 'count']).round(4)
        subset_accuracy.columns = ['accuracy', 'count']

        # Score diff stats (only for successfully parsed pairs)
        valid_mask = ~df['parse_failed']
        valid_df = df[valid_mask]

        if len(valid_df) > 0:
            mean_score_diff = valid_df['score_diff'].mean()
            median_score_diff = valid_df['score_diff'].median()
            std_score_diff = valid_df['score_diff'].std()
        else:
            mean_score_diff = 0.0
            median_score_diff = 0.0
            std_score_diff = 0.0

        # Parse failure stats
        total_parse_failures = int(df['parse_failed'].sum())

        # Accuracy excluding parse failures (among valid pairs only)
        if len(valid_df) > 0:
            accuracy_valid_only = valid_df['correct'].mean()
        else:
            accuracy_valid_only = 0.0

        compiled_results = {
            'overall': {
                'accuracy': round(overall_accuracy, 4),
                'accuracy_valid_only': round(accuracy_valid_only, 4),
                'total_examples': total_examples,
                'correct_predictions': int(df['correct'].sum()),
                'mean_score_difference': round(mean_score_diff, 4),
                'median_score_difference': round(median_score_diff, 4),
                'std_score_difference': round(std_score_diff, 4),
                'valid_pairs': int(valid_mask.sum()),
                'total_parse_failures': total_parse_failures,
                'parse_failure_rate': round(total_parse_failures / max(total_examples, 1), 4),
            },
            'by_aspect': aspect_accuracy.to_dict('index'),
            'by_language': language_accuracy.to_dict('index'),
            'by_subset': subset_accuracy.to_dict('index'),
            'raw_results': results,
        }

        return compiled_results

    def save_results(self, results: dict, output_dir: str):
        """Save evaluation results to the specified output directory."""
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

    def print_results(self, results: dict):
        """Print formatted evaluation results."""
        print("\n" + "=" * 60)
        print("R3 GENERATIVE REWARD MODEL EVALUATION RESULTS")
        print("=" * 60)

        overall = results['overall']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Examples: {overall['total_examples']}")
        print(f"  Correct Predictions: {overall['correct_predictions']}")
        print(f"  Accuracy: {overall['accuracy']:.2%}")
        print(f"  Accuracy (valid only): {overall['accuracy_valid_only']:.2%}")
        print(f"  Valid Pairs (both parsed): {overall['valid_pairs']}")
        print(f"  Parse Failures: {overall['total_parse_failures']} ({overall['parse_failure_rate']:.2%})")
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate R3 generative reward model on CRB dataset"
    )
    parser.add_argument(
        "model_path",
        help="Path to the R3 reward model (e.g., rubricreward/R3-Qwen3-14B-14k)"
    )
    parser.add_argument(
        "--dataset",
        default="project-themis/Themis-CodeRewardBench",
        help="Dataset name"
    )
    parser.add_argument("--config", help="Dataset configuration name")
    parser.add_argument("--split", help="Specific subset to evaluate")
    parser.add_argument(
        "--output", required=True,
        help="Output directory name to save results"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=10000,
        help="Maximum model context length for vLLM"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for evaluation (dataset-level batching)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=2,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory to use"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--min-p", type=float, default=0.0,
        help="Min-p sampling parameter"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum tokens to generate per evaluation"
    )
    parser.add_argument(
        "--no-thinking", action="store_true", default=False,
        help="Disable Qwen3 thinking mode"
    )
    parser.add_argument(
        "--use-system-prompts", action="store_true", default=False,
        help="Use aspect-specific rubrics in evaluation prompts"
    )
    parser.add_argument(
        "--use-aspect-prompts", action="store_true", default=False,
        help="Use per-aspect rubrics (otherwise uses Full rubric)"
    )

    args = parser.parse_args()

    evaluator = R3RewardModelEvaluator(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=not args.no_thinking,
        use_system_prompts=args.use_system_prompts,
        use_aspect_prompts=args.use_aspect_prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
    )

    results = evaluator.evaluate_dataset(
        args.dataset,
        args.config,
        args.split,
        args.batch_size,
    )

    evaluator.print_results(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()