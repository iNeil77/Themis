# CUDA_VISIBLE_DEVICES=0,1,2,3 python coderewardbench-lmunit.py "ContextualAI/LMUnit-qwen2.5-72b" --output "LMUnit-Aspect" --use-aspect-prompts --batch-size 256 --max-length 4096 --tensor-parallel-size 4 --max-new-tokens 2048


"""
This script evaluates a generative reward model (LMUnit-style) on the Code RewardBench dataset
using vLLM for fast batched inference, calculating accuracy across different aspects and
programming languages.
"""

import json
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset


ASPECT_UNIT_TEST_MAP = {
    'Full': "Does the response comprehensively address the prompt with high quality, correctness, readability, efficiency, and security?",
    'Functional_Correctness': "Does the response correctly implement the requested functionality, handle edge cases properly, and produce the expected outputs?",
    'Memory_Efficiency': "Does the response use memory-efficient data structures and practices, minimizing memory usage and avoiding memory leaks?",
    'Readability_Maintainability': "Does the response use clear naming, consistent formatting, appropriate modularization, and sufficient documentation?",
    'Runtime_Efficiency': "Does the response use efficient algorithms and data structures, minimizing time complexity and avoiding unnecessary computations?",
    'Security_Hardness': "Does the response follow secure coding practices, including input validation, output encoding, and proper error handling?"
}

DEFAULT_UNIT_TEST = "Does the response correctly and helpfully address the prompt?"


class GenerativeRewardModelEvaluator:
    """CRB evaluator for LMUnit: unit-test-style generative scoring via vLLM."""

    def __init__(self, model_path, max_length, max_new_tokens=32, use_aspect_prompts=False, tensor_parallel_size=1):
        print(f"Aspect-specific unit tests enabled: {use_aspect_prompts}")
        self.use_aspect_prompts = use_aspect_prompts
        self.max_length = max_length

        print(f"Loading model and tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=model_path,
            max_model_len=max_length,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        print(f"Using max sequence length: {self.max_length}")
        print(f"Using max new tokens: {max_new_tokens}")

    def format_input(self, prompt, response, aspect=None):
        """
        Format prompt and response into the LMUnit input format.
        """
        if self.use_aspect_prompts and aspect and aspect in ASPECT_UNIT_TEST_MAP:
            unit_test = ASPECT_UNIT_TEST_MAP[aspect]
        else:
            unit_test = DEFAULT_UNIT_TEST

        content = f"Query: {prompt}\n\nResponse: {response}\n\nUnit Test: {unit_test}"
        messages = [{"role": "user", "content": content}]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

    def parse_score(self, text):
        """
        Parse the score from model output like '{"score": 5}<|im_end|>'.
        Returns (score: float, parse_failed: bool). On failure, score is 0.0.
        """
        try:
            match = re.search(r'\{.*?"score".*?\}', text.strip(), re.DOTALL)
            if match:
                data = json.loads(match.group())
                return float(data["score"]), False
        except Exception:
            pass
        # Fallback: look for a bare number
        match = re.search(r'\b([1-5])\b', text)
        if match:
            return float(match.group(1)), False
        print(f"Warning: Could not parse score from: {repr(text)}")
        return 0.0, True

    def get_reward_scores(self, formatted_inputs):
        """
        Get reward scores for a list of pre-formatted input strings using vLLM.
        Returns (scores: list[float], parse_failures: int).
        """
        outputs = self.llm.generate(formatted_inputs, self.sampling_params)
        scores = []
        parse_failures = 0
        for output in outputs:
            score, failed = self.parse_score(output.outputs[0].text)
            scores.append(score)
            if failed:
                parse_failures += 1
        return scores, parse_failures

    def evaluate_dataset(self, dataset_name="project-themis/Themis-CodeRewardBench", config=None, split=None, batch_size=256):
        print(f"Loading dataset: {dataset_name}")
        if config:
            print(f"Using config: {config} and split: {split}")
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, "Full", split="Full")

        print(f"Evaluating on {len(dataset)} examples.")

        all_ids = dataset['id']
        all_languages = dataset['language']
        all_aspects = dataset['aspect']
        all_subsets = dataset['subset']
        all_prompts = dataset['prompt']
        all_chosen = dataset['chosen']
        all_rejected = dataset['rejected']

        print("Formatting inputs...")
        chosen_inputs = [
            self.format_input(p, c, a)
            for p, c, a in tqdm(zip(all_prompts, all_chosen, all_aspects),
                                 total=len(all_prompts), desc="Formatting chosen")
        ]
        rejected_inputs = [
            self.format_input(p, r, a)
            for p, r, a in tqdm(zip(all_prompts, all_rejected, all_aspects),
                                 total=len(all_prompts), desc="Formatting rejected")
        ]

        # Interleave chosen and rejected so vLLM schedules them together
        interleaved_inputs = [x for pair in zip(chosen_inputs, rejected_inputs) for x in pair]

        print(f"Running vLLM inference on {len(interleaved_inputs)} total sequences...")
        all_scores, parse_failures = self.get_reward_scores(interleaved_inputs)

        # De-interleave scores
        chosen_scores = all_scores[0::2]
        rejected_scores = all_scores[1::2]

        results = []
        detailed_scores = []
        for i, (cs, rs) in enumerate(zip(chosen_scores, rejected_scores)):
            result = {
                'chosen_score': cs,
                'rejected_score': rs,
                'correct': cs > rs,
                'score_diff': cs - rs,
                'id': all_ids[i],
                'language': all_languages[i],
                'aspect': all_aspects[i],
                'subset': all_subsets[i]
            }
            results.append(result)
            detailed_scores.append({
                'id': all_ids[i],
                'chosen_score': cs,
                'rejected_score': rs,
                'correct': cs > rs,
                'score_diff': cs - rs,
                'language': all_languages[i],
                'aspect': all_aspects[i],
                'subset': all_subsets[i]
            })

        compiled_results = self.compile_results(results, parse_failures)
        compiled_results['detailed_scores'] = detailed_scores
        return compiled_results

    def compile_results(self, results, parse_failures=0):
        df = pd.DataFrame(results)

        overall_accuracy = df['correct'].mean()
        total_examples = len(df)
        total_sequences = total_examples * 2  # chosen + rejected

        aspect_accuracy = df.groupby('aspect')['correct'].agg(['mean', 'count']).round(4)
        aspect_accuracy.columns = ['accuracy', 'count']

        language_accuracy = df.groupby('language')['correct'].agg(['mean', 'count']).round(4)
        language_accuracy.columns = ['accuracy', 'count']

        subset_accuracy = df.groupby('subset')['correct'].agg(['mean', 'count']).round(4)
        subset_accuracy.columns = ['accuracy', 'count']

        score_diffs = df['score_diff']

        return {
            'overall': {
                'accuracy': round(overall_accuracy, 4),
                'total_examples': total_examples,
                'correct_predictions': int(df['correct'].sum()),
                'parse_failures': parse_failures,
                'parse_failure_rate': round(parse_failures / total_sequences, 4) if total_sequences > 0 else 0.0,
                'mean_score_difference': round(score_diffs.mean(), 4),
                'median_score_difference': round(score_diffs.median(), 4),
                'std_score_difference': round(score_diffs.std(), 4)
            },
            'by_aspect': aspect_accuracy.to_dict('index'),
            'by_language': language_accuracy.to_dict('index'),
            'by_subset': subset_accuracy.to_dict('index'),
            'raw_results': results
        }

    def save_results(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        scores_df = pd.DataFrame(results['detailed_scores'])
        scores_df.set_index('id', inplace=True)
        scores_path = os.path.join(output_dir, 'scores.parquet')
        scores_df.to_parquet(scores_path, index=True)
        print(f"Detailed scores saved to: {scores_path}")

        results_to_save = {k: v for k, v in results.items() if k not in ['raw_results', 'detailed_scores']}
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Results saved to: {results_path}")

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("GENERATIVE REWARD MODEL EVALUATION RESULTS")
        print("=" * 60)

        overall = results['overall']
        total_sequences = overall['total_examples'] * 2
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Examples:      {overall['total_examples']}")
        print(f"  Correct Predictions: {overall['correct_predictions']}")
        print(f"  Accuracy:            {overall['accuracy']:.2%}")
        print(f"  Parse Failures:      {overall['parse_failures']} / {total_sequences} ({overall['parse_failure_rate']:.2%})")
        print(f"  Mean Score Diff:     {overall['mean_score_difference']:.4f}")
        print(f"  Median Score Diff:   {overall['median_score_difference']:.4f}")
        print(f"  Score Diff Std:      {overall['std_score_difference']:.4f}")

        print(f"\nACCURACY BY ASPECT:")
        for aspect, stats in results['by_aspect'].items():
            print(f"  {aspect:<30}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print(f"\nACCURACY BY LANGUAGE:")
        for language, stats in results['by_language'].items():
            print(f"  {language:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print(f"\nACCURACY BY SUBSET:")
        for subset, stats in results['by_subset'].items():
            print(f"  {subset:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative reward model on CRB dataset using vLLM")
    parser.add_argument("model_path", help="Path to the generative reward model")
    parser.add_argument("--dataset", default="project-themis/Themis-CodeRewardBench", help="Dataset name")
    parser.add_argument("--config", help="Dataset configuration name")
    parser.add_argument("--split", help="Specific split to evaluate")
    parser.add_argument("--output", required=True, help="Output directory name to save results")
    parser.add_argument("--max-length", type=int, default=4096, help="Max input token length")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens to generate per sequence")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Hint for vLLM's internal scheduling (not used directly; kept for CLI compatibility)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--use-aspect-prompts", action="store_true", default=False,
                        help="Use aspect-specific unit tests")
    args = parser.parse_args()

    evaluator = GenerativeRewardModelEvaluator(
        args.model_path,
        args.max_length,
        args.max_new_tokens,
        args.use_aspect_prompts,
        args.tensor_parallel_size
    )

    results = evaluator.evaluate_dataset(
        args.dataset,
        args.config,
        args.split,
        args.batch_size
    )

    evaluator.print_results(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()