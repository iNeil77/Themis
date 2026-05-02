# Invoked as:
# CUDA_VISIBLE_DEVICES=0 python coderewardbench-cerm-vllm.py "PKU-ONELab/CE-RM-4B" --output "CE-RM-4B" --batch-size 32 --tensor-parallel-size 1

"""
This script evaluates a generative criteria-based reward model (CE-RM) on the
Code RewardBench dataset using vLLM for fast batched inference.

Unlike scalar reward models, CE-RM uses a two-turn conversation per response:
  Turn 1: Generate evaluation criteria for the query
  Turn 2: Score the response using those criteria (overall score parsed from \\boxed{})

vLLM adaptation:
  - Uses vllm.LLM + SamplingParams instead of AutoModelForCausalLM + model.generate()
  - Criteria and evaluation turns are each dispatched as a single batched llm.chat() call,
    processing all examples in the batch simultaneously rather than sequentially.
  - Multi-GPU support via --tensor-parallel-size (replaces device_map="auto").
"""

import re
import json
import os
import argparse

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from vllm import LLM, SamplingParams


# ── Prompts ────────────────────────────────────────────────────────────────────

CRITERIA_PROMPT = """Your task is to produce a minimal set of criteria for evaluating the quality of potential responses to the user query given below.

Begin by carefully analyzing the query to fully understand the user's intent and requirements, and then take into account all common and tangible factors that can indicate the response quality.

From these considerations, derive the final evaluation criteria list, which **must adhere to the following requirements:**

- Each criterion should consist of a concise term as well as its unambiguous description.
- The number of criteria is not necessarily the more the better; Fewer yet comprehensive is more desired.
- The criteria should be sufficient and complete, ensuring that no essential aspects or key signals of response quality are omitted.
- The criteria should be necessary and non-overlapping, with each one indispensable, distinct in perspective, and strictly orthogonal to others.

Provide the relevant analysis first, followed by the numbered list of criteria between [Start of Criteria] and [End of Criteria], with one criterion per line and the more important ones coming first.

Below is the user query:

[Start of Query]
{query}
[End of Query]
"""

EVALUATION_PROMPT = """Now that you have a response to the previous user query, your new task is to evaluate it using the criteria list you have produced.

For each criterion, focus on its concerns and carefully evaluate the corresponding specific quality of the response, providing the detailed analysis as well as relevant arguments, followed by the corresponding quality score from 0 to 5 within $\\boxed{{}}$.

Moreover, if the response demonstrates strengths or weaknesses beyond the scope of your criteria list, introduce an additional criterion titled \"Other Point(s),\" discussing them and considering them as bonus points or deductions as appropriate.

Finally, based on the analyses of these criteria, including their relative importance and scores, **conduct a comprehensive evaluation of the response's overall quality with sufficient and explicit evidence**, and then provide a corresponding overall quality score from 0 to 10 within $\\boxed{{}}$.

Use integers or half-point increments for all scores, with higher numbers representing higher quality.

Below is the response:

[Start of Response]
{response}
[End of Response]
"""


# ── Score parsing ──────────────────────────────────────────────────────────────

def parse_overall_score(evaluation_text: str) -> float | None:
    """
    Parse the final overall score (0–10) from the model's evaluation output.

    The model emits one \\boxed{} per criterion (0–5) and a final \\boxed{}
    for the overall score (0–10). We take the last match.

    Args:
        evaluation_text: Raw decoded text from the evaluation turn.

    Returns:
        The overall score as a float, or None if parsing fails.
    """
    matches = re.findall(r'\\boxed\{(\d+(?:\.5)?)\}', evaluation_text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


# ── Evaluator ──────────────────────────────────────────────────────────────────

class CERMEvaluator:
    """CRB evaluator for CE-RM: two-turn generative evaluation with criteria generation and boxed score parsing."""

    def __init__(
            self,
            model_path: str,
            max_new_tokens_criteria: int = 4096,
            max_new_tokens_evaluation: int = 8192,
            tensor_parallel_size: int = 1,
            max_model_len: int | None = None,
        ):
        """
        Initialize the CE-RM evaluator backed by vLLM.

        Args:
            model_path: HuggingFace model ID or local path.
            max_new_tokens_criteria: Token budget for criteria generation.
            max_new_tokens_evaluation: Token budget for response evaluation.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            max_model_len: Optional cap on the model's KV-cache context length.
                           Useful for reducing memory when batching long sequences.
        """
        print(f"Loading model with vLLM: {model_path}")
        print(f"  tensor_parallel_size : {tensor_parallel_size}")
        print(f"  max_model_len        : {max_model_len or 'model default'}")

        llm_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="auto",
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)

        # Greedy decoding (temperature=0 matches the original script)
        self.sampling_criteria = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens_criteria,
        )
        self.sampling_evaluation = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens_evaluation,
        )

        self.max_new_tokens_criteria   = max_new_tokens_criteria
        self.max_new_tokens_evaluation = max_new_tokens_evaluation


    def _generate_batch(
            self,
            conversations: list[list[dict]],
            sampling_params: SamplingParams,
        ) -> list[str]:
        """
        Run a single batched generation pass over a list of conversations.

        Uses llm.chat() which applies the model's chat template internally,
        mirroring tokenizer.apply_chat_template(..., add_generation_prompt=True).

        Args:
            conversations: List of conversation message lists, each in
                           [{"role": ..., "content": ...}, ...] format.
            sampling_params: vLLM SamplingParams for this generation pass.

        Returns:
            List of decoded output strings, one per input conversation,
            in the same order.
        """
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            # Mirrors enable_thinking=False from the original script.
            # Passed through to apply_chat_template via chat_template_kwargs.
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        # vLLM returns RequestOutput objects; extract the first (and only)
        # completion text from each.
        return [output.outputs[0].text for output in outputs]


    def score_responses_batch(
            self,
            prompts: list[str],
            responses: list[str],
        ) -> list[float]:
        """
        Score a batch of (prompt, response) pairs using the two-turn CE-RM pipeline.

        The two turns are dispatched as two separate batched vLLM calls:

          Batch call 1 — criteria generation (all prompts in parallel):
            user: CRITERIA_PROMPT(query=prompt)
            → criteria_text  (one per example)

          Batch call 2 — evaluation (all prompt+response pairs in parallel):
            user:      CRITERIA_PROMPT(query=prompt)
            assistant: criteria_text
            user:      EVALUATION_PROMPT(response=response)
            → evaluation_text  (one per example)

        Args:
            prompts: List of user queries.
            responses: List of assistant responses to evaluate.

        Returns:
            List of overall scores (0–10). Failed parses yield 0.0.
        """
        assert len(prompts) == len(responses)

        # ── Turn 1: generate criteria for every prompt in the batch ──────────
        criteria_conversations = [
            [{"role": "user", "content": CRITERIA_PROMPT.format(query=p)}]
            for p in prompts
        ]
        criteria_texts = self._generate_batch(
            criteria_conversations, self.sampling_criteria
        )

        # ── Turn 2: evaluate every response using its generated criteria ─────
        evaluation_conversations = [
            [
                {"role": "user",      "content": CRITERIA_PROMPT.format(query=p)},
                {"role": "assistant", "content": c},
                {"role": "user",      "content": EVALUATION_PROMPT.format(response=r)},
            ]
            for p, c, r in zip(prompts, criteria_texts, responses)
        ]
        evaluation_texts = self._generate_batch(
            evaluation_conversations, self.sampling_evaluation
        )

        # ── Parse scores ──────────────────────────────────────────────────────
        scores = []
        for eval_text in evaluation_texts:
            score = parse_overall_score(eval_text)
            # Conservative fallback: treat parse failures as score=0
            # (marks the example as incorrect)
            scores.append(score if score is not None else 0.0)

        return scores


    def evaluate_dataset(
            self,
            dataset_name: str = "project-themis/Themis-CodeRewardBench",
            config: str | None = None,
            split: str | None = None,
            batch_size: int = 16,
        ) -> dict:
        """
        Evaluate CE-RM on the Code RewardBench dataset.

        Each preference pair (prompt, chosen, rejected) requires two batched
        vLLM calls per response (criteria + evaluation), so four batched calls
        per batch of pairs in total:

          1. criteria  for all chosen responses
          2. evaluation for all chosen responses
          3. criteria  for all rejected responses
          4. evaluation for all rejected responses

        Args:
            dataset_name: HuggingFace dataset identifier.
            config: Optional dataset configuration name.
            split: Optional dataset split.
            batch_size: Number of preference pairs per vLLM batch.

        Returns:
            Compiled results dict.
        """
        print(f"Loading dataset: {dataset_name}")
        if config:
            print(f"Config: {config}  |  Split: {split}")
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, "Full", split="Full")

        print(f"Evaluating {len(dataset)} examples with batch size {batch_size} …")

        results       = []
        detailed_scores = []

        for batch_start in tqdm(
                range(0, len(dataset), batch_size),
                desc="Processing batches",
            ):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch     = dataset[batch_start:batch_end]

            try:
                prompts            = batch['prompt']   if isinstance(batch['prompt'],   list) else [batch['prompt']]
                chosen_responses   = batch['chosen']   if isinstance(batch['chosen'],   list) else [batch['chosen']]
                rejected_responses = batch['rejected'] if isinstance(batch['rejected'], list) else [batch['rejected']]

                if isinstance(batch['id'], list):
                    ids       = batch['id']
                    languages = batch['language']
                    aspects   = batch['aspect']
                    subsets   = batch['subset']
                else:
                    ids       = [batch['id']]
                    languages = [batch['language']]
                    aspects   = [batch['aspect']]
                    subsets   = [batch['subset']]

                # Score chosen and rejected responses in two batched passes each
                chosen_scores   = self.score_responses_batch(prompts, chosen_responses)
                rejected_scores = self.score_responses_batch(prompts, rejected_responses)

            except Exception as e:
                print(f"\nError processing batch starting at {batch_start}: {e}")
                continue

            for j, (chosen_score, rejected_score) in enumerate(
                    zip(chosen_scores, rejected_scores)
                ):
                correct    = chosen_score > rejected_score
                score_diff = chosen_score - rejected_score

                record = {
                    'id':             ids[j],
                    'chosen_score':   chosen_score,
                    'rejected_score': rejected_score,
                    'correct':        correct,
                    'score_diff':     score_diff,
                    'language':       languages[j],
                    'aspect':         aspects[j],
                    'subset':         subsets[j],
                }
                results.append(record)
                detailed_scores.append(record.copy())

        compiled = self.compile_results(results)
        compiled['detailed_scores'] = detailed_scores
        return compiled


    def compile_results(self, results: list[dict]) -> dict:
        """
        Compile per-example results into aggregate statistics.
        Schema is identical to the scalar evaluator for drop-in compatibility.
        """
        df = pd.DataFrame(results)

        overall_accuracy  = df['correct'].mean()
        aspect_accuracy   = df.groupby('aspect')['correct'].agg(['mean','count']).round(4)
        language_accuracy = df.groupby('language')['correct'].agg(['mean','count']).round(4)
        subset_accuracy   = df.groupby('subset')['correct'].agg(['mean','count']).round(4)

        for acc_df in (aspect_accuracy, language_accuracy, subset_accuracy):
            acc_df.columns = ['accuracy', 'count']

        return {
            'overall': {
                'accuracy':                round(float(overall_accuracy), 4),
                'total_examples':          len(df),
                'correct_predictions':     int(df['correct'].sum()),
                'mean_score_difference':   round(float(df['score_diff'].mean()), 4),
                'median_score_difference': round(float(df['score_diff'].median()), 4),
                'std_score_difference':    round(float(df['score_diff'].std()), 4),
            },
            'by_aspect':   aspect_accuracy.to_dict('index'),
            'by_language': language_accuracy.to_dict('index'),
            'by_subset':   subset_accuracy.to_dict('index'),
            'raw_results': results,
        }


    def save_results(self, results: dict, output_dir: str) -> None:
        """Save scores as Parquet and summary statistics as JSON."""
        os.makedirs(output_dir, exist_ok=True)

        scores_df   = pd.DataFrame(results['detailed_scores']).set_index('id')
        scores_path = os.path.join(output_dir, 'scores.parquet')
        scores_df.to_parquet(scores_path, index=True)
        print(f"Detailed scores saved to: {scores_path}")

        to_save      = {k: v for k, v in results.items()
                        if k not in ('raw_results', 'detailed_scores')}
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(to_save, f, indent=2)
        print(f"Results saved to: {results_path}")


    def print_results(self, results: dict) -> None:
        """Print a human-readable summary of evaluation results."""
        print("\n" + "=" * 60)
        print("CE-RM EVALUATION RESULTS (vLLM)")
        print("=" * 60)

        o = results['overall']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Examples:        {o['total_examples']}")
        print(f"  Correct Predictions:   {o['correct_predictions']}")
        print(f"  Accuracy:              {o['accuracy']:.2%}")
        print(f"  Mean Score Difference: {o['mean_score_difference']:.4f}")
        print(f"  Median Score Diff:     {o['median_score_difference']:.4f}")
        print(f"  Score Diff Std:        {o['std_score_difference']:.4f}")

        print(f"\nACCURACY BY ASPECT:")
        for aspect, stats in results['by_aspect'].items():
            print(f"  {aspect:<25}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print(f"\nACCURACY BY LANGUAGE:")
        for lang, stats in results['by_language'].items():
            print(f"  {lang:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print(f"\nACCURACY BY SUBSET:")
        for subset, stats in results['by_subset'].items():
            print(f"  {subset:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")

        print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CE-RM on Code RewardBench using vLLM"
    )
    parser.add_argument("model_path",  help="HuggingFace model ID or local path")
    parser.add_argument("--dataset",   default="project-themis/Themis-CodeRewardBench", help="Dataset name")
    parser.add_argument("--config",    default=None, help="Dataset configuration name")
    parser.add_argument("--split",     default=None, help="Dataset split")
    parser.add_argument("--output",    required=True, help="Output directory for results")
    parser.add_argument(
        "--max-new-tokens-criteria",
        type=int, default=4096,
        help="Max new tokens for criteria generation (default: 4096)"
    )
    parser.add_argument(
        "--max-new-tokens-evaluation",
        type=int, default=8192,
        help="Max new tokens for evaluation generation (default: 8192)"
    )
    parser.add_argument(
        "--batch-size",
        type=int, default=16,
        help="Preference pairs per vLLM batch (default: 16)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int, default=None,
        help="Cap on vLLM KV-cache context length. Reduce to save GPU memory."
    )

    args = parser.parse_args()

    evaluator = CERMEvaluator(
        model_path=args.model_path,
        max_new_tokens_criteria=args.max_new_tokens_criteria,
        max_new_tokens_evaluation=args.max_new_tokens_evaluation,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    results = evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        batch_size=args.batch_size,
    )

    evaluator.print_results(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()