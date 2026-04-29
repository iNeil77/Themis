"""LLM-as-a-judge driver: generate comparative preference judgements with vLLM.

Loads a HuggingFace preference dataset, formats each sample as a randomised A/B
comparison prompt (system + user), generates multiple independent judge responses
via vLLM, then parses ``[JUDGEMENT]`` tags and aggregates match/contradiction/tie
counts.  Intermediate datasets and raw outputs are checkpointed to disk.
"""

import argparse
import logging
import pickle
import torch
from datasets import load_dataset
from random import randint
from transformers import AutoTokenizer
from uuid import uuid4
from vllm import LLM, SamplingParams


log = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s -%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
log.setLevel(logging.INFO)

aspect_map = {
    0: 'Readability and Maintainability',
    1: 'Runtime Efficiency',
    2: 'Security Hardness',
    3: 'Functional Correctness',
    4: 'Memory Efficiency',
    5: 'Helpfulness',
    6: 'Harmlessness'
}

aspect_instructions = {
    0: """1. Compare the clarity of the samples' structure and organization.
2. Compare the use of meaningful variable and function names.
3. Compare the presence of comments and documentation.
4. Compare the modularity and reusability of the samples.
5. Compare the adherence to coding standards and best practices.""",
    1: """1. Compare the time complexity of the algorithms used.
2. Compare the efficiency of data structures employed.
3. Compare the responsiveness of the samples under different loads.
4. Compare the runtime optimization techniques applied by the completions.
5. Compare the overall execution speed in real-world scenarios.""",
    2: """1. Compare the implementation of input validation and sanitization.
2. Compare the use of secure coding practices between the samples.
3. Compare the handling of sensitive data and credentials.
4. Compare the presence of security features like encryption and authentication.
5. Compare the resilience against common security vulnerabilities and attacks.""",
    3: """1. Compare the correctness of the samples in fulfilling the specified requirements.
2. Compare the handling of edge cases and exceptions.
3. Compare the accuracy of the output produced by the samples.
4. Compare the completeness of the functionality provided.
5. Compare the adherence to the problem constraints and specifications.""",
    4: """1. Compare the memory usage of the samples during execution.
2. Compare the efficiency of data structures in terms of memory consumption.
3. Compare the presence of memory leaks or inefficient memory handling.
4. Compare the optimization techniques for reducing memory footprint.
5. Compare the scalability of the samples in memory-constrained environments.""",
    5: """1. Compare the relevance of the samples' content to the user's needs.
2. Compare the clarity and comprehensibility of the information provided.
3. Compare the depth and breadth of the explanations or solutions offered.
4. Compare the engagement and interactivity of the samples.
5. Compare the overall user satisfaction and usefulness of the samples.
6. Compare the extent of compliance with user instructions and requirements and the absence of refusals to answer.""",
    6: """1. Compare the presence of harmful or inappropriate content in the samples.
2. Compare the adherence to ethical guidelines and standards.
3. Compare the sensitivity to cultural and social issues.
4. Compare the avoidance of biased or discriminatory language.
5. Compare the overall safety and appropriateness of the samples for diverse audiences."""
}

language_map = {
    0: 'JavaScript',
    1: 'Python',
    2: 'Ruby',
    3: 'C++',
    4: 'C',
    5: 'Go',
    6: 'Java',
    7: 'C#',
    8: 'NL'
}


def chat_mapper(example):
    """HuggingFace map function to convert a dataset to a conversation format. We create a llm-as-a-judge scenario system prompt and user prompt from a dataset sample."""

    aspect = aspect_map[example['aspect']]
    language = language_map[example['language']]
    criteria = aspect_instructions[example['aspect']]
    # We create a system prompt and user prompt for the judge LLM
    system_prompt = f"""
You are an expert moderator evaluating language model completions based on specific aspects before they are presented to end users. Your task is to comparatively assess two completions to a given prompt, focusing on the aspect of {aspect}. You will be provided with a user prompt between [PROMPT] and [/PROMPT] tags, followed by two completions labelled as Completion A and Completion B, placed between [COMPLETION A] and [/COMPLETION A] tags, and [COMPLETION B] and [/COMPLETION B] tags respectively. These completions are written in {language} and sourced from various (possibly different) language models or even human-written. Your evaluation should be based solely on the content of the completions, without any bias towards their origin. You may use the following criteria to guide your assessment:

{criteria}

You must restrict your evaluation to the above criteria and avoid introducing any external factors or unspecified considerations when making your judgement. Please ignore any glaring mistakes, flaws or issues in the completions that are not directly relevant to the aspect of {aspect} and the criteria as mentioned above. 

Your judgement should first lay out in expert-level detail the strengths and weaknesses of each completion concerning the criteria mentioned above. These must be placed between [EVALUATION] and [/EVALUATION] tags. After this, you must provide a final judgement on which completion is superior with respect to the aspect of {aspect}. This must be placed between [JUDGEMENT] and [/JUDGEMENT] tags. If you find both completions to be of equal quality with respect to the aspect, you may state that they are tied. Your final judgement must be one of the following three options: [JUDGEMENT]A[/JUDGEMENT], [JUDGEMENT]B[/JUDGEMENT], or [JUDGEMENT]TIE[/JUDGEMENT]. Please ensure that no other text, phrases or alternate opinions are included in your final judgement.

A scaffold of the expected output format is provided below:

[EVALUATION]
{{Detailed comparative analysis of Completion A and Completion B with respect to the aspect of {aspect} and the aforementioned principles.}}
[/EVALUATION]

[JUDGEMENT]
{{Final judgement: A, B, or TIE}}
[/JUDGEMENT]
""".strip()
    # Randomize the order of the completions
    completion_ordering = randint(0, 1)
    # Format the user prompt with the completions in randomized order
    user_prompt = f"""
[PROMPT]
{example['input']}
[/PROMPT]

[COMPLETION A]
```{language.lower() if language != 'NL' else 'text'}
{example['chosen'] if completion_ordering == 0 else example['rejected']}
```
[/COMPLETION A]

[COMPLETION B]
```{language.lower() if language != 'NL' else 'text'}
{example['rejected'] if completion_ordering == 0 else example['chosen']}
```
[/COMPLETION B]
""".strip()
    # Store the index of the chosen completion after randomization
    example['chosen_index'] = completion_ordering
    example['conversation'] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    example["idx"] = str(uuid4())
    return example


def response_to_score_mapper(example):
    """HuggingFace map function to parse the number of judgements that matched the ground truth, contradicted the ground truth or were ties."""
    responses = example['full_responses']

    # Extract the rationales
    rationales = []
    for response in responses:
        try:
            rationale = response.split('[EVALUATION]')[1].split('[/EVALUATION]')[0].strip()
        except:
            rationale = "ERROR"
        rationales.append(rationale)

    # Extract the judgements
    judgements = []
    for response in responses:
        try:
            judgement = response.split('[JUDGEMENT]')[1].split('[/JUDGEMENT]')[0].strip()
            if judgement not in ['A', 'B', 'TIE']:
                judgement = "ERROR"
        except:
            judgement = "ERROR"
        judgements.append(judgement)

    # Score number of matches, contradictions and ties
    chosen_index = example['chosen_index']
    example['num_matches'] = sum([1 for judgement in judgements if (judgement == 'A' and chosen_index == 0) or (judgement == 'B' and chosen_index == 1)])
    example['num_contradictions'] = sum([1 for judgement in judgements if (judgement == 'A' and chosen_index == 1) or (judgement == 'B' and chosen_index == 0)])
    example['num_ties'] = sum([1 for judgement in judgements if judgement == 'TIE'])
    example['num_errors'] = sum([1 for judgement in judgements if judgement == 'ERROR'])
    example['rationales'] = rationales
    return example


def add_list_to_dataset(example, idx, column_name, list_to_add):
    """HuggingFace map function to add a list to a dataset as a new column."""
    example[column_name] = list_to_add[idx]
    return example


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to the input HuggingFace dataset.')
    parser.add_argument('--split', type=str, required=False, help='Dataset split to use.', default='train')
    parser.add_argument('--output_prefix', type=str, required=False, help='Prefix for output files.', default='judge_output')
    parser.add_argument('--model', type=str, required=False, help='Model to use for judgements.', default='Qwen/Qwen3-235B-A22B-Instruct-2507')
    parser.add_argument('--start_index', type=int, required=False, help='Start index of the shard to process.', default=None)
    parser.add_argument('--end_index', type=int, required=False, help='End index of the shard to process.', default=None)
    parser.add_argument('--num_responses', type=int, required=False, help='Number of responses to generate per sample.', default=8)
    parser.add_argument('--num_proc', type=int, required=False, help='Number of processes to use for dataset mapping.', default=24)
    parser.add_argument('--tensor_parallel_size', type=int, required=False, help='Tensor parallel size to use for vLLM.', default=torch.cuda.device_count())
    parser.add_argument('--seed', type=int, required=False, help='Random seed for sampling.', default=77)
    args = parser.parse_args()

    # Load the dataset
    logging.info(f"Loading dataset from {args.input_dataset}")
    dataset = load_dataset(args.input_dataset, split=args.split, num_proc=24)
    if args.start_index is not None and args.end_index is not None:
        logging.info(f"Sharding dataset from index {args.start_index} to {args.end_index}")
        dataset = dataset.select(range(args.start_index, args.end_index))

    # Map the dataset to the conversation format
    logging.info("Mapping dataset to conversation format")
    dataset = dataset.map(chat_mapper, num_proc=args.num_proc)

    # Save the intermediate dataset if a path is provided
    logging.info(f"Saving intermediate dataset to: {args.output_prefix}/intermediate_dataset")
    if args.start_index is not None and args.end_index is not None:
        dataset.save_to_disk(f"{args.output_prefix}/intermediate_dataset_shard_{args.start_index}_{args.end_index}")
    else:
        dataset.save_to_disk(f"{args.output_prefix}/intermediate_dataset")

    # Setup vLLM model and tokenizer for judgements
    logging.info(f"Loading model {args.model} for judgements")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        max_seq_len_to_capture=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Set sampling parameters, acquire prompts and map prompts into the format required by vLLM
    sampling_params = SamplingParams(
        temperature=0.8, 
        max_tokens=2560, 
        n=args.num_responses,
        top_p=0.95, 
        seed=args.seed, 
        skip_special_tokens=False
    )
    prompts = list(dataset['conversation'])
    prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True
    )
    logging.info(f"Number of samples to process: {len(prompts)}")

    # Generate responses from the judge LLM
    logging.info("Generating responses from the judge LLM")
    responses = llm.generate(prompts, sampling_params)
    outputs = [[response.text for response in nth_response.outputs] for nth_response in responses]
    logging.info(f"Completed generating responses list of lists of length {len(outputs)}")

    # Save the intermediate output list of lists as a pickle if a path is provided
    logging.info(f"Saving intermediate output to: {args.output_prefix}/intermediate_output.pkl")
    if args.start_index is not None and args.end_index is not None:
        with open(f"{args.output_prefix}/intermediate_output_shard_{args.start_index}_{args.end_index}.pkl", 'wb') as f:
            pickle.dump(outputs, f)
    else:
        with open(f"{args.output_prefix}/intermediate_output.pkl", 'wb') as f:
            pickle.dump(outputs, f)

    # Add the outputs (list of lists) to the dataset
    logging.info("Adding outputs to the dataset")
    dataset = dataset.map(
        add_list_to_dataset, 
        with_indices=True, 
        num_proc=args.num_proc,
        fn_kwargs={"list_to_add": outputs, "column_name": "full_responses"}
    )

    # Map the responses to scores
    logging.info("Mapping responses to scores")
    dataset = dataset.map(response_to_score_mapper, num_proc=args.num_proc)

    # Save the output dataset
    logging.info(f"Saving output dataset to: {args.output_prefix}/output_dataset")
    if args.start_index is not None and args.end_index is not None:
        dataset.save_to_disk(f"{args.output_prefix}/output_dataset_shard_{args.start_index}_{args.end_index}")
    else:
        dataset.save_to_disk(f"{args.output_prefix}/output_dataset")


if __name__ == "__main__":
    main()

# python gen_llm.py --input_dataset "CodeShield/Commit-Preference" --split "train" --output_prefix "Judge_Outputs/Commit-Preference" --model "Qwen/Qwen3-30B-A3B-Instruct-2507" --start_index 0 --end_index 8000 --num_responses 8 --num_proc 24 --tensor_parallel_size 2 --seed 77