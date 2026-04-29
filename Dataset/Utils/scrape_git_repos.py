"""Clone GitHub repositories and serialise their contents using yek.

Loads a HuggingFace dataset of repository metadata, shallow-clones each repo
(optionally at a specific commit), runs ``yek --json`` to produce a single-file
JSON serialisation, and saves the enriched dataset to disk.  Used to build and
extend the curated repository allowlists in ``Repos/``.
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import datasets
import logging
import os
import subprocess
import torch
import json
import glob
from collections import Counter

logger = logging.getLogger(__name__)

NUM_THREADS = 6
torch.set_num_threads(8)


def run_in_shell(cmd: str, cwd=None, timeout=150):
    """Execute *cmd* in a shell subprocess, raising on timeout."""
    return subprocess.run([cmd], capture_output=False, shell=True, cwd=cwd, timeout=timeout)


def clone_repo_to_commit(ex):
    """Clone a single repo, optionally reset to a commit, and serialise with yek."""
    commit_id = ex.get("revision_id")
    repo = f"git@github.com:{ex['repo_name']}.git"
    ex["serialised_repo"] = None

    try:
        cwd = os.path.join(os.getcwd(), OUTPUT_PREFIX, ex["repo_name"])
        completed = run_in_shell(f"git clone {repo} {cwd} --depth 1", cwd=os.getcwd())
        logger.error(f"git clone for repo: {repo} finished with status: {completed}")
        if completed.returncode != 0:
            logger.error(f"Failed to clone repo {repo}. Skipping...")
            ex["status"] = "error"
            logger.error(f"Error message: {completed.stderr.decode('utf-8')}")
            return ex

        if RESET_TO_COMMIT and commit_id:
            completed = run_in_shell(f"git reset --hard {commit_id}", cwd=cwd)
            logger.error(f"git checkout for repo: {repo} at commit: {commit_id} finished with status: {completed}")
            if completed.returncode != 0:
                logger.error(f"Failed to checkout commit {commit_id} for repo {repo}. Skipping...")
                ex["status"] = "error"
                logger.error(f"Error message: {completed.stderr.decode('utf-8')}")
                return ex

        ex["status"] = "success"
        logger.info(f"Successfully cloned {repo}")
        completed = run_in_shell("yek --json --output-dir .", cwd=cwd)
        if completed.returncode != 0:
            logger.error(f"Failed to run yek on repo {repo}. Skipping...")
            ex["status"] = "error"
            logger.error(f"Error message: {completed.stderr.decode('utf-8')}")
            return ex

        yek_files = glob.glob(os.path.join(cwd, "yek-output-*.json"))
        if yek_files:
            with open(yek_files[0], "r") as f:
                ex["serialised_repo"] = json.load(f)
            logger.info(f"Serialized repo {repo} to {ex['serialised_repo']}")

    except Exception as e:
        logger.error(f"Exception processing repo {repo} at commit {commit_id}: {e}")
        ex["status"] = "exception"

    return ex


def get_repo_multi_threaded_processed(batch):
    """HuggingFace batched map function: process a batch of repos in parallel threads."""
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(clone_repo_to_commit, [dict(zip(batch,t)) for t in zip(*batch.values())]))
        return {k: [dic[k] for dic in results] for k in results[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name or path (e.g. iNeil77/filtered-repos)"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Directory prefix for cloned repos and output dataset (e.g. Repos)"
    )
    parser.add_argument(
        "--reset_to_commit",
        action="store_true",
        default=False,
        help="After cloning, git reset --hard to the revision_id column value"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=103000
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20
    )
    args = parser.parse_args()

    global OUTPUT_PREFIX, RESET_TO_COMMIT
    OUTPUT_PREFIX = args.output_prefix
    RESET_TO_COMMIT = args.reset_to_commit

    ds = datasets.load_dataset(
        args.dataset,
        split="train",
        num_proc=args.workers
    )
    if args.start_index < 0 or args.end_index > len(ds):
        raise ValueError("Please enter valid split boundaries!")

    logging.basicConfig(
        filename=f"scrape_{args.output_prefix}_{args.start_index}_to_{args.end_index}.log",
        encoding="utf-8",
        level=logging.INFO
    )

    ds = ds.select(range(args.start_index, args.end_index))
    print(f"Cloning repos for span {args.start_index} to {args.end_index}")
    ds = ds.map(
        get_repo_multi_threaded_processed,
        num_proc=args.workers,
        batch_size=NUM_THREADS,
        batched=True,
    )
    print(f"Status of repos after cloning: {Counter(ds['status'])}")
    ds = ds.filter(
        lambda ex: ex["status"] == "success" and ex["serialised_repo"] is not None,
        num_proc=args.workers
    ).remove_columns("status")
    print(f"Cloned {len(ds)} repos successfully.")
    ds.save_to_disk(
        f"{args.output_prefix}_{args.start_index}_to_{args.end_index}",
        num_proc=args.workers
    )
