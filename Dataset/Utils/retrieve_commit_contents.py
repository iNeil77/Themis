"""Retrieve pre- and post-commit file contents from GitHub via shallow git fetches.

For each commit in a HuggingFace dataset, this script shallow-fetches the commit
(``git fetch --depth 2``) from GitHub, checks out the old and new versions of the
changed file, and writes the results to JSONL.  Uses two-level parallelism:
outer HuggingFace ``map(num_proc=workers)`` and inner ``ThreadPoolExecutor``
with ``NUM_THREADS`` threads per process.
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import datasets
import logging
import os
import random
import subprocess
import torch

logger = logging.getLogger(__name__)

NUM_THREADS = 12

torch.set_num_threads(16)

def run_in_shell(cmd: str, cwd=None, timeout=120):
    """Execute *cmd* in a shell subprocess, capturing stdout/stderr."""
    return subprocess.run([cmd], capture_output=True, shell=True, cwd=cwd, timeout=timeout)

def get_file_contents(commit, old_file, new_file, repo, cwd=None):
    """Shallow-fetch a single commit and return (new_contents, old_contents, returncode, stderr)."""
    completed = run_in_shell("git init", cwd=cwd)
    logger.error(f"git init for repo: {repo} finished with status: {completed}")
    completed = run_in_shell("git remote add origin " + repo, cwd=cwd)
    logger.error(f"git remote add origin for repo: {repo} finished with status: {completed}")
    completed = run_in_shell("git fetch --depth 2 origin " + commit, cwd=cwd)
    logger.error(f"git fetch --depth 2 origin for repo {repo} finished with status: {completed}")
    # If it requires authentication
    if completed.returncode != 0:
        return ("", "", completed.returncode, completed.stderr.decode(errors='ignore'))

    completed = run_in_shell("git checkout FETCH_HEAD -- " + new_file, cwd=cwd)
    logger.error(f"git checkout FETCH_HEAD pre-commit for commit {commit} in repo: {repo} finished with status: {completed}")
    new_contents = run_in_shell("cat " + new_file, cwd=cwd).stdout.decode(errors='ignore')
    completed = run_in_shell("git checkout FETCH_HEAD^ -- " + old_file, cwd=cwd)
    logger.error(f"git checkout FETCH_HEAD post-commit for commit {commit} in repo: {repo} finished with status: {completed}")
    
    # If there's only a new file, but no old file
    if completed.returncode != 0:
        return (new_contents, "", completed.returncode, completed.stderr.decode(errors='ignore'))
    old_contents = run_in_shell("cat " + old_file, cwd=cwd).stdout.decode(errors='ignore')
    return (new_contents, old_contents, completed.returncode, completed.stderr.decode(errors='ignore'))

def get_diff(ex):
    """Fetch old/new file contents for one commit, trying each listed repo until success."""
    commit_id = ex["commit"]
    repos = list(set(ex["repos"].split(",")))
    old_file = ex["old_file"]
    new_file = ex["new_file"]
    # Initialize
    returncode = 0
    stderr = "unknown"

    for i, repo in enumerate(repos):
        repo = f"git@github.com:{repo}.git"
        # Create a random directory to store the repo
        random_dir = CWD + "/" + str(random.randint(0, 100000000))
        # Can take very long when running many processes
        run_in_shell("mkdir " + random_dir, timeout=450)
        try:
            new_contents, old_contents, returncode, stderr = get_file_contents(commit_id, old_file, new_file, repo, cwd=random_dir)
        except Exception as e:
            #print("ERROR", commit_id, old_file, new_file, repo, str(random_dir), e)
            # Break in case of many repos that all lead us nowhere
            if i > 15:
                break
            continue
        finally:
            run_in_shell("rm -rf " + random_dir) # clean up again
        ex["new_contents"] = new_contents
        ex["old_contents"] = old_contents
        ex["returncode"] = returncode
        ex["stderr"] = stderr
        return ex

    # If no repo worked
    ex["new_contents"] = ""
    ex["old_contents"] = ""
    ex["returncode"] = returncode
    ex["stderr"] = stderr
    return ex

def get_diff_multi_threaded_processed(batch):
    """HuggingFace batched map function: fetch diffs for a batch in parallel threads."""
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Convert dict of lists to list of dicts then map to threads
        results = list(executor.map(get_diff, [dict(zip(batch,t)) for t in zip(*batch.values())]))
        # Convert list of dicts to dict of lists
        return {k: [dic[k] for dic in results] for k in results[0]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_index",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1000000
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/single-file-commits"
    )
    args = parser.parse_args()
    ds = datasets.load_dataset(
        "commitpack-meta-base-filtered",
        use_auth_token=True,
        split="train"
    )
    if args.start_index<0 or args.end_index>len(ds):
        raise ValueError("Please enter valid split boundaries!")

    global CWD
    CWD = args.base_dir
    if not os.path.exists(CWD):
        os.makedirs(
            CWD,
            exist_ok=True
        )
    logging.basicConfig(
        filename=os.path.join(CWD, f"scrape_{args.start_index}_to_{args.end_index}.log"),
        encoding="utf-8",
        level=logging.INFO
    )

    ds = ds.select(range(args.start_index, args.end_index))
    print(f"Scraping diffs for span {args.start_index} to {args.end_index}")
    ds.map(
        get_diff_multi_threaded_processed,
        num_proc=args.workers,
        batch_size=NUM_THREADS,
        batched=True
    ).filter(
        lambda x: (x["old_contents"] is not None) and (x["old_contents"]!="") and (x["new_contents"] is not None) and (x["new_contents"]!=""),
        num_proc=args.workers
    ).to_json(
        os.path.join(CWD, f"diffs_{args.start_index}_to_{args.end_index}.jsonl"),
        num_proc=args.workers,
        batch_size=20000
    )
