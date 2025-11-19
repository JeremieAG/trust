from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Dict, List, Optional


def load_sections(path: str) -> Dict[int, str]:
    """
    Load sections JSON and return a mapping page -> section name.
    
    Args:
        path (str): Path to the sections JSON file
    Returns:
        Dict[int, str]: Mapping of page number to section name
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sections = data.get("sections", [])
    page_map: Dict[int, str] = {}
    for sec in sections:
        name = sec.get("name")
        start = int(sec.get("startPage", 0))
        end = int(sec.get("endPage", start))
        for p in range(start, end + 1):
            page_map[p] = name
    return page_map


def compare_maps(gt: Dict[int, str], out: Dict[int, str]) -> int:
    """
    Return number of page errors between ground-truth and output.
    Any page present in gt but missing or different in out counts as an error.
    Pages present in out but not in gt are ignored.
    
    Args:
        gt (Dict[int, str]): Ground-truth mapping of page number to section name
        out (Dict[int, str]): Output mapping of page number to section name
    Returns:
        int: Number of page errors
    """
    errors = 0
    for p, gt_name in gt.items():
        out_name = out.get(p)
        if out_name is None:
            errors += 1
        else:
            if str(gt_name).strip().lower() != str(out_name).strip().lower():
                errors += 1
    return errors


def parse_stdout_for_metrics(stdout: str) -> tuple[Optional[float], Optional[int]]:
    """
    Try to extract time and API calls from solution stdout.

    Expected pattern: "Processing completed in {time:.2f} seconds with {call_count} API calls."
    Returns (time_seconds, call_count) or (None, None) if not found.
    
    Args:
        stdout (str): Standard output from the solution script
    Returns:
        tuple[Optional[float], Optional[int]]: (time_seconds, call_count)
    """
    m = re.search(
        r"Processing completed in ([0-9]+\.?[0-9]*) seconds with ([0-9]+) API calls",
        stdout,
    )
    if m:
        t = float(m.group(1))
        calls = int(m.group(2))
        return t, calls
    return None, None


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run solution.py multiple times and compute accuracy against a ground-truth JSON."
    )
    parser.add_argument(
        "--runs", "-n", type=int, default=50, help="Number of runs to execute"
    )
    parser.add_argument(
        "--pdf",
        default=os.path.join(os.path.dirname(__file__), "..", "DEMO_MinuteBook_FR.pdf"),
        help="PDF file to pass to solution.py",
    )
    parser.add_argument(
        "--expected_results",
        default=os.path.join(os.path.dirname(__file__), "demo-results.json"),
        help="Ground truth JSON file to compare against",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "accuracy.json"),
        help="Output accuracy JSON file",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when running solution.py",
    )
    args = parser.parse_args()

    runs = args.runs
    pdf = os.path.abspath(args.pdf)
    expected_results = os.path.abspath(args.expected_results)
    out_file = os.path.abspath(args.out)
    py = args.python

    if not os.path.exists(pdf):
        print(f"PDF not found: {pdf}")
        return
    if not os.path.exists(expected_results):
        print(f"expected_results results not found: {expected_results}")
        return

    gt_map = load_sections(expected_results)

    workspace_dir = os.path.dirname(__file__)
    tmp_dir = os.path.join(workspace_dir, "_accuracy_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    errors_list: List[int] = []
    requests_list: List[Optional[int]] = []
    times_list: List[float] = []
    perfect_count = 0
    run_details = []

    solution_path = os.path.join(workspace_dir, "solution.py")
    if not os.path.exists(solution_path):
        print(f"solution.py not found at {solution_path}")
        return

    for i in range(1, runs + 1):
        run_out = os.path.join(tmp_dir, f"result_{i}.json")
        cmd = [py, solution_path, pdf, "-o", run_out]
        print(f"Run {i}/{runs}: executing: {' '.join(cmd)}")

        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.perf_counter() - start
            stdout = proc.stdout + "\n" + proc.stderr
        except Exception as e:
            elapsed = time.perf_counter() - start
            stdout = str(e)
            print(f"Run {i} failed: {e}")

        parsed_time, parsed_calls = parse_stdout_for_metrics(stdout)
        run_time = parsed_time if parsed_time is not None else elapsed

        out_map = {}
        if os.path.exists(run_out):
            try:
                out_map = load_sections(run_out)
            except Exception as e:
                print(f"Error reading output JSON for run {i}: {e}")
                out_map = {}
        else:
            print(f"Output file not produced for run {i}: expected {run_out}")

        errors = compare_maps(gt_map, out_map)
        if errors == 0:
            perfect_count += 1

        errors_list.append(errors)
        requests_list.append(parsed_calls)
        times_list.append(run_time)

        run_details.append(
            {
                "run": i,
                "errors": errors,
                "requests": parsed_calls,
                "time": run_time,
            }
        )

    numeric_requests = [r for r in requests_list if r is not None]

    summary = {
        "Average errors": mean(errors_list) if errors_list else 0,
        "Min error": min(errors_list) if errors_list else 0,
        "Max error": max(errors_list) if errors_list else 0,
        "perfect count": perfect_count,
        "Avg requests": mean(numeric_requests) if numeric_requests else None,
        "Min requests": min(numeric_requests) if numeric_requests else None,
        "Max requests": max(numeric_requests) if numeric_requests else None,
        "Avg time": mean(times_list) if times_list else 0,
        "Min time": min(times_list) if times_list else 0,
        "Max time": max(times_list) if times_list else 0,
        "runs": run_details,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"Accuracy summary written to {out_file}")


if __name__ == "__main__":
    main()
