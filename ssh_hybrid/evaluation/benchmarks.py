"""General benchmark evaluation via lm-eval-harness.

Measures benign task performance (HellaSwag, MMLU, ARC-Challenge)
to ensure MBCA monitoring does not degrade model quality.
"""

from __future__ import annotations

import logging
import subprocess
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""

    benchmark: str
    model_name: str
    accuracy: float
    stderr: float
    n_samples: int
    raw_results: dict[str, Any]


def run_lm_eval_benchmark(
    model_name_or_path: str,
    benchmarks: list[str] | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    num_fewshot: int | None = None,
    trust_remote_code: bool = False,
    output_dir: str | None = None,
) -> list[BenchmarkResult]:
    """Run lm-eval-harness benchmarks on a model.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        benchmarks: List of benchmark names. Default: hellaswag, mmlu, arc_challenge.
        batch_size: Evaluation batch size.
        device: Device for evaluation.
        num_fewshot: Number of few-shot examples. None = benchmark default.
        trust_remote_code: Whether to trust remote code.
        output_dir: Directory for result output. None = temp dir.

    Returns:
        List of BenchmarkResult for each benchmark.
    """
    if benchmarks is None:
        benchmarks = ["hellaswag", "mmlu", "arc_challenge"]

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ssh_hybrid_eval_")

    results = []

    for benchmark in benchmarks:
        logger.info("Running %s on %s", benchmark, model_name_or_path)

        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_name_or_path},trust_remote_code={str(trust_remote_code).lower()}",
            "--tasks", benchmark,
            "--batch_size", str(batch_size),
            "--device", device,
            "--output_path", output_dir,
            "--log_samples",
        ]

        if num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(num_fewshot)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per benchmark
            )

            if proc.returncode != 0:
                logger.error(
                    "lm_eval failed for %s: %s", benchmark, proc.stderr
                )
                continue

            # Parse results from output directory
            result = _parse_lm_eval_output(output_dir, benchmark, model_name_or_path)
            if result:
                results.append(result)
                logger.info(
                    "%s on %s: accuracy=%.4f (±%.4f)",
                    benchmark, model_name_or_path, result.accuracy, result.stderr,
                )

        except subprocess.TimeoutExpired:
            logger.error("Timeout running %s on %s", benchmark, model_name_or_path)
        except FileNotFoundError:
            logger.error(
                "lm_eval command not found. Install with: pip install lm-eval"
            )
            break

    return results


def _parse_lm_eval_output(
    output_dir: str,
    benchmark: str,
    model_name: str,
) -> BenchmarkResult | None:
    """Parse lm-eval-harness output files to extract results."""
    output_path = Path(output_dir)

    # lm-eval writes results as JSON
    for json_file in output_path.rglob("results*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            if "results" in data:
                for task_name, task_results in data["results"].items():
                    if benchmark in task_name:
                        acc_key = next(
                            (k for k in task_results if "acc" in k.lower() and "stderr" not in k.lower()),
                            None,
                        )
                        stderr_key = next(
                            (k for k in task_results if "stderr" in k.lower()),
                            None,
                        )

                        if acc_key:
                            return BenchmarkResult(
                                benchmark=benchmark,
                                model_name=model_name,
                                accuracy=task_results[acc_key],
                                stderr=task_results.get(stderr_key, 0.0) if stderr_key else 0.0,
                                n_samples=task_results.get("n_samples", 0),
                                raw_results=task_results,
                            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse %s: %s", json_file, e)

    return None


def compare_benchmark_results(
    baseline_results: list[BenchmarkResult],
    monitored_results: list[BenchmarkResult],
) -> dict[str, float]:
    """Compare benchmark results between baseline and MBCA-monitored model.

    Args:
        baseline_results: Results without MBCA.
        monitored_results: Results with MBCA active.

    Returns:
        Dict of benchmark -> absolute accuracy difference.
    """
    baseline_map = {r.benchmark: r for r in baseline_results}
    monitored_map = {r.benchmark: r for r in monitored_results}

    comparisons = {}
    for benchmark in baseline_map:
        if benchmark in monitored_map:
            diff = monitored_map[benchmark].accuracy - baseline_map[benchmark].accuracy
            comparisons[benchmark] = diff
            logger.info(
                "%s: baseline=%.4f, monitored=%.4f, diff=%.4f",
                benchmark,
                baseline_map[benchmark].accuracy,
                monitored_map[benchmark].accuracy,
                diff,
            )

    return comparisons
