"""General benchmark evaluation via lm-eval-harness.

Measures benign task performance (HellaSwag, MMLU, ARC-Challenge)
to ensure MBCA monitoring does not degrade model quality.

Supports both subprocess-based evaluation (baseline model) and
in-process evaluation with MBCA wrapping (monitored model), so
benign degradation is measured on the actual monitored system.
"""

from __future__ import annotations

import logging
import subprocess
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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
    """Run lm-eval-harness benchmarks on a model via subprocess.

    NOTE: This runs the UNMODIFIED model. To measure MBCA-monitored
    performance, use run_lm_eval_in_process() with a wrapped model.

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


def run_lm_eval_in_process(
    model: torch.nn.Module,
    tokenizer: Any,
    model_name: str,
    benchmarks: list[str] | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    num_fewshot: int | None = None,
    output_dir: str | None = None,
) -> list[BenchmarkResult]:
    """Run lm-eval-harness benchmarks in-process on an already-loaded model.

    This enables evaluating MBCA-wrapped models where the subprocess
    approach cannot inject the MBCA register. The model object is passed
    directly to lm-eval's API.

    Args:
        model: Pre-loaded model (potentially wrapped with MBCA).
        tokenizer: Associated tokenizer.
        model_name: Model identifier for logging.
        benchmarks: List of benchmark names.
        batch_size: Evaluation batch size.
        device: Device for evaluation.
        num_fewshot: Number of few-shot examples.
        output_dir: Output directory.

    Returns:
        List of BenchmarkResult for each benchmark.
    """
    if benchmarks is None:
        benchmarks = ["hellaswag", "mmlu", "arc_challenge"]

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ssh_hybrid_eval_inproc_")

    results = []

    try:
        from lm_eval import evaluator as lm_evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error(
            "lm-eval not installed for in-process evaluation. "
            "Install with: pip install lm-eval. "
            "Falling back to subprocess evaluation (which cannot measure "
            "MBCA-monitored performance)."
        )
        return results

    try:
        # Wrap the pre-loaded model for lm-eval
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
        )

        for benchmark in benchmarks:
            logger.info(
                "Running %s on %s (in-process)", benchmark, model_name
            )

            eval_results = lm_evaluator.simple_evaluate(
                model=lm,
                tasks=[benchmark],
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                log_samples=False,
            )

            if eval_results and "results" in eval_results:
                for task_name, task_results in eval_results["results"].items():
                    if benchmark in task_name:
                        acc_key = next(
                            (k for k in task_results
                             if "acc" in k.lower() and "stderr" not in k.lower()),
                            None,
                        )
                        stderr_key = next(
                            (k for k in task_results if "stderr" in k.lower()),
                            None,
                        )

                        if acc_key:
                            result = BenchmarkResult(
                                benchmark=benchmark,
                                model_name=model_name,
                                accuracy=task_results[acc_key],
                                stderr=task_results.get(stderr_key, 0.0) if stderr_key else 0.0,
                                n_samples=task_results.get("n_samples", 0),
                                raw_results=task_results,
                            )
                            results.append(result)
                            logger.info(
                                "%s on %s (in-process): accuracy=%.4f (±%.4f)",
                                benchmark, model_name,
                                result.accuracy, result.stderr,
                            )

    except Exception as e:
        logger.error(
            "In-process lm-eval failed: %s. The lm-eval API may have changed. "
            "Check lm-eval version compatibility.",
            str(e),
        )

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

    IMPORTANT: For a valid comparison, baseline_results should come from
    run_lm_eval_benchmark (unmodified model) and monitored_results should
    come from run_lm_eval_in_process (MBCA-wrapped model). Using subprocess
    evaluation for both does NOT measure MBCA overhead.

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
