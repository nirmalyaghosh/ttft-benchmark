"""
Analyze TTFT benchmark JSONL logs.

Parses api_request and first_token_received events
(run_id-bearing only), joins by run_id, groups by
model and prompt size bucket, and produces:

1. Summary statistics (mean, median, stddev, CI, P95)
2. Linear regression per model (slope, R-squared,
   p-value) to answer: does prompt size
   significantly affect TTFT?
3. Data sufficiency report (target: N >= 30)

Outputs (to benchmark-results/):
  - analysis_summary.json  (machine-readable)
  - analysis_summary.txt   (human-readable)
  - analysis_data.csv      (for plotting)
"""

import csv
import json
import math
import statistics
import sys

import structlog

from collections import defaultdict
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

LOGS_DIR = Path(__file__).parent / "logs"
RESULTS_DIR = Path(__file__).parent / "benchmark-results"

TARGET_N = 30
CONFIDENCE_LEVEL = 0.95
# t-critical for 95% CI, df=29 (two-tailed)
T_CRITICAL_29 = 2.045

# Approximate t-critical values for small samples
# (two-tailed, 95% CI). Keyed by degrees of freedom.
T_CRITICAL_TABLE: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
    5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
    25: 2.060, 29: 2.045, 30: 2.042, 40: 2.021,
    50: 2.009, 100: 1.984,
}

log = structlog.get_logger()


def _compute_ci(
    data: List[float],
) -> Optional[Tuple[float, float]]:
    """Compute 95% confidence interval for mean.

    Returns (lower, upper) or None if N < 2.
    """
    n = len(data)
    if n < 2:
        return None
    mean = statistics.mean(data)
    stderr = statistics.stdev(data) / math.sqrt(n)
    t_crit = _get_t_critical(n - 1)
    margin = t_crit * stderr
    return (
        round(mean - margin, 4),
        round(mean + margin, 4),
    )


def _compute_linear_regression(
    x: List[float],
    y: List[float],
) -> Dict[str, Optional[float]]:
    """Ordinary least squares linear regression.

    Returns slope (ms/token), intercept (s),
    R-squared, and p-value for slope != 0.

    Uses t-test on slope coefficient:
      t = slope / SE(slope)
      p approximated from t-distribution.
    """
    n = len(x)
    if n < 3:
        return {
            "slope_ms_per_token": None,
            "intercept_s": None,
            "r_squared": None,
            "p_value": None,
            "n": n,
        }

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_yy = sum((yi - y_mean) ** 2 for yi in y)
    ss_xy = sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(x, y)
    )

    if ss_xx == 0:
        return {
            "slope_ms_per_token": None,
            "intercept_s": None,
            "r_squared": None,
            "p_value": None,
            "n": n,
        }

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = (
        (ss_xy ** 2) / (ss_xx * ss_yy)
        if ss_yy != 0
        else 0.0
    )

    # Residual standard error
    residuals = [
        yi - (slope * xi + intercept)
        for xi, yi in zip(x, y)
    ]
    ss_res = sum(r ** 2 for r in residuals)
    se_residual = math.sqrt(ss_res / (n - 2))
    se_slope = se_residual / math.sqrt(ss_xx)

    if se_slope > 0:
        t_stat = abs(slope / se_slope)
        p_value = _approximate_p_value(
            t_stat, n - 2,
        )
    else:
        p_value = None

    return {
        "slope_ms_per_token": round(
            slope * 1000, 4,
        ),
        "intercept_s": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "p_value": (
            round(p_value, 6)
            if p_value is not None
            else None
        ),
        "n": n,
    }


def _compute_stats(
    data: List[float],
) -> Dict[str, Any]:
    """Compute summary statistics for TTFT values.

    Returns dict with n, mean, median, stddev,
    min, max, p95, ci_lower, ci_upper.
    """
    n = len(data)
    mean = statistics.mean(data)
    sorted_d = sorted(data)
    if n % 2:
        median = sorted_d[n // 2]
    else:
        median = (
            sorted_d[n // 2 - 1]
            + sorted_d[n // 2]
        ) / 2
    min_v = min(data)
    max_v = max(data)
    stddev = (
        statistics.stdev(data) if n > 1 else 0.0
    )

    p95 = None
    if n >= 20:
        p95 = sorted_d[int(n * 0.95)]

    ci = _compute_ci(data)

    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "stddev": round(stddev, 4),
        "min": round(min_v, 4),
        "max": round(max_v, 4),
        "p95": (
            round(p95, 4)
            if p95 is not None
            else None
        ),
        "ci_lower": ci[0] if ci else None,
        "ci_upper": ci[1] if ci else None,
        "raw": [round(x, 4) for x in data],
    }


def _approximate_p_value(
    t_stat: float,
    df: int,
) -> float:
    """Approximate two-tailed p-value from t-stat.

    Uses a rough approximation suitable for
    reporting significance. For df >= 30,
    uses normal approximation. For smaller df,
    uses conservative estimates from the
    t-critical table.
    """
    if df >= 30:
        # Normal approximation (conservative)
        if t_stat > 3.5:
            return 0.001
        elif t_stat > 2.576:
            return 0.01
        elif t_stat > 1.96:
            return 0.05
        elif t_stat > 1.645:
            return 0.10
        else:
            return 0.50

    # Use t-critical table for small df
    t_crit = _get_t_critical(df)
    if t_stat > t_crit * 1.8:
        return 0.001
    elif t_stat > t_crit * 1.3:
        return 0.01
    elif t_stat > t_crit:
        return 0.05
    elif t_stat > t_crit * 0.85:
        return 0.10
    else:
        return 0.50


def _get_t_critical(df: int) -> float:
    """Look up t-critical value for given df.

    Returns exact value if in table, otherwise
    interpolates or uses closest available.
    """
    if df in T_CRITICAL_TABLE:
        return T_CRITICAL_TABLE[df]

    keys = sorted(T_CRITICAL_TABLE.keys())
    if df < keys[0]:
        return T_CRITICAL_TABLE[keys[0]]
    if df > keys[-1]:
        return T_CRITICAL_TABLE[keys[-1]]

    # Find surrounding keys and interpolate
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (df - lo) / (hi - lo)
            return (
                T_CRITICAL_TABLE[lo]
                + frac * (
                    T_CRITICAL_TABLE[hi]
                    - T_CRITICAL_TABLE[lo]
                )
            )
    return T_CRITICAL_29


def _group_by_model_and_bucket(
    requests: Dict[str, Dict[str, Any]],
    responses: Dict[str, float],
) -> Dict[Tuple[str, int], List[float]]:
    """Group TTFT values by (model, token_bucket)."""
    groups: Dict[
        Tuple[str, int], List[float]
    ] = defaultdict(list)
    for run_id, ttft in responses.items():
        req = requests.get(run_id)
        if not req or req["tokens"] == 0:
            continue
        bucket = _token_bucket(req["tokens"])
        groups[(req["model"], bucket)].append(ttft)
    return dict(groups)


def _parse_logs(
    log_files: List[Path],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Parse JSONL log files.

    Extracts api_request and first_token_received
    events that carry a run_id. Skips warmup runs.

    Returns:
        requests: run_id -> {model, tokens, timestamp}
        responses: seq_id -> ttft (seconds)

    Note: the benchmark script reuses the same
    run_id for all iterations of a given prompt
    size. This function generates unique seq_ids
    (run_id + counter) to preserve all
    measurements.
    """
    requests: Dict[str, Dict[str, Any]] = {}
    responses: Dict[str, float] = {}
    run_id_counts: Dict[str, int] = {}

    for path in log_files:
        try:
            with open(
                path, "r", encoding="utf-8",
            ) as f:
                for line_num, line in enumerate(
                    f, start=1,
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning(
                            "skipped_malformed_line",
                            file=str(path),
                            line=line_num,
                        )
                        continue

                    run_id = rec.get("run_id")
                    if not run_id:
                        continue
                    if "warmup" in run_id:
                        continue

                    event = rec.get("event")
                    if event == "api_request":
                        model = rec.get("model")
                        if not model:
                            log.warning(
                                "missing_model",
                                run_id=run_id,
                                file=str(path),
                                line=line_num,
                            )
                            continue
                        requests[run_id] = {
                            "model": model,
                            "tokens": rec.get(
                                "approximate_"
                                "num_tokens",
                                0,
                            ),
                            "timestamp": rec.get(
                                "timestamp",
                                "",
                            ),
                        }
                    elif (
                        event
                        == "first_token_received"
                    ):
                        ttft = rec.get("ttft")
                        if ttft is not None:
                            count = (
                                run_id_counts
                                .get(run_id, 0)
                            )
                            seq_id = (
                                f"{run_id}"
                                f"#{count}"
                            )
                            responses[seq_id] = ttft
                            run_id_counts[
                                run_id
                            ] = count + 1
                            # Map seq_id to same
                            # request metadata
                            if (
                                run_id in requests
                                and seq_id
                                not in requests
                            ):
                                requests[
                                    seq_id
                                ] = requests[
                                    run_id
                                ]

        except OSError as e:
            log.error(
                "log_file_read_error",
                file=str(path),
                error=str(e),
            )
            continue

    return requests, responses


def _token_bucket(n: int) -> int:
    """Round to nearest target prompt size.

    Targets: 100, 200, 500, 1000, 2000, 5000.
    """
    targets = [100, 200, 500, 1000, 2000, 5000]
    return min(targets, key=lambda t: abs(t - n))


def main() -> None:
    """Entry point: parse, analyze, write."""
    log_files = sorted(
        LOGS_DIR.glob("ttft_benchmark_data.jsonl*")
    )
    if not log_files:
        log.error(
            "no_log_files",
            directory=str(LOGS_DIR),
        )
        sys.exit(1)

    log.info(
        "parsing_logs",
        file_count=len(log_files),
    )
    requests, responses = _parse_logs(log_files)
    log.info(
        "parsing_complete",
        requests=len(requests),
        responses=len(responses),
    )

    groups = _group_by_model_and_bucket(
        requests, responses,
    )

    json_path = (
        RESULTS_DIR / "analysis_summary.json"
    )
    txt_path = (
        RESULTS_DIR / "analysis_summary.txt"
    )
    csv_path = RESULTS_DIR / "analysis_data.csv"

    write_csv(requests, responses, csv_path)
    write_json(
        groups, requests, responses, json_path,
    )
    write_txt(
        groups,
        txt_path,
        len(requests),
        len(responses),
    )

    log.info(
        "analysis_complete",
        json=str(json_path),
        txt=str(txt_path),
        csv=str(csv_path),
    )


def write_csv(
    requests: Dict[str, Dict[str, Any]],
    responses: Dict[str, float],
    output_path: Path,
) -> None:
    """Write individual TTFT measurements as CSV.

    One row per measurement with timestamp,
    suitable for scatter plots, time-of-day
    analysis, and external statistical tools.
    """
    output_path.parent.mkdir(
        parents=True, exist_ok=True,
    )

    rows: List[Tuple[str, int, float, str]] = []
    for run_id, ttft in responses.items():
        req = requests.get(run_id)
        if not req or req["tokens"] == 0:
            continue
        bucket = _token_bucket(req["tokens"])
        rows.append((
            req["model"],
            bucket,
            round(ttft, 4),
            req.get("timestamp", ""),
        ))
    rows.sort()

    with open(
        output_path, "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "prompt_tokens",
            "ttft_seconds",
            "timestamp_utc",
        ])
        for row in rows:
            writer.writerow(row)


def write_json(
    groups: Dict[Tuple[str, int], List[float]],
    requests: Dict[str, Dict[str, Any]],
    responses: Dict[str, float],
    output_path: Path,
) -> None:
    """Write analysis results as JSON.

    Includes per-group statistics, per-model
    linear regression, and timestamped
    measurements.
    """
    # Build run_id -> timestamp lookup
    ts_by_run: Dict[str, str] = {
        rid: req.get("timestamp", "")
        for rid, req in requests.items()
    }

    # Build (model, bucket) -> [(ttft, ts)] map
    measurements: Dict[
        Tuple[str, int],
        List[Dict[str, Any]],
    ] = defaultdict(list)
    for run_id, ttft in responses.items():
        req = requests.get(run_id)
        if not req or req["tokens"] == 0:
            continue
        bucket = _token_bucket(req["tokens"])
        measurements[
            (req["model"], bucket)
        ].append({
            "ttft_seconds": round(ttft, 4),
            "timestamp_utc": ts_by_run.get(
                run_id, "",
            ),
        })

    results = []
    for (model, bucket), data in sorted(
        groups.items()
    ):
        stats = _compute_stats(data)
        results.append({
            "model": model,
            "prompt_tokens": bucket,
            **{
                k: v
                for k, v in stats.items()
                if k != "raw"
            },
            "measurements": sorted(
                measurements.get(
                    (model, bucket), [],
                ),
                key=lambda m: (
                    m["timestamp_utc"]
                ),
            ),
        })

    # Per-model regression
    models = sorted(
        set(m for m, _ in groups)
    )
    regressions = {}
    for model in models:
        x_vals: List[float] = []
        y_vals: List[float] = []
        for (m, bucket), data in groups.items():
            if m != model:
                continue
            for ttft in data:
                x_vals.append(float(bucket))
                y_vals.append(ttft)
        regressions[model] = (
            _compute_linear_regression(
                x_vals, y_vals,
            )
        )

    output = {
        "generated_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "description": (
            "TTFT benchmark analysis: summary "
            "stats, linear regression, and data "
            "sufficiency"
        ),
        "results": results,
        "regressions": regressions,
    }
    output_path.parent.mkdir(
        parents=True, exist_ok=True,
    )
    with open(
        output_path, "w", encoding="utf-8",
    ) as f:
        json.dump(output, f, indent=2)


def write_txt(
    groups: Dict[Tuple[str, int], List[float]],
    output_path: Path,
    num_requests: int,
    num_responses: int,
) -> None:
    """Write human-readable analysis report."""
    models = sorted(
        set(m for m, _ in groups)
    )
    buckets = sorted(
        set(b for _, b in groups)
    )

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("TTFT Benchmark Analysis")
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"Generated: {timestamp}")
    lines.append(
        f"Requests: {num_requests}, "
        f"TTFT measurements: {num_responses}"
    )
    lines.append("=" * 78)
    lines.append("")

    # Summary table
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 78)
    lines.append(
        f"{'Model':<16} {'Tok':>5} {'N':>3}"
        f" {'Mean':>7} {'Med':>7}"
        f" {'SD':>7} {'P95':>7}"
        f" {'95% CI':>17}"
    )
    lines.append("-" * 78)

    for model in models:
        for bucket in buckets:
            data = groups.get((model, bucket))
            if not data:
                continue
            s = _compute_stats(data)

            if s["p95"] is not None:
                p95_s = f"{s['p95']:.3f}s"
            else:
                p95_s = "   N/A"

            if (
                s["ci_lower"] is not None
                and s["ci_upper"] is not None
            ):
                ci_s = (
                    f"[{s['ci_lower']:.3f},"
                    f" {s['ci_upper']:.3f}]"
                )
            else:
                ci_s = "       N/A"

            lines.append(
                f"{model:<16}"
                f" {bucket:>5}"
                f" {s['n']:>3}"
                f" {s['mean']:>6.3f}s"
                f" {s['median']:>6.3f}s"
                f" {s['stddev']:>6.3f}s"
                f" {p95_s}"
                f" {ci_s}"
            )
        lines.append("")

    # Linear regression per model
    lines.append("=" * 78)
    lines.append("LINEAR REGRESSION (TTFT vs prompt size)")
    lines.append("-" * 78)
    lines.append(
        "Question: does prompt size significantly"
        " affect TTFT on this infrastructure?"
    )
    lines.append("")

    for model in models:
        x_vals: List[float] = []
        y_vals: List[float] = []
        for (m, bucket), data in groups.items():
            if m != model:
                continue
            for ttft in data:
                x_vals.append(float(bucket))
                y_vals.append(ttft)

        reg = _compute_linear_regression(
            x_vals, y_vals,
        )

        lines.append(f"  {model}:")
        if reg["slope_ms_per_token"] is not None:
            sig = (
                "YES"
                if (
                    reg["p_value"] is not None
                    and reg["p_value"] < 0.05
                )
                else "NO"
            )
            p_str = (
                f"{reg['p_value']:.4f}"
                if reg["p_value"] is not None
                else "N/A"
            )
            lines.append(
                f"    Slope: "
                f"{reg['slope_ms_per_token']:+.4f}"
                f" ms/token"
            )
            lines.append(
                f"    R-squared: "
                f"{reg['r_squared']:.4f}"
            )
            lines.append(
                f"    p-value: {p_str}"
            )
            lines.append(
                f"    Significant (p<0.05): {sig}"
            )
            lines.append(
                f"    N: {reg['n']}"
            )
        else:
            lines.append(
                "    Insufficient data for"
                " regression (need >= 3 points)"
            )
        lines.append("")

    # Data sufficiency
    lines.append("=" * 78)
    lines.append(
        f"DATA SUFFICIENCY"
        f" (target: N >= {TARGET_N})"
    )
    lines.append("-" * 78)
    for model in models:
        for bucket in buckets:
            data = groups.get((model, bucket))
            if not data:
                continue
            n = len(data)
            gap = max(0, TARGET_N - n)
            if gap == 0:
                status = "OK"
            else:
                status = f"NEED {gap} MORE"
            lines.append(
                f"  {model:<16}"
                f" {bucket:>5} tokens:"
                f" N={n:<4} {status}"
            )
        lines.append("")

    output_path.parent.mkdir(
        parents=True, exist_ok=True,
    )
    with open(
        output_path, "w", encoding="utf-8",
    ) as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
