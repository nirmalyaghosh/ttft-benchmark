"""
TTFT Micro-Benchmark Script

Measures Time to First Token (TTFT) across different prompt sizes using various
OpenAI-compatible APIs.

Supports multiple providers:
- OpenRouter: Use OPENAI_BASE_URL="https://openrouter.ai/api/v1",
  OPENAI_API_KEY=<router-key>
- Azure OpenAI: Use
  OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/v1/",
  OPENAI_API_KEY=<azure-key>, MODEL=<deployment-name>
- Other OpenAI-compatible endpoints: Configure BASE_URL and API_KEY accordingly

To switch providers,
- update the environment variables in `ttft_benchmark_.env`
- and restart the script.

Configuration - customize these environment variables to switch providers
For OpenRouter (default):
  OPENAI_BASE_URL="https://openrouter.ai/api/v1"
  OPENAI_API_KEY="sk-or-v1-..."  (your OpenRouter API key)
  MODEL="anthropic/claude-sonnet-4.5"  (or other OpenRouter model)
For Azure OpenAI:
  OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/v1/"
  OPENAI_API_KEY="your-azure-api-key"
  MODEL="your-deployment-name"
  (e.g., "gpt-4", make sure to use the deployment name, not model name)
  AZURE_DEPLOYMENT="deployment-name-if-needed"
  (not currently used, kept for future expansion)

"""

import json
import logging
import time
import statistics
import os
import uuid

import structlog

from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------------------------------------------------------
# Environment & configuration
# -----------------------------------------------------------------------------

# First, load environment variables from the .env file
load_dotenv("ttft_benchmark.env")

# Configurations read from environment variables
ITERATIONS_PER_SIZE = int(os.getenv("ITERATIONS_PER_SIZE", "1"))
# Number of runs per prompt size

MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_SIZES_STR = os.getenv("PROMPT_SIZES", "100")
PROMPT_SIZES = [int(x.strip()) for x in PROMPT_SIZES_STR.split(",")]

SLEEP_SECONDS_BETWEEN_REQUESTS =\
    float(os.getenv("SLEEP_SECONDS_BETWEEN_REQUESTS", "15"))

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

log_file = Path(os.getenv("LOG_FILE_PATH", "ttft_benchmark_data.jsonl"))
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8")]
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Next, configure structlog to use stdlib logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Next, create a structlog logger for this module
log = structlog.get_logger()

# -----------------------------------------------------------------------------
# Initialize the client
# (works with OpenRouter, Azure OpenAI, and other compatible endpoints)
# -----------------------------------------------------------------------------

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _count_words(text: str) -> int:
    """
    Count the number of words in a given text.
    """
    words = text.replace("\n", " ").split()
    return len(words)


def _generate_prompt(target_tokens: int) -> str:
    """
    Generate a prompt of approximately target_tokens length.
    Uses repetitive text for consistency.
    Average English word ≈ 1.3 tokens, so we use ~0.75 words per token.
    """

    prompt_prefix = [
        "This prompt is used to measure Time to First Token (TTFT).\n",
        "Respond only with 'OK'.\n\n",
        "LONG TEXT STARTS BELOW:\n\n"]
    prompt_prefix_text = " ".join(prompt_prefix)
    num_words = _count_words(text=prompt_prefix_text)

    # Next, calculate how many more words we need to reach target_tokens
    words_needed = int(target_tokens * 0.75) - num_words
    # Use simple repetitive text for consistency
    base_text = "The quick brown fox jumps over the lazy dog. "
    repetitions = max(words_needed // 9, 0)  # 9 words in base_text

    # Construct final prompt
    constructed_prompt = prompt_prefix_text + (base_text * repetitions)
    num_words = _count_words(text=constructed_prompt)
    if num_words < int(target_tokens * 0.75):
        # Add extra words if needed
        constructed_prompt += "The quick brown fox. " * (
            int(target_tokens * 0.75) - num_words)

    return constructed_prompt


def _generate_run_id() -> str:
    """
    Generate a unique run identifier.
    """
    return str(uuid.uuid4())


def _get_model_deployment_region() -> str:
    """
    Get information about the region where model endpoint is, if available
    """
    model_region = os.getenv("MODEL_ENDPOINT_DEPLOYMENT_REGION", "")
    if model_region and len(model_region) > 0:
        model_region = f" (deployed at {model_region})"
    else:
        model_region = ""
    return model_region


def _identify_provider() -> str:
    """
    Identify the API provider based on the base URL.
    """
    if OPENAI_BASE_URL and OPENAI_BASE_URL.startswith("https://openrouter.ai"):
        return "OpenRouter"
    elif OPENAI_BASE_URL and "azure" in OPENAI_BASE_URL:
        return "Azure OpenAI"
    else:
        return "OpenAI-compatible endpoint"


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def _log_request(prompt: str, run_id: str) -> None:
    """
    Log request metadata using structlog.
    """
    num_words = _count_words(text=prompt)
    log.info(
        "api_request",
        model=MODEL,
        prompt=prompt if num_words <= 50 else prompt[:197] + "...",
        num_words=num_words,
        approximate_num_tokens=int(_count_words(text=prompt)*(4/3)),
        run_id=run_id,
        timestamp=datetime.now().isoformat()
    )


def _log_response(
    *,
    is_chunk: bool,
    chunk: Optional[str],
    response: Optional[Any],
    run_id: str,
) -> None:
    key = "stream_chunk" if is_chunk else "api_response"

    usage = None
    if response is not None and hasattr(response, "usage"):
        usage = response.usage.model_dump()

    log.info(
        key,
        model=MODEL,
        chunk=chunk,
        usage=usage,
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
    )


# -----------------------------------------------------------------------------
# TTFT measurement, benchmark runner and associated functions
# -----------------------------------------------------------------------------


def _calculate_and_display_cost(run_id_base: str) -> str:
    """
    Helper function used to calculate total tokens used for this benchmark
    run by parsing the JSONL log file.
    """
    log_file_path = os.getenv("LOG_FILE_PATH", "ttft_benchmark_data.jsonl")
    if not os.path.exists(log_file_path):
        print(f"\nLog file {log_file_path} not found for cost calculation.")
        return ""

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    model_used = ""  # Must be read from logs

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if (entry.get("event") == "api_response" and
                        entry.get("run_id", "").startswith(run_id_base + "-")):
                    usage = entry.get("usage", {})
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)
                    model_used = entry.get("model", MODEL)
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    cost_summary_str = "\n" + "="*80
    cost_summary_str += "\nCOST SUMMARY"
    cost_summary_str += "\n" + "="*80
    cost_summary_str += f"\nRun ID Base: {run_id_base}"
    cost_summary_str += f"\nModel Used: {model_used}"
    cost_summary_str += f"\nTotal Input Tokens: {total_input_tokens:,}"
    cost_summary_str += f"\nTotal Output Tokens: {total_output_tokens:,}"
    cost_summary_str += f"\nTotal Tokens: {total_tokens:,}"
    cost_summary_str += "\n" + "="*80
    print(cost_summary_str)

    # Note: Pricing calculation depends on provider and model
    # e.g., for GPT-4, input cost ~$0.03/1K tokens, output ~$0.06/1K tokens

    return cost_summary_str


def _display_results(
        results: Dict[int, Dict[str, Any]],
        cost_summary: str,
        results_file: Optional[Any] = None) -> None:
    """
    Display benchmark results in a formatted table.
    """
    lines = []
    lines.append("="*80)
    lines.append("RESULTS")
    lines.append("="*80)
    lines.append("")

    lines.append(
        f"{'Prompt Size':<15} {'Mean TTFT':<15} {'Median TTFT':<15} "
        f"{'Min':<10} {'Max':<10} {'StdDev':<10}")
    lines.append("-"*80)

    for size in sorted(results.keys()):
        r = results[size]
        lines.append(
            f"{size:<15} {r['mean']:.3f}s{'':<8} {r['median']:.3f}s{'':<8} "
            f"{r['min']:.3f}s{'':<3} {r['max']:.3f}s{'':<3} {r['stdev']:.3f}s")

    lines.append("")
    lines.append("="*80)
    lines.append("PER-TOKEN LATENCY ANALYSIS")
    lines.append("="*80)
    lines.append("")

    # Calculate per-token latency between prompt sizes
    sorted_sizes = sorted(results.keys())

    if len(sorted_sizes) >= 2:
        lines.append("Calculating per-token latency between "
                     "consecutive prompt sizes:")
        lines.append("")

        per_token_rates = []

        for i in range(len(sorted_sizes) - 1):
            size1, size2 = sorted_sizes[i], sorted_sizes[i+1]
            ttft1 = results[size1]["mean"]
            ttft2 = results[size2]["mean"]

            delta_tokens = size2 - size1
            delta_time = (ttft2 - ttft1) * 1000  # Convert to milliseconds

            ms_per_token = delta_time / delta_tokens
            per_token_rates.append(ms_per_token)

            lines.append(
                f"{size1:>5} → {size2:>5} tokens: {delta_time:>7.1f}ms / "
                f"{delta_tokens:>4} tokens = {ms_per_token:.3f} ms/token")

        lines.append("")
        avg_rate = statistics.mean(per_token_rates)
        lines.append(f"Average per-token latency: {avg_rate:.3f} ms/token")
        lines.append(f"Range: {min(per_token_rates):.3f} - "
                     f"{max(per_token_rates):.3f} ms/token")

    # Append cost summary if available
    if cost_summary and len(cost_summary) > 0:
        lines.append(cost_summary)

    # Finally, add notes about the benchmark
    lines.append("")
    lines.append("="*80)
    lines.append("NOTES")
    lines.append("="*80)

    # Get information about the proivider and region where model endpoint is
    provider = _identify_provider()
    model_region = _get_model_deployment_region()
    model_desc_str = f"{MODEL} via {provider}{model_region}"

    # Get information about the geography of the API client
    api_client_geo = os.getenv("API_CLIENT_GEOGRAPHY", "Singapore")

    # Construct the lines
    lines.append("- Measurements include network latency")
    if provider == "OpenRouter":
        lines.append("- Infrastructure: OpenRouter shared API "
                     "(not dedicated capacity)")
    else:
        lines.append("- Infrastructure: Configured API endpoint")
    lines.append("- Results will vary based on server load "
                 "and network conditions")
    lines.append(f"- Model: {model_desc_str}")
    lines.append(f"- Tests conducted in: {api_client_geo}")
    lines.append("="*80)

    output = "\n".join(lines) + "\n"
    if results_file:
        results_file.write(output)
    else:
        print(output)


def _measure_ttft(
    *,
    run_id: str,
    prompt: str,
    output_file: Optional[Any] = None,
) -> Optional[float]:
    """
    Measure Time to First Token for a given prompt.
    Returns TTFT in seconds.
    """
    try:
        # Log request metadata (avoid logging full prompt if sensitive)
        _log_request(prompt=prompt, run_id=run_id)

        # Keep track of the start time
        start_time = time.time()
        ttft: Optional[float] = None
        full_text = ""

        with client.responses.stream(
            model=MODEL,
            input=[{"role": "user", "content": prompt}],
        ) as stream:
            stream = stream

            for event in stream:
                log.info(
                    "stream_event",
                    event_type=event.type,
                    run_id=run_id,
                )

                # FIRST TOKEN — this is the TTFT moment
                if event.type == "response.output_text.delta":
                    if ttft is None:
                        ttft = time.time() - start_time
                        log.info(
                            "first_token_received",
                            ttft=ttft,
                            run_id=run_id,
                        )

                    full_text += event.delta
                    _log_response(
                        is_chunk=True,
                        chunk=event.delta,
                        response=None,
                        run_id=run_id,
                    )

                # Final response + usage
                if event.type == "response.completed":
                    _log_response(
                        is_chunk=False,
                        chunk=None,
                        response=event.response,
                        run_id=run_id,
                    )

        return ttft

    except Exception as e:
        msg = f"TTFT measurement failed: {e}"
        log.warning("ttft_error", error=str(e), run_id=run_id)
        if output_file:
            output_file.write(msg + "\n")
        else:
            print(msg)
        return None


def run_benchmark(
        results_file: Any,
        benchmark_run_dt: datetime) -> Tuple[Dict[int, Dict[str, Any]], str]:
    """
    Run the complete benchmark across all prompt sizes.
    """

    # Get information about the proivider and region where model endpoint is
    provider = _identify_provider()
    model_region = _get_model_deployment_region()
    model_desc_str = f"{MODEL} via {provider}{model_region}"

    header = (
        "="*80 + "\nTTFT Micro-Benchmark\n" +
        f"Model: {model_desc_str}\n" +
        f"Iterations per size: {ITERATIONS_PER_SIZE}\n" +
        f"Sleep between requests: {SLEEP_SECONDS_BETWEEN_REQUESTS} seconds\n" +
        f"Benchmark Run Start Date & Time: {benchmark_run_dt.isoformat()}\n" +
        "="*80 + "\n\n"
    )
    if results_file:
        results_file.write(header)
    # Always print header to console
    print(header)

    results: Dict[int, Dict[str, Any]] = {}

    run_id_base = _generate_run_id()
    for size in PROMPT_SIZES:
        run_id = f"{run_id_base}-{size}"
        size_line = f"Testing prompt size: ~{size} tokens. Run ID: {run_id}\n"
        if results_file:
            results_file.write(size_line)
        prompt = _generate_prompt(size)
        ttfts = []

        print(f"Testing ~{size} tokens (run_id={run_id})")

        for i in range(ITERATIONS_PER_SIZE):
            run_line = f"  Run {i+1}/{ITERATIONS_PER_SIZE}... "
            if results_file:
                results_file.write(run_line)
            # Always show run progress on console
            print(run_line, end="")

            ttft = _measure_ttft(
                prompt=prompt,
                output_file=results_file,
                run_id=run_id)

            if ttft is not None:
                ttfts.append(ttft)
                result_line = f"{ttft:.3f}s\n"
            else:
                result_line = "FAILED (TTFT not captured)\n"

            if results_file:
                results_file.write(result_line)
            # Always show result on console
            print(result_line, end="")

            # Small delay between requests to avoid rate limiting
            time.sleep(SLEEP_SECONDS_BETWEEN_REQUESTS)

        if ttfts:
            results[size] = {
                "mean": statistics.mean(ttfts),
                "median": statistics.median(ttfts),
                "min": min(ttfts),
                "max": max(ttfts),
                "stdev": statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
                "measurements": ttfts,
            }

        if results_file:
            results_file.write("\n")
        else:
            print()

    return results, run_id_base


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("Starting TTFT benchmark…\n")

    benchmark_run_dt = datetime.now()
    timestamp = benchmark_run_dt.strftime("%Y%m%d_%H%M%S")
    filename = f"ttft_benchmark_{timestamp}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as results_file:
            # Run the benchmarks
            results, run_id_base = run_benchmark(
                results_file=results_file,
                benchmark_run_dt=benchmark_run_dt)

            # Calculate and display cost summary
            cost_summary = _calculate_and_display_cost(run_id_base=run_id_base)

            # Display results and write to file
            _display_results(
                results=results,
                cost_summary=cost_summary,
                results_file=results_file)

        print(f"\nResults saved to {filename}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")


if __name__ == "__main__":
    main()
