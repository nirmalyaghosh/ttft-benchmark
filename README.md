# TTFT Micro-Benchmark

A simple, reproducible tool for measuring Time to First Token (TTFT) across different prompt sizes and LLM providers.

This benchmark script was created to empirically validate theoretical per-token latency figures from published sources. The results reveal insights about when prompt size optimisation matters — and when it does not.

**Related blog posts:**
- [How Prompt Size Directly Impacts LLM Response Latency](https://www.nirmalya.net/posts/2026/02/prompt-size-impact-on-llm-response/)
- [TTFT Optimisation: Practical Patterns](https://www.nirmalya.net/posts/2026/02/ttft-optimisation-practical-patterns/)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/nirmalyaghosh/ttft-benchmark.git
cd ttft-benchmark

# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment:
# (if using Windows)
.\.venv\Scripts\activate
# (if using macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp ttft_benchmark.env.example ttft_benchmark.env
# Edit ttft_benchmark.env with your API credentials

# Run the standard benchmark (TTFT vs prompt size)
python ttft_benchmark.py

# Run the prefix caching benchmark (cache-miss vs cache-hit TTFT)
python ttft_benchmark.py --prefix-caching
```

Notes:
- If using Azure OpenAI,
  - refer to [https://learn.microsoft.com/en-us/azure/reliability/regions-list](https://learn.microsoft.com/en-us/azure/reliability/regions-list) when setting `MODEL_ENDPOINT_DEPLOYMENT_REGION`.
  - refer to model deployment when setting the `API_VERSION`.
- Experiment with `SLEEP_SECONDS_BETWEEN_REQUESTS` to replicate cold-start effects.

---

## What This Measures

The script measures **Time to First Token (TTFT)**, the latency between sending a request and receiving the first token of the response. This metric matters for user-perceived responsiveness in conversational AI applications.

### Standard mode (default)

Tests five prompt sizes (100, 500, 1,000, 2,000, and 5,000 tokens) with 10 iterations each, calculating:
- Mean, median, min, max, and standard deviation per size
- Per-token latency between consecutive prompt sizes
- Total token usage for cost estimation

### Prefix caching mode (`--prefix-caching`)

Measures the TTFT difference between cache-miss and cache-hit requests across configurable prefix sizes (default: 1,500, 3,000, 5,000 tokens). For each prefix size:
- 1 warmup request (discarded)
- N=3 cache-miss measurements (distinct system prompts to avoid inadvertent caching)
- N=20 cache-hit measurements (same prefix, varying user queries)
- Reports P50, P95, mean, stdev for both miss and hit distributions
- Verifies cache hits via the API response's `cached_tokens` field
- Compares measured TTFT reduction against arithmetic estimates ("Calculated vs Measured")

---

## Methodology

### Test Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Prompt sizes | 100, 500, 1K, 2K, 5K tokens | Covers typical use cases without excessive cost |
| Iterations per size | 10 | Balances statistical validity against API costs |
| Inter-request delay | 15 seconds | Minimises rate-limiting effects |
| Prompt content | Repetitive text | Ensures consistent tokenisation across runs |

The prompts use "*The quick brown fox jumps over the lazy dog*" repeated to target length. Content does not affect TTFT, only token count matters.

### What's Included in the Measurement

TTFT as measured here includes:
- Network round-trip latency (client → endpoint → client)
- API gateway processing
- Queue wait time (on shared infrastructure)
- Model loading/warm-up (if applicable)
- Prompt processing (prefill phase)
- Time to generate first output token

This is deliberate. Production applications experience all these factors, not just isolated model processing time.

### What's NOT Controlled

- Server-side load (varies by time of day, other users)
- Network routing variations
- Provider-side caching or optimisation

These uncontrolled factors are the point: they represent real-world conditions most developers encounter.

### Prefix Caching Test Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Prefix sizes | 1,500, 3,000, 5,000 tokens | Covers range above OpenAI's 1,024-token caching threshold |
| Cache-miss iterations | 3 | Distinct system prompts per measurement to avoid inadvertent caching |
| Cache-hit iterations | 20 | Sufficient for P50/P95 reporting |
| Warmup | 1 request per prefix size | Avoids TCP/TLS handshake contamination |
| Inter-request delay | 15 seconds | Matches standard mode; avoids rate limiting |
| System prompt content | Padded instructional text | Ensures consistent tokenisation above caching threshold |

The benchmark validates cache hits by checking `cached_tokens` in the API response usage data. A "Calculated vs Measured" comparison surfaces the gap between arithmetic estimates (which assume prefill cost dominates) and empirical results (which include network latency and infrastructure overhead).

---

## Results: Five Models Compared

Tests conducted from Singapore, February 2026.

### Summary Table

| Model | Provider | Endpoint Region | Mean TTFT Range | Avg Per-Token Rate | Pattern |
|-------|----------|-----------------|-----------------|-------------------|---------|
| Claude Sonnet 4.5 | OpenRouter | - | 2.8-3.1s | Negative | Infrastructure variance dominates |
| gpt-4o-mini | Azure | Australia East | 1.11-1.41s | -0.030 ms/token | Near-flat with cold-start spike |
| gpt-4.1-mini | Azure | Australia East | 1.21-1.36s | **+0.017 ms/token** | Only positive average |
| gpt-4.1-nano | Azure | South India | 0.61-0.84s | -0.133 ms/token | Inverted (larger prompts faster) |
| gpt-5-nano | Azure | South India | 1.28-2.13s | -0.436 ms/token | Strongly inverted |

---

## Analysis

### Key Finding: Infrastructure Variance Dominates

The data leads to an uncomfortable but important conclusion: **on shared API infrastructure, optimising prompt size for TTFT is largely futile.**

Here's the math. At the theoretical per-token rate of 0.20ms (from [Glean's PTU benchmarks](https://glean.com/blog/how-we-built-low-latency-ai)), reducing a prompt from 5,000 to 1,000 tokens saves:

```
4,000 tokens × 0.20ms = 800ms
```

But these benchmarks show standard deviation typically between 200-400ms per request on shared infrastructure, with some models exhibiting variance exceeding 1,000ms. The optimisation signal (800ms) is smaller than the variance observed in these benchmarks, where standard deviation ranged from 200ms to over 1,000ms. One cannot reliably measure nor guarantee that improvement in production.

```
StdDev ranges from the detailed results from the tttf_benchmark.py script:

gpt-4o-mini: 211ms to 427ms
gpt-4.1-mini: 194ms to 655ms
gpt-4.1-nano: 47ms to 732ms
gpt-5-nano: 137ms to 1,088ms
```


### The Practical Optimisation Hierarchy

| Priority | Optimisation | Expected Impact | When It Matters |
|----------|--------------|-----------------|-----------------|
| 1 | Infrastructure (dedicated vs shared) | High (variance reduction) | Always |
| 2 | Geographic routing (endpoint proximity) | Medium (network latency) | Multi-region deployments |
| 3 | Prompt caching (if available) | High (50-90% TTFT reduction) | Repeated prompt prefixes |
| 4 | Prompt size reduction | Low on shared infrastructure | **Only on dedicated infrastructure** |


### Why Negative Per-Token Rates?

4 of 5 models showed larger prompts completing at similar or *faster* speeds than smaller ones. This is not measurement error, but real infrastructure behaviour:

1. **Cold-start effects** disproportionately hit the first test (smallest prompts)
2. **Connection warm-up** amortises across subsequent requests
3. **Queue variability** on shared infrastructure adds ±500ms noise
4. **The actual per-token signal** (which ranged from -0.15 to +0.14 ms/token across different models and prompt size transitions) is too small to reliably detect through this infrastructure noise

### The Cold-Start Pattern

A consistent observation across all models: Run 1 of the smallest prompt size shows dramatically higher latency.

| Model | Run 1 (100 tokens) | Runs 2-10 Average |
|-------|-------------------|-------------------|
| gpt-4.1-nano | 2.92s | 0.61s |
| gpt-5-nano | 4.65s | 1.68s |
| gpt-4o-mini | 2.02s | 1.07s |
| gpt-4.1-mini | 3.06s | 0.99s |

This suggests connection establishment, model loading, or infrastructure warm-up adds 1-3 seconds to the first request.

> For latency-sensitive applications, consider strategies such as:
> - Warm-up requests to avoid cold-start penalties
> - Using dedicated infrastructure where cold starts are controlled
> - Implementing request retry logic for outlier latencies

### When Per-Token Latency DOES Matter

Some models showed positive per-token latency under certain conditions, confirming the theoretical relationship exists. However, this signal becomes measurable and actionable primarily when:

- Using **dedicated capacity** with controlled infrastructure variance (for example, when using Azure PTU, reserved instances)
- With **controlled network paths** to minimize external noise (for example, same-region deployment, subject to model & quota availability)
- At **scale** where small per-request savings compound across many requests

For most developers using shared API endpoints, the infrastructure variance dominates any optimisation from prompt size reduction.

---

## Results: Prefix Caching

Test conducted from Singapore against Azure OpenAI (gpt-4o-mini, Australia East), March 2026.

### Summary Table

| Prefix Size | Cache Miss (mean) | Cache Hit (P50) | Cache Hit (P95) | Measured Reduction | Cached |
|---|---|---|---|---|---|
| ~1,500 tokens | 1.657s | 1.577s | 1.645s | 4.9% | 19/20 |
| ~3,000 tokens | 1.473s | 1.330s | 1.628s | 9.7% | 20/20 |
| ~5,000 tokens | 1.744s | 1.387s | 1.651s | 20.5% | 20/20 |

### Calculated vs Measured

| Prefix Size | Calculated Reduction | Measured Reduction | Delta |
|---|---|---|---|
| ~1,500 tokens | 99.0% | 4.9% | -94.1pp |
| ~3,000 tokens | 99.5% | 9.7% | -89.8pp |
| ~5,000 tokens | 99.7% | 20.5% | -79.2pp |

The "Calculated" column assumes 100% of prefix tokens are cached and only user query tokens (~15) incur prefill cost. The gap is explained by network round-trip latency (Singapore to Australia East) and infrastructure overhead, which represent a large, roughly fixed portion of TTFT. Prefix caching eliminates server-side prefill compute, but this saving is small relative to the total latency budget on a shared API endpoint.

The measured reduction scales with prefix size (5% to 20%), confirming that caching is working, but bounded by network and infrastructure latency. For self-hosted deployments where network latency is negligible, the measured reduction would approach the calculated estimate.

This finding reinforces the Practical Optimisation Hierarchy from the standard benchmark: infrastructure tier and geographic routing dominate on shared endpoints.

---

## Per-Token Latency: Which Figure to Use?

Published benchmarks report figures ranging from 0.04ms to 0.24ms → a 6x variance explained by infrastructure type:

| Source | Figure | Infrastructure |
|--------|--------|----------------|
| Talkative | 0.04-0.06 ms/token | OpenAI standard API (shared) |
| Glean | 0.20-0.24 ms/token | Azure PTU (dedicated) |
| This benchmark | -0.03 to +0.14 ms/token | Azure OpenAI (shared) |

**Recommendation:**

- **Dedicated infrastructure:** Use 0.10-0.25 ms/token for capacity planning
- **Shared infrastructure:** Treat per-token latency as negligible; focus on caching and infrastructure tier

---

## Output

### Benchmark script (`ttft_benchmark.py`)

1. **Console output** — real-time progress
2. **JSONL log** — `logs/ttft_benchmark_data.jsonl` with per-request details for further analysis
3. **Results file** — `benchmark-results/ttft_benchmark_YYYYMMDD_HHMMSS.txt` with full statistics

### Analysis script (`analyze_logs.py`)

Parses the JSONL logs produced by the benchmark script and generates:

1. **`benchmark-results/analysis_summary.txt`** — human-readable report with summary statistics (mean, median, stddev, 95% CI, P95), linear regression per model (slope in ms/token, R-squared, p-value), and data sufficiency check (target: N >= 30 per condition)
2. **`benchmark-results/analysis_summary.json`** — machine-readable equivalent with timestamped per-measurement data
3. **`benchmark-results/analysis_data.csv`** — one row per measurement (model, prompt_tokens, ttft_seconds, timestamp_utc), suitable for scatter plots and external statistical tools

```bash
# Run analysis (requires venv with structlog)
python analyze_logs.py
```

### Sample Contents Of `.jsonl` File Generated
```jsonl
{"model": "gpt-5-nano", "prompt": "This prompt is used to measure Time to First Token (TTFT).\n Respond only with 'OK'.\n\n LONG TEXT STARTS BELOW:\n\nThe quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy d...", "num_words": 3765, "approximate_num_tokens": 5020, "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "timestamp": "2026-02-03T15:54:22.728853Z", "event": "api_request"}
{"event_type": "response.created", "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "stream_event", "timestamp": "2026-02-03T15:54:22.996480Z"}
{"event_type": "response.output_text.delta", "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "stream_event", "timestamp": "2026-02-03T15:54:23.822165Z"}
{"ttft": 1.0933125019073486, "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "first_token_received", "timestamp": "2026-02-03T15:54:23.822165Z"}
{"model": "gpt-5-nano", "chunk": "OK", "usage": null, "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "timestamp": "2026-02-03T15:54:23.822165Z", "event": "stream_chunk"}
{"event_type": "response.output_text.done", "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "stream_event", "timestamp": "2026-02-03T15:54:24.000370Z"}
{"event_type": "response.content_part.done", "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "stream_event", "timestamp": "2026-02-03T15:54:24.000370Z"}
{"event_type": "response.output_item.done", "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "stream_event", "timestamp": "2026-02-03T15:54:24.000370Z"}
{"event_type": "response.completed", "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "event": "stream_event", "timestamp": "2026-02-03T15:54:24.000370Z"}
{"model": "gpt-5-nano", "chunk": null, "usage": {"input_tokens": 4198, "input_tokens_details": {"cached_tokens": 4096}, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 4205}, "run_id": "56fa5a4e-6593-45f5-b215-107582f58a31-5000", "timestamp": "2026-02-03T15:54:24.000370Z", "event": "api_response"}
```
Note:
- Some lines omitted for brevity
- A new file will be created for each day with date based suffix (for example `ttft_benchmark_data.jsonl.20260204`), and kept for the number of days as set in the environment variable `LOG_FILE_BACKUP_COUNT`


### Sample Contents Of Results File Generated
```
================================================================================
TTFT Micro-Benchmark
Model: gpt-5-nano via Azure OpenAI (deployed at South India)
Iterations per size: 10
Sleep between requests: 15.0 seconds
Benchmark Run Start Date & Time: 2026-02-03T23:40:34.840399
================================================================================

Testing prompt size: ~100 tokens. Run ID: 56fa5a4e-6593-45f5-b215-107582f58a31-100
  Run 1/10... 4.652s
  Run 2/10... 1.683s
  Run 3/10... 1.460s
  Run 4/10... 1.603s
  Run 5/10... 1.755s
  Run 6/10... 3.307s
  Run 7/10... 1.432s
  Run 8/10... 1.589s
  Run 9/10... 1.217s
  Run 10/10... 2.603s

Testing prompt size: ~500 tokens. Run ID: 56fa5a4e-6593-45f5-b215-107582f58a31-500
  Run 1/10... 1.377s
  Run 2/10... 1.465s
  Run 3/10... 1.395s
  Run 4/10... 1.362s
  Run 5/10... 1.114s
  Run 6/10... 1.640s
  Run 7/10... 1.451s
  Run 8/10... 1.286s
  Run 9/10... 1.469s
  Run 10/10... 1.331s

Testing prompt size: ~1000 tokens. Run ID: 56fa5a4e-6593-45f5-b215-107582f58a31-1000
  Run 1/10... 1.309s
  Run 2/10... 1.913s
  Run 3/10... 1.607s
  Run 4/10... 1.624s
  Run 5/10... 1.408s
  Run 6/10... 1.138s
  Run 7/10... 1.176s
  Run 8/10... 1.545s
  Run 9/10... 1.293s
  Run 10/10... 1.577s

Testing prompt size: ~2000 tokens. Run ID: 56fa5a4e-6593-45f5-b215-107582f58a31-2000
  Run 1/10... 2.005s
  Run 2/10... 1.433s
  Run 3/10... 1.286s
  Run 4/10... 1.287s
  Run 5/10... 1.567s
  Run 6/10... 1.415s
  Run 7/10... 1.528s
  Run 8/10... 1.207s
  Run 9/10... 2.154s
  Run 10/10... 1.152s

Testing prompt size: ~5000 tokens. Run ID: 56fa5a4e-6593-45f5-b215-107582f58a31-5000
  Run 1/10... 1.272s
  Run 2/10... 1.100s
  Run 3/10... 1.406s
  Run 4/10... 1.506s
  Run 5/10... 1.076s
  Run 6/10... 1.079s
  Run 7/10... 1.461s
  Run 8/10... 1.606s
  Run 9/10... 1.162s
  Run 10/10... 1.093s

================================================================================
RESULTS
================================================================================

Prompt Size     Mean TTFT       Median TTFT     Min        Max        StdDev
--------------------------------------------------------------------------------
100             2.130s         1.643s         1.217s    4.652s    1.088s
500             1.389s         1.386s         1.114s    1.640s    0.137s
1000            1.459s         1.477s         1.138s    1.913s    0.239s
2000            1.504s         1.424s         1.152s    2.154s    0.333s
5000            1.276s         1.217s         1.076s    1.606s    0.202s

================================================================================
PER-TOKEN LATENCY ANALYSIS
================================================================================

Calculating per-token latency between consecutive prompt sizes:

  100 →   500 tokens:  -741.0ms /  400 tokens = -1.853 ms/token
  500 →  1000 tokens:    70.1ms /  500 tokens = 0.140 ms/token
 1000 →  2000 tokens:    44.5ms / 1000 tokens = 0.044 ms/token
 2000 →  5000 tokens:  -227.5ms / 3000 tokens = -0.076 ms/token

Average per-token latency: -0.436 ms/token
Range: -1.853 - 0.140 ms/token

================================================================================
COST SUMMARY
================================================================================
Run ID Base: 56fa5a4e-6593-45f5-b215-107582f58a31
Model Used: gpt-5-nano
Total Input Tokens: 73,000
Total Output Tokens: 3,870
Total Tokens: 76,870
================================================================================

================================================================================
NOTES
================================================================================
- Measurements include network latency
- Infrastructure: Configured API endpoint
- Results will vary based on server load and network conditions
- Model: gpt-5-nano via Azure OpenAI (deployed at South India)
- Tests conducted in: Singapore
================================================================================
```

---

## Acknowledgements

This benchmark was created as part of preparatory work for my blog post, *[How Prompt Size Directly Impacts LLM Response Latency](https://www.nirmalya.net/posts/2026/02/prompt-size-impact-on-llm-response/)*. The analysis refers to blog posts published by [Glean](https://glean.com/blog/how-we-built-low-latency-ai) and [Talkative](https://talkative.uk/info/gpt4o-latency-and-performance).
