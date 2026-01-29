# Benchmark Prompt (Free-Form)

Default:

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --max-tokens 64 --no-tools --local-jsonl StreetMathDataset/data/street_math_test.jsonl --output runs/streetmath_freeform.jsonl`

With hints:

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --max-tokens 64 --no-tools --hint --local-jsonl StreetMathDataset/data/street_math_test.jsonl --output runs/streetmath_freeform_hint.jsonl`

## Layerwise Metrics (optional)

- Available metrics for `--layerwise-metrics`:

  - `spectral_entropies`
  - `effective_ranks`
  - `activation_entropies`
  - `trace_covariances`
  - `gradient_norms` (variance-based proxy)
  - `cosine_similarities`
  - `l2_distances`
  - `angular_distances`
- Collect only specific metrics and write to a separate JSONL (compressed):

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --max-tokens 256 --no-tools --layerwise --layerwise-output runs/streetmath_layerwise.jsonl.gz --layerwise-metrics spectral_entropies,effective_ranks,gradient_norms,trace_covariances --layerwise-max-tokens 128`

- Summary-only aggregation (smallest output):

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --max-tokens 256 --no-tools --layerwise --layerwise-summary-only --layerwise-output runs/streetmath_layerwise_summary.jsonl`

## Resume + Logging (for large runs)

- Resume from an existing output file:

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --output runs/streetmath.jsonl --resume`

- Resume and re-run error cases:

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --output runs/streetmath.jsonl --resume --resume-errors`

- Log progress/errors and capture raw responses:

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --output runs/streetmath.jsonl --log-file runs/streetmath.log --log-raw-response`

## Quick Subsampling (per topic)

- Run only a percentage of samples per topic (for fast checks):

`python -m streetmath_benchmark.cli --provider transformers --model Qwen/Qwen2-7B-Instruct --device cuda:0 --output runs/streetmath_quick.jsonl --samples-percentage 10`

3) Files are written under `data/`.

## Send Requests to vLLM on Runpod

`python -m streetmath_benchmark.cli     --provider openai     --model Qwen/Qwen2-7B-Instruct     --base-url https://api.runpod.ai/v2/uwwjl36ui3dzs0/run    --max-tokens 256     --local-jsonl StreetMathDataset/data/street_math_test.jsonl     --output ../runs/vllm_qwen2_5_3b_instruct.jsonl`

Replace <RUNPOD_IP>:8000 with your vLLM endpoint (e.g., http://localhost:8000/v1 if you port‑forward).
