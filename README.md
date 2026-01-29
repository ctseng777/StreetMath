# StreetMath

Repository structure and where to look for each part of the project.

## Top-level layout

```
BenchmarkAnalysis/         Scripts and outputs for benchmark tables/plots
BenchmarkAndLayerwiseRuns/ Raw run artifacts for benchmarks and layerwise analyses
LayerwiseAnalysis/         Scripts and outputs for layerwise figures/appendix
LinearProbe/               Linear probe experiments, results, and plotting
PruneStudyAnalysis/        Analysis code, plots, and summaries for prune studies
PruneStudyRuns/            Raw run artifacts for prune studies
StreetMathDataset/         Dataset assets, data, and dataset-related docs
README.md                  This file
```

## Directory details

### `BenchmarkAnalysis/`

- `generate_benchmark_tables.py`: Build LaTeX tables for benchmark results.
- `plot_benchmark_overall.py`: Generate overall benchmark plots.
- `Benchmark_overall.tex`, `Benchmark_by_subtopic.tex`: LaTeX outputs.
- `plots/`: Generated figures.
- `energy_consumption_table`: Energy/compute table artifact.

### `BenchmarkAndLayerwiseRuns/`

- `gsm8k-runs/`, `streetmath-runs/`: Raw run outputs used by benchmark and layerwise analyses.

### `LayerwiseAnalysis/`

- `generate_layerwise_figures.py`: Produce layerwise analysis figures.
- `appendix_d_figures.tex`: LaTeX output for appendix figures.
- `figures/`, `analysis/`, `averages/`: Generated assets and intermediates.

### `LinearProbe/`

- `setup_linear_probe_experiments.py`: Configure/launch linear probe experiments.
- `analyze_probe_results.py`: Aggregate and analyze probe runs.
- `plot_acc_per_layer.py`, `plot_accuracy_by_number.py`: Plotting utilities.
- `generate_latex_tables.py`: LaTeX table generation.
- `results/`, `tables/`, `figures/`, `analysis_output/`: Generated outputs.
- `0_*` directories: Per-model run folders.
- `mapping.txt`: Label/value mapping used by probes.
- `requirements.txt`: Local deps for probe analysis.

### `PruneStudyAnalysis/`

- `analyze_prune_study.py`: Analyze pruning experiments.
- `plot_model_correlations.py`: Correlation plots across models.
- `plots/`, `summaries/`: Generated outputs.

### `PruneStudyRuns/`

- Per-model folders for pruning experiment outputs (e.g., `Falcon-H1-7B-Instruct*`, `qwen2.5-3b-instruct*`).

### `StreetMathDataset/`

- `streetmath_dataset/`, `streetmath_benchmark/`: Core dataset assets.
- `data/`: Data files used by dataset/benchmark.
- `CUSTOMIZE_MODEL.md`: Notes on adapting models to the dataset.
- `requirements.txt`: Local deps for dataset scripts.

## Using the StreetMath Benchmark

The benchmark runner lives in `StreetMathDataset/streetmath_benchmark`. It can load the dataset from Hugging Face (requires network) or from a local JSONL copy via `--local-jsonl`.

Run with transformers (local model):

Without hint

```
```bash
python -m streetmath_benchmark.cli \
    --provider transformers \
    --model qwen/qwen2.5-32b-instruct \
    --max-tokens 256 \
    --local-jsonl data/street_math_test.jsonl \
    --output ../runs/qwen2_5_32b_instruct_hint/qwen2_5_32b_instruct_hint.jsonl \
    --log-file ../runs/qwen2_5_32b_instruct_hint/qwen2_5_32b_instruct_hint.log \
    --log-raw-response \
    --hint  

```

With hint, record layerwise metrics

```bash
python -m streetmath_benchmark.cli \
    --provider transformers \
    --model qwen/qwen2.5-32b-instruct \
    --max-tokens 256 \
    --local-jsonl data/street_math_test.jsonl \
    --output ../runs/qwen2_5_32b_instruct_hint/qwen2_5_32b_instruct_hint.jsonl \
    --log-file ../runs/qwen2_5_32b_instruct_hint/qwen2_5_32b_instruct_hint.log \
    --log-raw-response \
    --layerwise \
    --layerwise-output ../runs/qwen2_5_32b_instruct_hint/qwen2_5_32b_instruct_layerwise_hint.jsonl \
    --layerwise-metrics spectral_entropies,effective_ranks,gradient_norms,trace_covariances \
    --layerwise-max-tokens 256 \
    --hint  

```

Notes

- The runner writes one JSON object per line and appends a final `summary` record with counts and the average score.
- You can customize instructions with `--prompt-template-file` or disable the system prompt via `--no-system`.
- For diffusion models, pass `--arch diffusion` and `--diffusion-steps` to estimate FLOPs.

## Generating the Dataset

The dataset is release at [StreetMath/StreetMath](https://huggingface.co/datasets/StreetMath/StreetMath)

The generator creates questions emphasizing approximation. Outputs are JSONL files like `data/street_math_train.jsonl` and `data/street_math_test.jsonl`.

Recommended invocation from the repo root:

```bash
export PYTHONPATH=StreetMathDataset:$PYTHONPATH
python -m streetmath_dataset.scripts.generate_street_math_dataset --seed 42 --train 1000 --test 1000 --outdir StreetMathDataset/data
```

## MathNeuro

Please visit https://github.com/ctseng777/MathNeuro for the adapted version of [MathNeuro](https://github.com/bryanchrist/MathNeuro). See original repository for usage.
