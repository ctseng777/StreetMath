# Customizing Model Configurations in StreetMath Experiments

This guide explains how to customize model configurations, including changing the number of layers, when running StreetMath benchmark experiments.

## Overview

The `TransformersProvider` now supports customizing model configurations before loading. This is useful for:
- Experimenting with different model sizes
- Reducing memory usage by using fewer layers
- Testing architectural variations

## Usage

### Command Line

You can customize the number of layers using the `--num-layers` argument:

```bash
python -m streetmath_benchmark.cli \
    --provider transformers \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-layers 12 \
    --output results.jsonl
```

### Programmatic Usage

If you're using the `TransformersProvider` directly in your code:

```python
from streetmath_benchmark.providers import TransformersProvider

# Load model with custom number of layers
provider = TransformersProvider(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    num_layers=12,
    device="cuda:0",
    torch_dtype="float16"
)

# Use the provider as normal
result = provider.generate("What is 2+2?")
```

### Additional Config Customizations

You can also customize other configuration parameters:

```python
provider = TransformersProvider(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    num_layers=8,
    hidden_size=2048,  # If supported by the model
    num_attention_heads=16,  # If supported by the model
    device="cuda:0"
)
```

## Important Notes

### Layer Modification Behavior

When you change the number of layers:

1. **Reducing layers**: The model will truncate the extra layers. The first N layers (where N is your specified number) will be loaded from the pretrained weights.

2. **Increasing layers**: New layers will be randomly initialized. The pretrained layers will be loaded, and additional layers will be added with random weights.

3. **Performance**: Models with modified layer counts may not perform well without fine-tuning. The pretrained weights were trained with a specific architecture.

### Supported Parameters

The most commonly customizable parameters include:
- `num_layers` / `num_hidden_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension size
- `intermediate_size`: Feed-forward network size
- `num_attention_heads`: Number of attention heads
- `max_position_embeddings`: Maximum sequence length

Note: Not all parameters can be safely modified. Some models have architectural constraints.

## Example Use Cases

### 1. Memory-Constrained Experiments

```bash
# Use fewer layers to fit in memory
python -m streetmath_benchmark.cli \
    --provider transformers \
    --model large-model-name \
    --num-layers 6 \
    --device cuda:0 \
    --output results.jsonl
```

### 2. Architecture Exploration

```bash
# Test different layer counts
for layers in 4 8 12 16; do
    python -m streetmath_benchmark.cli \
        --provider transformers \
        --model base-model \
        --num-layers $layers \
        --output results_${layers}layers.jsonl
done
```

### 3. Full Example with All Options

```bash
python -m streetmath_benchmark.cli \
    --provider transformers \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-layers 12 \
    --device cuda:0 \
    --torch-dtype float16 \
    --trust-remote-code \
    --temperature 0.2 \
    --max-tokens 512 \
    --output results.jsonl \
    --dataset LuxMuseAI/StreetMathDataset \
    --split test
```

## Technical Details

### How It Works

1. The `TransformersProvider` loads the original model configuration using `AutoConfig.from_pretrained()`
2. Modifies the specified parameters (e.g., `num_hidden_layers`)
3. Loads the model with the modified configuration
4. HuggingFace Transformers automatically handles:
   - Loading pretrained weights for matching layers
   - Initializing new layers (if increasing)
   - Truncating layers (if decreasing)

### Limitations

- **Weight Compatibility**: When reducing layers, only the first N layers are kept. When increasing, new layers are randomly initialized.
- **Model-Specific**: Some models may not support all configuration modifications.
- **Performance**: Modified models typically need fine-tuning to work well.

## Troubleshooting

### Error: "Config does not have attribute 'X'"
Some models don't support all configuration parameters. Check the model's config class for available parameters.

### Poor Performance After Modification
This is expected! Models with modified architectures need fine-tuning. The pretrained weights were optimized for the original architecture.

### Memory Issues
If you're still running out of memory:
- Reduce `--num-layers` further
- Use `--device cpu` to run on CPU (slower but uses less GPU memory)
- Reduce `--max-tokens`
- Use a smaller model

## See Also

- `streetmath_benchmark/README.md` - Main benchmark documentation
- `streetmath_benchmark/providers.py` - Provider implementation details

