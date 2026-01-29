# Linear Probe Analysis Summary

## Key Findings

### 1. Architecture-level Patterns
- **Diffusion models**: Late convergence (~48% depth)
- **State-space models**: Early convergence (~34% depth)
- **Autoregressive models**: Early-medium convergence (~30% depth)

### 2. Task-specific Insights
- **Performance gap** (Digits - Words): 0.126
- **Surface-form encoding**: Strong evidence from degraded word performance

### 3. Mathematical Hierarchy
- **Special numbers** (2,5,10): 0.977 avg accuracy
- **Ordinary numbers**: 0.719 avg accuracy
- **Hierarchy confirmed**: True

### 4. Cross-validation
- **Ranking consistency**: 1.000 correlation
- **Robust patterns**: Architecture differences persist across distances

## Implications for StreetMath Paper

1. **Cognitive miserliness absence**: Models use complex pathways even for simple proximity detection
2. **Architecture specialization**: State-space models excel at early numerical pattern recognition
3. **Surface-form limitation**: Critical gap in abstract numerical reasoning
4. **Mathematical intuition**: Models internalize human-like numerical salience hierarchies
