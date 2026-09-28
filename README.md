# CircuitLens 🔬

CircuitLens is a Transformer interpretability and circuit analysis toolkit
designed to inspect what happens inside a Transformer model.

Instead of only looking at the final prediction, CircuitLens analyzes
attention patterns, internal activations, and the effect of controlled
activation interventions.

## Key Features

- Transformer model loading using TransformerLens
- Internal activation collection using hooks
- Attention pattern analysis
- Attention entropy analysis
- Attention head comparison
- Activation patching
- Original vs patched output comparison
- Experiment result storage
- Interactive Streamlit dashboard
- Automated unit testing with pytest

## Architecture

```text
Input Prompt
     |
     v
Transformer Model
     |
     +----> Activation Hooks
     |
     +----> Attention Analyzer
     |
     +----> Activation Collector
     |
     v
Analysis & Metrics
     |
     v
Activation Patching
     |
     v
Original vs Patched Output
     |
     +----> Streamlit Dashboard
     |
     +----> Experiment Storage
```
