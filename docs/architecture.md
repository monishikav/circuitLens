# CircuitLens Architecture

## Overview

CircuitLens is organized into separate modules for model loading,
internal activation analysis, attention analysis, activation patching,
experiment storage, and visualization.

## Data Flow

```text
Prompt
  ↓
Model Loader
  ↓
TransformerLens Model
  ↓
┌─────────────────────┐
│ Internal Analysis   │
│                     │
│ • Activations       │
│ • Attention         │
│ • Metrics           │
└─────────────────────┘
  ↓
Activation Patching
  ↓
Output Comparison
  ↓
Experiment Storage
  ↓
Streamlit Dashboard
```
