# SSH-Hybrid: Do SSM Spectral Properties Predict Safety Degradation in Hybrid Architectures?

An empirical investigation of architectural safety in hybrid SSM-Transformer models.

## Overview

SSH-Hybrid investigates whether the spectral decay inherent in State Space Model (SSM) layers creates exploitable safety vulnerabilities in hybrid SSM-Transformer models (e.g., Jamba-1.5, Zamba-7B). The primary metric is **Attack Success Rate (ASR)** — whether models actually produce harmful output — not representation-level proxies.

### Four Contributions

1. **Context-Distance Safety Degradation (Experiment 6):** Tests whether hybrid models lose safety alignment at longer context distances compared to pure Transformers. This is the key novel experiment.
2. **Spectral Radius as Safety Predictor (Experiment 2):** Tests whether the measured spectral radius rho predicts the distance at which safety degrades.
3. **MBCA Effectiveness (Experiment 7):** Proposes Monotone Boolean Composition Accumulator (MBCA) and measures whether it actually reduces ASR.
4. **Open-Source Audit Pipeline:** Measures spectral radius and deploys MBCA for any hybrid model.

## Key Limitations

- **CHSS measures representation shift, not safety.** Cosine similarity between hidden states under clean vs. perturbed conditions is a proxy metric. It should be validated against behavioral safety metrics (attack success rate).
- **Propositions are hypotheses, not proven bounds.** The core assumption that hidden-state decay implies proportional safety degradation is approximate — attention layers and residual connections may compensate.
- **Safety margin baselines should be measured empirically**, not assumed (delta_star_transformer=1.0 is a normalization placeholder).
- **Evaluation data must be scaled** beyond hardcoded samples for statistical validity.

## Models

| Model | r_SSM | Role |
|-------|-------|------|
| Jamba-1.5-Mini | 0.875 | Primary hybrid (most SSM-heavy) |
| Zamba-7B | 0.850 | Secondary hybrid |
| Pythia-2.8B | 0.000 | Pure Transformer baseline |
| Mamba-2.8B | 1.000 | Pure SSM baseline |

## Installation

```bash
pip install -e .

# For Mamba model support:
pip install -e ".[mamba]"

# For lm-eval benchmark support:
pip install -e ".[eval]"

# For development:
pip install -e ".[dev]"
```

## Project Structure

```
ssh_hybrid/
├── spectral/           # Spectral analysis (radius, horizon, margin propositions)
│   ├── radius.py       # Spectral radius measurement from model weights
│   ├── horizon.py      # Safety memory horizon H(rho) and attenuation factor
│   └── margin.py       # Proposition 1 & 2 implementations
├── models/             # Model loading and configuration
│   ├── config.py       # Model registry (Jamba, Zamba, Pythia, Mamba)
│   └── loader.py       # HuggingFace model loading
├── mbca/               # Monotone Boolean Composition Accumulator
│   ├── register.py     # MBCA register with monotone OR updates
│   ├── probes.py       # Per-category safety probe training on BeaverTails
│   └── monitor.py      # Runtime safety monitoring with KV cache
├── evaluation/         # Evaluation pipelines
│   ├── chss.py         # Content-Hidden State Score (representation shift metric)
│   ├── hispa.py        # HiSPA trigger evaluation
│   ├── safety_margin.py # Behavioral safety measurement (attack success rate)
│   └── benchmarks.py   # lm-eval-harness integration (subprocess + in-process)
├── audit/              # Architectural safety audit
│   └── procedure.py    # Full 3-phase audit procedure
└── experiments/        # Experiment scripts (1-7)
    ├── exp1_spectral_radius.py      # Spectral radius measurement
    ├── exp2_theorem_validation.py   # Spectral prediction validation
    ├── exp3_mbca_coverage.py        # MBCA coverage measurement
    ├── exp4_audit_validation.py     # Audit ranking validation
    ├── exp5_k_sensitivity.py        # MBCA K sensitivity analysis
    ├── exp6_distance_decay.py       # KEY: Context-distance safety degradation (ASR)
    └── exp7_mbca_asr.py             # MBCA effectiveness via ASR
```

## Running Experiments

### Experiment 1: Spectral Radius Measurement
```bash
python -m ssh_hybrid.experiments.exp1_spectral_radius --device auto
```

### Experiment 2: Proposition 1 Validation
```bash
python -m ssh_hybrid.experiments.exp2_theorem_validation --device auto
```

### Experiment 3: MBCA Coverage
```bash
python -m ssh_hybrid.experiments.exp3_mbca_coverage --model jamba-1.5-mini --K 8 16
```

### Experiment 4: Audit Validation
```bash
python -m ssh_hybrid.experiments.exp4_audit_validation --device auto
```

### Experiment 5: K Sensitivity
```bash
python -m ssh_hybrid.experiments.exp5_k_sensitivity --K-values 4 8 12 16 24
```

### Experiment 6: Context-Distance Safety Degradation (KEY EXPERIMENT)
```bash
python -m ssh_hybrid.experiments.exp6_distance_decay --device auto --max-prompts 500
```

### Experiment 7: MBCA Effectiveness via ASR
```bash
python -m ssh_hybrid.experiments.exp7_mbca_asr --model jamba-1.5-mini --K 4 8 12 16
```

## Running Tests

```bash
pytest tests/ -v
```

## Acceptance Criteria (Empirical)

| KPI | Target | Metric |
|-----|--------|--------|
| Distance-decay effect exists | ASR increases >10pp at d=800 vs d=0 for hybrid models | ASR (behavioral) |
| Transformer control is flat | ASR increase <5pp for Pythia across all distances | ASR (behavioral) |
| Spectral radius predicts degradation distance | Pearson r > 0.70 between H(rho) and d* | Correlation |
| MBCA reduces ASR | ASR reduction >15pp with K=8 | ASR (behavioral) |
| MBCA false positive rate | <5% on benign prompts | FP rate |
| H(rho) computation time | <100ms per model | Wall clock |

**Note:** All results must be generated by running experiments. The `results/` directory is created on first run. No pre-existing results are included.
