# SSH-Hybrid: Spectral Safety Horizons for Hybrid SSM-Transformer Models

Architecture-specific oversight safety margins via SSM spectral decay analysis.

## Overview

SSH-Hybrid provides a formal framework for quantifying the oversight safety margin deficit in hybrid SSM-Transformer models (e.g., Jamba-1.5, Zamba-7B) compared to pure Transformers. The framework includes:

- **Theorem 1 (Spectral Safety Margin Bound):** Relates the safety margin of hybrid models to the SSM spectral radius and layer fraction
- **Theorem 2 (MBCA Compensation):** Proves that a Monotone Boolean Composition Accumulator can recover the safety margin deficit
- **Architectural Safety Audit Procedure:** A 3-phase audit for evaluating any hybrid model's safety margin

## Key Results

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
├── spectral/           # Spectral analysis (radius, horizon, margin theorems)
│   ├── radius.py       # Spectral radius measurement (SpectralGuard methodology)
│   ├── horizon.py      # Safety memory horizon H(rho) and attenuation factor
│   └── margin.py       # Theorem 1 & 2 implementations
├── models/             # Model loading and configuration
│   ├── config.py       # Model registry (Jamba, Zamba, Pythia, Mamba)
│   └── loader.py       # HuggingFace model loading
├── mbca/               # Monotone Boolean Composition Accumulator
│   ├── register.py     # MBCA register with monotone OR updates
│   ├── probes.py       # Safety probe training on BeaverTails
│   └── monitor.py      # Runtime safety monitoring wrapper
├── evaluation/         # Evaluation pipelines
│   ├── chss.py         # Content-Hidden State Score computation
│   ├── hispa.py        # HiSPA/RoBench-25 evaluation
│   └── benchmarks.py   # lm-eval-harness integration
├── audit/              # Architectural safety audit
│   └── procedure.py    # Full 3-phase audit procedure
└── experiments/        # Experiment scripts (1-5)
    ├── exp1_spectral_radius.py      # Spectral radius measurement
    ├── exp2_theorem_validation.py   # Theorem 1 validation vs HiSPA
    ├── exp3_mbca_coverage.py        # MBCA coverage measurement
    ├── exp4_audit_validation.py     # Audit ranking validation
    └── exp5_k_sensitivity.py        # MBCA K sensitivity analysis
```

## Running Experiments

### Experiment 1: Spectral Radius Measurement
```bash
python -m ssh_hybrid.experiments.exp1_spectral_radius --device auto
```

### Experiment 2: Theorem 1 Validation
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

## Running Tests

```bash
pytest tests/ -v
```

## Acceptance Criteria

| KPI | Target |
|-----|--------|
| Theorem 1 prediction accuracy | Pearson r > 0.80, MAE < 15% |
| Model ranking accuracy | Correct ordinal: Pythia > Zamba > Jamba > Mamba |
| MBCA coverage (beta_MBCA) | > 0.70 |
| MBCA benign degradation | < 2% absolute on MMLU/HellaSwag/ARC |
| CHSS recovery with MBCA | < 25% degradation (63% recovery) |
| H(rho) computation time | < 100ms per model |
