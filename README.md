# SSH-Hybrid: Spectral Safety Horizons for Hybrid SSM-Transformer Models

Architecture-specific oversight safety margins via SSM spectral decay analysis.

## Overview

SSH-Hybrid provides a framework for quantifying the predicted oversight safety margin deficit in hybrid SSM-Transformer models (e.g., Jamba-1.5, Zamba-7B) compared to pure Transformers. The framework includes:

- **Proposition 1 (Spectral Safety Margin Bound):** Hypothesizes a relationship between safety margin and the SSM spectral radius / layer fraction. This rests on the assumption that safety margin degrades proportionally to hidden-state signal attenuation — a testable hypothesis, not a proven theorem.
- **Proposition 2 (MBCA Compensation):** Proposes that a Monotone Boolean Composition Accumulator can recover the safety margin deficit, with recovery proportional to empirically measured coverage.
- **Architectural Safety Audit Procedure:** A 3-phase audit for evaluating any hybrid model's safety margin.

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
└── experiments/        # Experiment scripts (1-5)
    ├── exp1_spectral_radius.py      # Spectral radius measurement
    ├── exp2_theorem_validation.py   # Proposition 1 validation
    ├── exp3_mbca_coverage.py        # MBCA coverage measurement
    ├── exp4_audit_validation.py     # Audit ranking validation
    └── exp5_k_sensitivity.py        # MBCA K sensitivity analysis
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

## Running Tests

```bash
pytest tests/ -v
```

## Acceptance Criteria

| KPI | Target |
|-----|--------|
| Proposition 1 prediction accuracy | Pearson r > 0.80, MAE < 15% |
| Model ranking accuracy | Correct ordinal: Pythia > Zamba > Jamba > Mamba |
| MBCA coverage (beta_MBCA) | > 0.70 |
| MBCA benign degradation | < 2% absolute on MMLU/HellaSwag/ARC |
| CHSS recovery with MBCA | < 25% degradation (63% recovery) |
| H(rho) computation time | < 100ms per model |

**Note:** These acceptance criteria should be validated by running experiments and committing results. The `results/` directory is generated on first run.
