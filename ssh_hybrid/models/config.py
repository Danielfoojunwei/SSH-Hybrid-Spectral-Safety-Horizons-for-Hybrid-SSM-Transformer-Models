"""Model configurations for SSH-Hybrid experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a model used in SSH-Hybrid experiments."""

    name: str
    hf_model_id: str
    model_type: str  # 'jamba', 'zamba', 'mamba', 'pythia'
    r_ssm: float  # fraction of layers that are SSM
    params_billions: float
    vram_gb_bf16: float
    role: str  # 'primary', 'secondary', 'transformer_baseline', 'ssm_baseline'
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "jamba-1.5-mini": ModelConfig(
        name="jamba-1.5-mini",
        hf_model_id="ai21labs/Jamba-v0.1",
        model_type="jamba",
        r_ssm=0.875,
        params_billions=12.0,
        vram_gb_bf16=24.0,
        role="primary",
        trust_remote_code=True,
    ),
    "zamba-7b": ModelConfig(
        name="zamba-7b",
        hf_model_id="Zyphra/Zamba-7B-v1",
        model_type="zamba",
        r_ssm=0.850,
        params_billions=7.0,
        vram_gb_bf16=14.0,
        role="secondary",
        trust_remote_code=True,
    ),
    "pythia-2.8b": ModelConfig(
        name="pythia-2.8b",
        hf_model_id="EleutherAI/pythia-2.8b",
        model_type="pythia",
        r_ssm=0.0,
        params_billions=2.8,
        vram_gb_bf16=6.0,
        role="transformer_baseline",
    ),
    "mamba-2.8b": ModelConfig(
        name="mamba-2.8b",
        hf_model_id="state-spaces/mamba-2.8b",
        model_type="mamba",
        r_ssm=1.0,
        params_billions=2.8,
        vram_gb_bf16=6.0,
        role="ssm_baseline",
    ),
    # Lightweight models for CPU testing / pipeline validation
    "gpt2": ModelConfig(
        name="gpt2",
        hf_model_id="openai-community/gpt2",
        model_type="pythia",  # pure transformer architecture
        r_ssm=0.0,
        params_billions=0.124,
        vram_gb_bf16=0.25,
        role="cpu_test_transformer",
        torch_dtype="float32",
    ),
    "pythia-160m": ModelConfig(
        name="pythia-160m",
        hf_model_id="EleutherAI/pythia-160m",
        model_type="pythia",
        r_ssm=0.0,
        params_billions=0.160,
        vram_gb_bf16=0.32,
        role="cpu_test_transformer",
        torch_dtype="float32",
    ),
    # Gemma models — instruction-tuned, so they have real safety training
    # to test against (unlike GPT-2 which has no safety alignment)
    "gemma-3-1b-it": ModelConfig(
        name="gemma-3-1b-it",
        hf_model_id="google/gemma-3-1b-it",
        model_type="pythia",  # pure transformer
        r_ssm=0.0,
        params_billions=1.0,
        vram_gb_bf16=2.0,
        role="instruction_tuned_baseline",
        torch_dtype="float32",
    ),
}
