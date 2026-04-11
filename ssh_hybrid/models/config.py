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
    # Qwen2.5-0.5B-Instruct: modern, instruction-tuned with safety alignment,
    # small enough for CPU. This is the recommended CPU test model because it
    # has real safety training (unlike GPT-2 which has none).
    "qwen2.5-0.5b-instruct": ModelConfig(
        name="qwen2.5-0.5b-instruct",
        hf_model_id="/tmp/qwen_model",
        model_type="pythia",  # pure transformer architecture
        r_ssm=0.0,
        params_billions=0.5,
        vram_gb_bf16=1.0,
        role="cpu_test_instruction_tuned",
        torch_dtype="float32",
    ),
    "qwen2.5-1.5b-instruct": ModelConfig(
        name="qwen2.5-1.5b-instruct",
        hf_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_type="pythia",  # pure transformer
        r_ssm=0.0,
        params_billions=1.5,
        vram_gb_bf16=3.0,
        role="instruction_tuned_baseline",
        torch_dtype="float32",
    ),
    "gpt2": ModelConfig(
        name="gpt2",
        hf_model_id="openai-community/gpt2",
        model_type="pythia",
        r_ssm=0.0,
        params_billions=0.124,
        vram_gb_bf16=0.25,
        role="cpu_test_no_safety",
        torch_dtype="float32",
    ),
}
