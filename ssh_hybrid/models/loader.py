"""Production model loading for SSH-Hybrid experiments.

Loads real models from HuggingFace Hub. Supports Jamba, Zamba, Pythia, and Mamba.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssh_hybrid.models.config import MODEL_REGISTRY, ModelConfig

logger = logging.getLogger(__name__)

TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g., 'jamba-1.5-mini').

    Returns:
        ModelConfig for the requested model.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def load_model(
    model_name: str,
    device: str = "auto",
    dtype: str | None = None,
    load_tokenizer: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a model and tokenizer from HuggingFace Hub.

    Args:
        model_name: Key in MODEL_REGISTRY.
        device: Device map for model loading ('auto', 'cpu', 'cuda:0', etc.).
        dtype: Override torch dtype (default: from config).
        load_tokenizer: Whether to also load the tokenizer.
        **kwargs: Additional kwargs passed to AutoModelForCausalLM.from_pretrained.

    Returns:
        Dict with 'model', 'tokenizer' (if loaded), and 'config'.
    """
    config = get_model_config(model_name)
    torch_dtype = TORCH_DTYPE_MAP[dtype or config.torch_dtype]

    logger.info(
        "Loading %s (%s) with dtype=%s, device=%s",
        config.name, config.hf_model_id, torch_dtype, device,
    )

    if config.model_type == "mamba":
        model = _load_mamba_model(config, torch_dtype, device, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=config.trust_remote_code,
            **kwargs,
        )

    result: dict[str, Any] = {"model": model, "config": config}

    if load_tokenizer:
        result["tokenizer"] = _load_tokenizer(config)

    logger.info("Model %s loaded successfully", config.name)
    return result


def _load_mamba_model(
    config: ModelConfig,
    torch_dtype: torch.dtype,
    device: str,
    **kwargs: Any,
) -> torch.nn.Module:
    """Load a Mamba model using the mamba-ssm package.

    Falls back to transformers if mamba-ssm is not installed.
    """
    try:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(
            config.hf_model_id,
            dtype=torch_dtype,
            device=device if device != "auto" else "cuda",
            **kwargs,
        )
        return model
    except ImportError:
        logger.warning(
            "mamba-ssm not installed, falling back to transformers for %s",
            config.name,
        )
        return AutoModelForCausalLM.from_pretrained(
            config.hf_model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            **kwargs,
        )


def _load_tokenizer(config: ModelConfig) -> Any:
    """Load tokenizer for the given model config."""
    if config.model_type == "mamba":
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return AutoTokenizer.from_pretrained(
        config.hf_model_id,
        trust_remote_code=config.trust_remote_code,
    )
