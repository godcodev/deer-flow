import os
from pathlib import Path
from typing import Any, Dict, get_args

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.config import load_yaml_config
from src.config.agents import LLMType
from src.llms.providers.dashscope import ChatDashscope


# -------------------------------------------------------
# Global Cache
# -------------------------------------------------------
_llm_cache: dict[LLMType, BaseChatModel] = {}


# -------------------------------------------------------
# Helpers for config paths and key mapping
# -------------------------------------------------------
def _get_config_file_path() -> str:
    """Return absolute path to conf.yaml."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())


def _llm_type_to_config_key() -> dict[str, str]:
    """Map LLMType to their configuration sections in YAML."""
    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
        "code": "CODE_MODEL",
    }


# -------------------------------------------------------
# Environment variable helpers
# -------------------------------------------------------
def _read_env_llm_config(llm_type: str) -> Dict[str, Any]:
    """
    Collect configuration from environment variables.
    Expected format:
        {TYPE}_MODEL__key=value
    Example:
        BASIC_MODEL__api_key=xyz
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    env_conf = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            normalized_key = key[len(prefix):].lower()
            env_conf[normalized_key] = value

    return env_conf


# -------------------------------------------------------
# LLM model factory
# -------------------------------------------------------
def _build_llm_instance(llm_type: LLMType, config: Dict[str, Any]) -> BaseChatModel:
    """Create an LLM instance using merged config from YAML + environment."""
    type_to_key = _llm_type_to_config_key()
    config_key = type_to_key.get(llm_type)

    if not config_key:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    yaml_conf = config.get(config_key, {})
    if not isinstance(yaml_conf, dict):
        raise ValueError(f"Invalid configuration for LLM type '{llm_type}': {yaml_conf}")

    env_conf = _read_env_llm_config(llm_type)
    merged = {**yaml_conf, **env_conf}

    if not merged:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Default retries
    merged.setdefault("max_retries", 3)

    # SSL verification
    if merged.pop("verify_ssl", True) is False:
        merged["http_client"] = httpx.Client(verify=False)
        merged["http_async_client"] = httpx.AsyncClient(verify=False)

    # Azure model
    if "azure_endpoint" in merged or os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureChatOpenAI(**merged)

    # Dashscope model
    base_url = merged.get("base_url", "")
    if "dashscope." in base_url:
        merged["extra_body"] = {"enable_thinking": llm_type == "reasoning"}
        return ChatDashscope(**merged)

    # DeepSeek reasoning mode
    if llm_type == "reasoning":
        merged["api_base"] = merged.pop("base_url", None)
        return ChatDeepSeek(**merged)

    # Default OpenAI-compatible model
    return ChatOpenAI(**merged)


# -------------------------------------------------------
# Public API: LLM retrieval and configuration inspection
# -------------------------------------------------------
def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    """Return a cached LLM instance for the given type, creating it if needed."""
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    config = load_yaml_config(_get_config_file_path())
    llm = _build_llm_instance(llm_type, config)
    _llm_cache[llm_type] = llm

    return llm


def get_configured_llm_models() -> dict[str, list[str]]:
    """
    List all configured LLM model names grouped by type.
    Combines YAML and environment variables.
    """
    try:
        config = load_yaml_config(_get_config_file_path())
        type_to_key = _llm_type_to_config_key()
        result: dict[str, list[str]] = {}

        for llm_type in get_args(LLMType):
            yaml_conf = config.get(type_to_key.get(llm_type, ""), {})
            env_conf = _read_env_llm_config(llm_type)

            merged = {**yaml_conf, **env_conf}
            model_name = merged.get("model")

            if model_name:
                result.setdefault(llm_type, []).append(model_name)

        return result

    except Exception as e:
        print(f"Warning: Failed to load LLM configuration: {e}")
        return {}


# Future usage example:
# reasoning_llm = get_llm_by_type("reasoning")
# vision_llm = get_llm_by_type("vision")