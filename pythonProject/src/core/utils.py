from typing import List,Dict,Any
import os
from ..app_config.models import RagConfig
from openai import OpenAI


def create_llm_client(config: RagConfig):
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {
            "model": config.model
        }
        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key
        if config.llm_base:
            kwargs["base_url"] = config.llm_base
        return OpenAI(**kwargs)
    elif config.provider == "ollama":
        pass

    elif config.provider == "vllm":
        kwargs = {
            "model": config.model
        }
        if config.vllm_api_key:
            kwargs["api_key"] = config.vllm_api_key
        if config.vllm_base:
            kwargs["base_url"] = config.vllm_base
        return OpenAI(**kwargs)
    else:
        raise ValueError(f"不支持的 LLM 提供商: {config.provider}")

