from typing import List, Dict, Any
import os
from src.app_config.models import RagConfig
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from src.app_config.loder import ConfigLoader
from langchain_openai import ChatOpenAI

config_manager = ConfigLoader()


def create_llm_client(config: RagConfig):
    """创建LLM客户端"""
    if config.provider == "openai":
        kwargs = {

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


def create_llm_langchain(config: RagConfig) -> ChatOpenAI:
    kwargs = {}
    if config.provider == "openai":
        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key
        if config.llm_base:
            kwargs["base_url"] = config.llm_base
        if config.llm_model:
            kwargs["model"] = config.llm_model
    elif config.provider == "ollama":
        pass

    elif config.provider == "vllm":
        if config.vllm_api_key:
            kwargs["api_key"] = config.vllm_api_key
        if config.vllm_base:
            kwargs["base_url"] = config.vllm_base
        if config.vllm_model:
            kwargs["model"] = config.vllm_model
    else:
        raise ValueError(f"不支持的 LLM 提供商: {config.provider}")

    return ChatOpenAI(
        model=kwargs["model"],
        temperature=0.1,
        top_p=0.9,
        api_key=kwargs["api_key"],
        base_url=kwargs["base_url"],
    )


EMBEDDING_MODEL_PARAMS = {}


def load_rerank_model(model_name: str, model_path: str) -> EMBEDDING_MODEL_PARAMS:
    global EMBEDDING_MODEL_PARAMS
    """
    加载重排序模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    current_dir = Path(__file__).parent.parent.resolve()
    model_path = current_dir / "models" / "BAAI" / "bge-reranker-base"
    model_path = str(model_path)
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(config_manager.config.deviceSettings.device)

    return EMBEDDING_MODEL_PARAMS


