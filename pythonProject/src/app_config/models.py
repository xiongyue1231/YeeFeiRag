from typing import Dict, List, Optional, Literal, Any, Union
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Milvus 配置
# =============================================================================
class MilvusConfig(BaseModel):
    """Milvus配置"""
    dims: int = 768
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "current_collection_test"


class DatabaseConfig(BaseModel):
    engine: str = "mysql+pymysql"
    path: str = "rag.db"
    host: str = "localhost"
    port: int = 3306
    username: str = "yifei"
    password: str = "111111"
    mydb: str = "yeefeirag"


class RagConfig(BaseModel):
    llm_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = "sk-cbf8dce472b74ddfbf3210f7d52dd758"
    llm_model: str = "qwen-plus"

    vllm_base: str = "http://localhost:8000/v1"
    vllm_api_key: str = "43aa6570992d01fab078e157e07b4f52.ocKHwNGxhIBrDZD4"
    vllm_model: str = "models/Qwen/Qwen3.5-0.8B/"
    temperature: float = 0.1
    # model: str = "qwen-plus"

    embedding_model: str = "bge-small-zh-v1.5"
    rerank_model: str = "bge-reranker-base"
    chunk_size: int = 256
    chunk_overlap: int = 20
    chunk_candidate: int = 10
    use_embedding: bool = True
    use_rerank: bool = True
    use_rrf: bool = True
    port: int = 6010
    provider: str = "openai"


class DeviceConfig(BaseModel):
    """设备配置"""
    device: str = "cpu"


# ========== 第一层：单个模型配置 ==========

class EmbeddingModelInfo(BaseModel):
    """单个嵌入模型配置"""
    hf_url: str
    local_url: str
    dims: int


class RerankModelInfo(BaseModel):
    """单个重排序模型配置"""
    hf_url: str
    local_url: str


# ========== 第二层：模型仓库（动态键） ==========

class ModelsConfig(BaseModel):
    """所有模型配置仓库"""
    embedding_model: Dict[str, EmbeddingModelInfo]
    rerank_model: Dict[str, RerankModelInfo]


class RedisConfig(BaseModel):
    """Redis 配置"""
    host: str = "localhost"
    port: int = 6379
    SESSION_TTL: int = 604800


class AppConfig(BaseModel):
    """应用主配置 - 更新版"""
    milvus: MilvusConfig
    database: DatabaseConfig
    rag: RagConfig
    deviceSettings: DeviceConfig
    models: ModelsConfig
    redis: RedisConfig
