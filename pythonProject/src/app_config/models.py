"""
应用配置的唯一来源。

所有配置项都以 Pydantic 模型默认值的形式定义在本文件（src/app_config/models.py）中，
项目内统一通过 get_config() / ConfigLoader 读取，不再解析 config.yaml。
"""
import threading
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
    vllm_api_key: str = ""
    vllm_model: str = "qwen3.5-0.8b"
    temperature: float = 0.1
    # model: str = "qwen-plus"

    embedding_model: str = "bge-small-zh-v1.5"
    rerank_model: str = "bge-reranker-base"
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_candidate: int = 10
    use_embedding: bool = True
    use_rerank: bool = True
    use_rrf: bool = True
    # 对话前是否先用 LLM 改写问题再检索。关闭可省掉一次 LLM 调用（响应约快一倍），
    # 网关/前端有 60s 超时时可设为 False
    use_query_rewrite: bool = True
    port: int = 6010
    provider: str = "vllm"  # openai, ollama, vllm 本地/接口商
    router_model: str = ""  # 意图路由专用模型（留空则复用主模型）


class DeviceConfig(BaseModel):
    """设备配置"""
    device: str = "cuda"


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


class IntentModelInfo(BaseModel):
    """单个意图分类模型配置"""
    base_url: str
    model_name: str


# ========== 第二层：模型仓库（动态键） ==========

class ModelsConfig(BaseModel):
    """所有模型配置仓库"""
    embedding_model: Dict[str, EmbeddingModelInfo] = Field(
        default_factory=lambda: {
            "bge-small-zh-v1.5": EmbeddingModelInfo(
                hf_url="https://huggingface.co/BAAI/bge-small-zh-v1.5",
                local_url="src/models/BAAI/bge-small-zh-v1.5",
                dims=512,
            )
        }
    )
    rerank_model: Dict[str, RerankModelInfo] = Field(
        default_factory=lambda: {
            "bge-reranker-base": RerankModelInfo(
                hf_url="https://huggingface.co/BAAI/bge-reranker-base",
                local_url="src/models/BAAI/bge-reranker-base",
            )
        }
    )
    intent_model: Dict[str, IntentModelInfo] = Field(
        default_factory=lambda: {
            "qwen2.5-1.5b-lora-merged": IntentModelInfo(
                base_url="http://localhost:8000/v1",
                model_name="qwen-merged",
            )
        }
    )


class RedisConfig(BaseModel):
    """Redis 配置"""
    host: str = "localhost"
    port: int = 6379
    SESSION_TTL: int = 3600


class ChatMemoryConfig(BaseModel):
    """对话记忆配置"""
    estimate_token_fun: str = "avg"
    llm_max_token: int = 1024
    cut_dialogue_scale: int = 2
    max_token_threshold: float = 1.01
    smith_debug: bool = False
    console_debug: bool = True
    thinking_in_context: bool = False


class IntentClassifierConfig(BaseModel):
    """本地意图分类模型（LoRA 微调后的 Qwen）配置"""
    # 置为 False：完全不加载该模型，请求直接进入 RAG 分支（功能照常，只是不做闲聊/问答分流）。
    # 实测本机（Windows 页面文件不足）加载该 1.5B 模型会在 mmap 阶段卡死整个进程，
    # 扩到 ≥16GB 页面文件、或换内存更大的机器后，改回 True 即可启用意图路由。
    enabled: bool = False
    model_path: str = "src/lora/qwen_merged"  # 相对 pythonProject 根目录，也支持绝对路径
    # 注意：vLLM 默认会占满显卡（本机 3060 12G 已被占 11.9G），主进程再加载 1.5B 模型会 OOM。
    # 因此默认用 cpu；若要改回 cuda，请先调低 vLLM 的 --gpu-memory-utilization（如 0.7）腾出约 3G。
    device: str = "cpu"
    url: str = "http://localhost:8001/"  # 独立部署 Intent_classification.py 服务时的地址


class AppConfig(BaseModel):
    """应用主配置：全部字段均有默认值，可直接 AppConfig() 构造"""
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    deviceSettings: DeviceConfig = Field(default_factory=DeviceConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    chat_memory: ChatMemoryConfig = Field(default_factory=ChatMemoryConfig)
    intent_classification: IntentClassifierConfig = Field(default_factory=IntentClassifierConfig)


# =============================================================================
# 配置入口：进程内统一从这里取配置
# =============================================================================
_config: Optional[AppConfig] = None
_config_lock = threading.Lock()


def _deep_merge(base: Any, patch: Any) -> Any:
    """dict 深合并，其余类型直接替换"""
    if isinstance(base, dict) and isinstance(patch, dict):
        out = dict(base)
        for key, value in patch.items():
            out[key] = _deep_merge(out.get(key), value) if key in out else value
        return out
    return patch


def build_config(overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    """
    构造配置对象：默认值全部来自本文件的 Pydantic 模型。

    :param overrides: 可选的覆盖项（如测试/临时改参），深合并后重新校验
    """
    data: Dict[str, Any] = AppConfig().model_dump()
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig(**data)


def get_config(reload: bool = False) -> AppConfig:
    """获取全局配置单例（进程内只构造一次）"""
    global _config
    with _config_lock:
        if _config is None or reload:
            _config = build_config()
        return _config


def set_config(config: AppConfig) -> AppConfig:
    """替换全局配置单例（如运行时改参、测试注入）"""
    global _config
    with _config_lock:
        _config = config
        return _config
