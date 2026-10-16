from langchain_community.embeddings import HuggingFaceEmbeddings
from src.app_config.loder import ConfigLoader
import os
import threading
from typing import Optional

config_manager = ConfigLoader()

# BGE 中文 query 前缀（与 BAAI/bge-*-zh 官方建议一致）。
# 注意：只有 query 加前缀，文档入库时不加，两者必须与入库时的向量保持同一套规则，
# 否则已入库的 Milvus 向量全部失效，需要重建。
BGE_QUERY_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


def _resolve_local_path(cfg_path: str, current_file: str) -> str:
    """把配置文件里的 local_url 解析为绝对路径：
    - 绝对路径直接返回
    - 相对路径相对 pythonProject 根目录解析
    """
    if os.path.isabs(cfg_path):
        return os.path.normpath(cfg_path)
    # current_file 一般是 .../pythonProject/src/embed/embedding.py
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(current_file))))
    return os.path.normpath(os.path.join(project_root, cfg_path))


class VecEmbedding:
    def __init__(self, model_path: str = None, device: str = None):
        if model_path is None:
            cfg_path = config_manager.config.models.embedding_model["bge-small-zh-v1.5"].local_url
            model_path = _resolve_local_path(cfg_path, __file__)
        if device is None:
            device = config_manager.config.deviceSettings.device

        # 关键：路径不存在直接报错，不要让它去 huggingface 静默下载
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[VecEmbedding] 本地 embedding 模型路径不存在: {model_path}\n"
                f"请把 BAAI/bge-small-zh-v1.5 下载到该目录，或修改 src/app_config/models.py 的 local_url"
            )
        print(f"[VecEmbedding] loading from: {model_path}, device={device}", flush=True)

        # 用基类 HuggingFaceEmbeddings：它不接受 query_instruction 参数，
        # BGE 的 query 前缀由本类自己拼接（见 _with_query_instruction）
        self.model = HuggingFaceEmbeddings(
            model_name=model_path,  # 本地绝对路径 → 从磁盘加载
            model_kwargs={"device": device},  # "cuda" / "cpu"
            encode_kwargs={"normalize_embeddings": True},  # BGE 推荐归一化
        )

    # ---------- BGE 中文 query 前缀：只加在 query 上，文档不加 ----------
    def _with_query_instruction(self, text: str) -> str:
        return BGE_QUERY_INSTRUCTION_ZH + text

    # ---------- 兼容旧接口：保留 get_embedding 名称 ----------
    def get_embedding(self, text):
        """
        兼容旧调用方式。
        - text 是 str  → 走 embed_query（会加 query 前缀），返回 List[float]
        - text 是 list → 走 embed_documents（不加前缀），返回 List[List[float]]
        """
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)

    # ---------- 显式区分 query / document ----------
    def embed_query(self, text: str):
        """检索时用：加 BGE 中文 query 前缀"""
        return self.model.embed_query(self._with_query_instruction(text))

    def embed_documents(self, texts):
        """入库时用：不加前缀，支持批量"""
        return self.model.embed_documents(texts)


# ---------- 全局单例：embedding 模型进程内只加载一次 ----------
_shared_embedding: Optional[VecEmbedding] = None
_embedding_lock = threading.Lock()


def get_vec_embedding() -> VecEmbedding:
    """
    获取全局共享的 VecEmbedding 实例（懒加载）。
    首次调用时从磁盘加载模型，之后所有调用复用同一实例。
    线程安全：FastAPI 同步接口跑在线程池中，用锁防止并发重复加载。
    """
    global _shared_embedding
    if _shared_embedding is None:
        with _embedding_lock:
            if _shared_embedding is None:
                _shared_embedding = VecEmbedding()
    return _shared_embedding
