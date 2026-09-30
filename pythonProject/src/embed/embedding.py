from sentence_transformers import SentenceTransformer
from src.app_config.loder import ConfigLoader
import os
config_manager = ConfigLoader()


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
                f"请把 BAAI/bge-small-zh-v1.5 下载到该目录，或修改 config.yaml 的 local_url"
            )
        print(f"[VecEmbedding] loading from: {model_path}, device={device}", flush=True)
        self.model = SentenceTransformer(model_path, device=device)

    def get_embedding(self, text):
        result = self.model.encode(text, normalize_embeddings=True)
        return result
