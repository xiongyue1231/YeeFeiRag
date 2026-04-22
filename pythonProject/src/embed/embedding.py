from sentence_transformers import SentenceTransformer
from src.app_config.loder import ConfigLoader
import os
config_manager = ConfigLoader()


class VecEmbedding:
    def __init__(self, model_path: str = None, device: str = None):
        if model_path is None:
            model_path = config_manager.config.models.embedding_model["bge-small-zh-v1.5"].local_url
        if device is None:
            device = config_manager.config.deviceSettings.device

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 src/embed/ 回到项目根目录，再进入 models
        model_path = os.path.join(current_dir,  '..', 'models', 'BAAI', 'bge-small-zh-v1.5')
        model_path = os.path.normpath(model_path)
        self.model = SentenceTransformer(model_path, device=device)
        # print(self.model.get_sentence_embedding_dimension())

    def get_embedding(self, text):
        result = self.model.encode(text, normalize_embeddings=True)
        return result
