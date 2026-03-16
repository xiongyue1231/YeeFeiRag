from sentence_transformers import SentenceTransformer


class VecEmbedding:
    def __init__(self, model_path=None, device="cpu"):
        if model_path is None:
            model_path = '../models/BAAI/bge-small-zh-v1.5'

        self.model = SentenceTransformer(model_path, device=device)
        print(self.model.get_sentence_embedding_dimension())

    def get_embedding(self, text):
        result = self.model.encode(text, normalize_embeddings=True)
        return result
