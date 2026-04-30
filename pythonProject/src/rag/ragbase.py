from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from src.embed.embedding import VecEmbedding
from src.database.milvus import MilvusManager
from src.app_config.loder import ConfigLoader
from src.core.utils import create_llm_client, load_rerank_model
from typing import List
import numpy as np
import torch

config_manager = ConfigLoader()
device = config_manager.config.deviceSettings.device

EMBEDDING_MODEL_PARAMS = {}

if config_manager.config.rag.use_rerank:
    model_name = config_manager.config.rag.rerank_model
    model_info = config_manager.config.models.rerank_model[model_name]
    model_path = model_info.local_url
    print(f"加载重排序模型： {model_name} ")
    EMBEDDING_MODEL_PARAMS = load_rerank_model(model_name, model_path)


# ---------- 自定义 LangChain Retriever ----------
class HybridRetriever(BaseRetriever):
    """
    整合 Milvus 向量检索 + BM25 + RRF 融合 + 重排序的检索器
    """
    vec_embedding: VecEmbedding
    milvus: MilvusManager
    use_rrf: bool = True
    use_rerank: bool = True
    chunk_candidate: int = 5
    knowledge_id: int = 1  # 默认知识库 ID，可外部传入

    # class Config:
    #     arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 获取知识库对应的 collection 名称
        from src.database.db_api import Session, KnowledgeDatabase
        knowledge_record = Session().query(KnowledgeDatabase).filter(
            KnowledgeDatabase.knowledge_id == self.knowledge_id
        ).first()
        if not knowledge_record:
            return []

        collection_name = knowledge_record.category

        # 1. BM25 检索
        word_search_response = self.milvus.search_bm25(
            query, collection_name=collection_name, top_k=5
        )
        # 2. 向量检索
        embedding_vector = self.vec_embedding.get_embedding(query)
        vector_search_response = self.milvus.search_dense(
            embedding_vector, collection_name=collection_name, top_k=5
        )

        sorted_records = []
        sorted_contents = []

        if self.use_rrf:
            k = 60
            fusion_score = {}
            search_id2record = {}
            for idx, record in enumerate(word_search_response):
                _id = record["id"]
                fusion_score[_id] = fusion_score.get(_id, 0) + 1 / (idx + k)
                if _id not in search_id2record:
                    search_id2record[_id] = record["entity"]

            for idx, record in enumerate(vector_search_response):
                _id = record["id"]
                fusion_score[_id] = fusion_score.get(_id, 0) + 1 / (idx + k)
                if _id not in search_id2record:
                    search_id2record[_id] = record["entity"]

            sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
            sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
            sorted_contents = [x["text"] for x in sorted_records]

        if self.use_rerank and sorted_contents:
            def _get_rank(text_pairs: List[List[str]]) -> np.ndarray:
                rerank_model_name = config_manager.config.rag.rerank_model
                if rerank_model_name in ["bge-reranker-base"]:
                    with torch.no_grad():
                        inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                            text_pairs,
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=512,
                        )
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        scores = EMBEDDING_MODEL_PARAMS["rerank_model"](
                            **inputs, return_dict=True
                        ).logits.view(-1, ).float()
                        return scores.data.cpu().numpy()
                else:
                    raise NotImplementedError("暂不支持其他重排序模型")

            # 重排序
            text_pairs = [[query, content] for content in sorted_contents]
            rerank_scores = _get_rank(text_pairs)
            rerank_idx = np.argsort(rerank_scores)[::-1]
            sorted_records = [sorted_records[x] for x in rerank_idx]
            sorted_contents = [sorted_contents[x] for x in rerank_idx]

        # 转换为 LangChain Document 格式
        docs = []
        for record in sorted_records:
            doc = Document(
                page_content=record.get("text", ""),
                metadata={
                    "source": record.get("source", ""),
                    "id": record.get("id", ""),
                }
            )
            docs.append(doc)
        return docs


