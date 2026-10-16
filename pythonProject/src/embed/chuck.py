import re
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 注意：本模块不要直接用 HuggingFaceEmbeddings，向量统一走 get_vec_embedding()，
# 由 VecEmbedding 统一维护 BGE 的 query/文档前缀规则，避免入库与检索规则不一致
from src.embed.embedding import get_vec_embedding
from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()


class OCRChuck:
    def __init__(self):
        self.cleanSentence: List[Dict] = []
        self.chunk_size = config_manager.config.rag.chunk_size
        self.chunk_overlap = config_manager.config.rag.chunk_overlap
        self.embedding = get_vec_embedding()

        # 防御：overlap 必须小于 chunk_size，否则切分会异常
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})"
            )

        # 中文优先的分隔符，按优先级从大到小
        self.splitter = RecursiveCharacterTextSplitter(
            # 文本切分长度
            chunk_size=self.chunk_size,
            # 重叠字符   用来避免 一句话被切两半 导致语义丢失
            chunk_overlap=self.chunk_overlap,
            # 什么方式计算的文本长度
            length_function=len,
            # 保留标点
            keep_separator=True,
            # 元数据
            add_start_index=True,
            # 分隔符优先级
            separators=[
                "\n\n",  # 段落
                "\n",  # 换行
                "。", "！", "？", "；",  # 中文句末
                ".", "!", "?", ";",  # 英文句末
                "，", "、", ",",  # 句中标点
                " ",  # 空格
                "",  # 兜底逐字符
            ],
        )

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #
    def clean_sentences(
            self,
            sentences: List[str],
            source: str,
            source_type: str,
            source_hash: str,
            knowledge_id: int,
            document_id: int,
    ) -> List[Dict]:
        """
        sentences: [(text, confidence), ...]
        返回: 切分 + 向量化后的 chunk 列表
        """
        # 1. 归一化 + 过滤，组装 Document
        docs: List[Document] = []
        for idx, (text, confidence) in enumerate(sentences):
            normalized = self._normalize(text)
            if not normalized:
                continue
            # 太短且置信度低 → 丢弃
            if len(normalized) < 2 and confidence < 0.5:
                continue

            docs.append(
                Document(
                    page_content=normalized,
                    metadata={
                        "sentence_index": idx,
                        "source": str(source),
                        "source_hash": source_hash,
                        "knowledge_id": knowledge_id,
                        "document_id": document_id,
                        "source_type": source_type,
                        "total_sentences": len(sentences),
                    },
                )
            )

        # 2. 统一切分（短句也会保留为单 chunk）
        chunks = self.splitter.split_documents(docs)

        # 3. 向量化 + 组装结果
        if chunks:
            texts = [doc.page_content for doc in chunks]
            # 入库用 embed_documents，不加 query 前缀，且一次 batch 更快
            vectors = self.embedding.embed_documents(texts)  # List[List[float]]

            for i, (doc, vec) in enumerate(zip(chunks, vectors)):
                self._add_chunk(i, doc, vec)

        return self.cleanSentence

    # ------------------------------------------------------------------ #
    #  内部方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(text: str) -> str:
        """只归一化空白，保留标点（切分依赖标点）"""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _add_chunk(self, chunk_idx: int, doc: Document, vec: List[float]) -> None:
        meta = doc.metadata
        document_id = meta["document_id"]
        source_type = meta.get("source_type", "image")

        data = {
            # id 用 document_id + source_type + 句子索引 + chunk 序号，保证唯一
            "id": f"{document_id}_{source_type}_sent_{meta['sentence_index']}_chunk_{chunk_idx}",
            "text": doc.page_content,  # ← 修复：存的是 chunk 本身
            "vector": vec,
            "metadata": {
                "knowledge_id": meta["knowledge_id"],
                "document_id": document_id,
                "source_hash": meta["source_hash"],
                "source": meta["source"],
                "sentence_index": int(meta["sentence_index"]),
                "source_type": source_type,
                "total_sentences": int(meta["total_sentences"]),
                # split_documents 会带 start_index
                "start_index": int(meta.get("start_index", 0)),
                "chunk_index": chunk_idx,
                "chunk_type": "sentence",
            },
        }
        self.cleanSentence.append(data)
