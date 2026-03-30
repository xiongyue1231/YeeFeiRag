import re
from typing import List, Dict
import uuid
from embedding import VecEmbedding
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


class OCRChuck:
    def __init__(self):
        self.cleanSentence = []
        self.chunk_size = config["rag"]["chunk_size"]
        self.chunk_overlap = config["rag"]["chunk_overlap"]

    # ============ 2. 文本清洗 加结构化数据===========
    def clean_sentences(self, sentences: List[str], image_source: str) -> List[Dict]:
        """
        短句直接入库，无需额外 chunk
        """
        for idx, (text, confidence) in enumerate(sentences):
            cleaned_text = re.sub(r'\s+', '', text)
            cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]', '', cleaned_text)
            # 不满足条件的文本跳过
            if len(cleaned_text) < 2 and confidence < 0.5:  # ← 置信度过滤
                continue

            embedding = VecEmbedding()
            vec = embedding.get_embedding(text).tolist()
            print(type(vec))  # 应该是 list
            print(type(vec[0]))  # 应该是 float
            print(len(vec))  # 应该是 512
            if len(text) < self.chunk_size:
                # 短句：直接入库（单句=单chunk）
                data = self._add_chunk(idx, image_source, sentences, text, vec)
            else:
                # 长句：分块入库
                chunks = self.chunk_text(text, self.chunk_size, self.chunk_overlap)

            self.cleanSentence.append(data)

        return self.cleanSentence

    def _add_chunk(self, idx, image_source, sentences, text, vec):
        data = {
            "id": f"{image_source}_sent_{idx}",
            "text": text,  # 原始文本
            "vector": vec,
            "metadata": {
                "source_image": str(image_source),
                "sentence_index": int(idx),
                "total_sentences": int(len(sentences)),
                "chunk_type": "sentence"  # 标记为句子级
            }
        }
        return data
