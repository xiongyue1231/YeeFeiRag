import re
from typing import List, Dict
import uuid
from embedding import VecEmbedding


class Chuck:
    def __init__(self):
        self.cleanSentence=[]

    # ============ 2. 文本清洗 加结构化数据===========
    def clean_sentences(self, sentences: List[str], image_source: str) -> List[Dict]:
        """
        短句直接入库，无需额外 chunk
        """
        for idx, (text, confidence) in enumerate(sentences):
            cleaned_text = re.sub(r'\s+', '', text)
            cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]', '', cleaned_text)
            if len(cleaned_text) > 2 and confidence > 0.5:  # ← 置信度过滤
                embedding = VecEmbedding()
                data = {
                    "id": f"{image_source}_sent_{idx}",
                    "text": text,  # 原始文本
                    "embedding": embedding.get_embedding(text),  # 你的 embedding 方法
                    "metadata": {
                        "source_image": image_source,
                        "sentence_index": idx,
                        "total_sentences": len(sentences),
                        "chunk_type": "sentence"  # 标记为句子级
                    }
                }
            self.cleanSentence.append(data)

        return self.cleanSentence
