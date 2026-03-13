
import re
from typing import List, Dict
import uuid

class Chuck:
    def __init__(self, name):
        self.name = name

    # ============ 2. 文本清洗 ============
    def clean_text(text_list):
        """清洗 OCR 结果"""
        cleaned = []
        for text, confidence in text_list:
            # 去除空格、特殊字符
            text = re.sub(r'\s+', '', text)
            text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
            if len(text) > 2 and confidence > 0.5:  # 过滤短文本和低置信度
                cleaned.append((text, confidence))
        return cleaned

    def clean_sentences(sentences: List[str], image_source: str) -> List[Dict]:
        """
        短句直接入库，无需额外 chunk
        """
        chunks = []
        for idx, text in enumerate(sentences):
            chunk = {
                "id": f"{image_source}_sent_{idx}",
                "text": text,  # 原始文本
                "embedding": '', # get_embedding(text),  # 你的 embedding 方法
                "metadata": {
                    "source_image": image_source,
                    "sentence_index": idx,
                    "total_sentences": len(sentences),
                    "chunk_type": "sentence"  # 标记为句子级
                }
            }
            chunks.append(chunk)

        return chunks