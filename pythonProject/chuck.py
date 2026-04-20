import re
from typing import List, Dict
import uuid
from embedding import VecEmbedding
import yaml
from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()


class OCRChuck:
    def __init__(self):
        self.cleanSentence = []
        self.chunk_size = config_manager.config.rag.chunk_size
        self.chunk_overlap = config_manager.config.rag.chunk_overlap

        # ============ 2. 文本清洗 加结构化数据===========

    def clean_sentences(self, sentences: List[str], source: str, source_type: str, semantic_chunking: bool = True) -> \
            List[Dict]:
        for idx, (text, confidence) in enumerate(sentences):
            cleaned_text = re.sub(r'\s+', '', text)
            cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]', '', cleaned_text)
            # 不满足条件的文本跳过
            if len(cleaned_text) < 2 and confidence < 0.5:  # ← 置信度过滤
                continue

            embedding = VecEmbedding()
            if len(text) < self.chunk_size:
                vec = embedding.get_embedding(text).tolist()
                # 短句：直接入库（单句=单chunk）
                self._add_chunk(idx, source, sentences, text, vec, source_type)
            else:
                # 长句：分块入库
                chucks = self._split_long_text(text, self.chunk_size, self.chunk_overlap, semantic_chunking)
                for chunk in chucks:
                    vec = embedding.get_embedding(chunk).tolist()
                    self._add_chunk(idx, source, sentences, text, vec, source_type)

        return self.cleanSentence

    def _add_chunk(self, idx, source, sentences, text, vec, source_type="image"):
        data = {
            "id": f"{source}_sent_{idx}",
            "text": text,  # 原始文本
            "vector": vec,
            "metadata": {
                "source": str(source),
                "sentence_index": int(idx),
                "source_type": source_type,
                "total_sentences": int(len(sentences)),
                "chunk_type": "sentence"  # 标记为句子级
            }
        }

        self.cleanSentence.append(data)

    def _split_long_text(
            self,
            text: str,
            max_size: int,
            overlap: int,
            semantic_first: bool
    ) -> List[str]:
        """
        长文本分层切分：
        1. 先尝试按语义切分（标点）
        2. 对超长语义段使用滑动窗口
        """
        chunks = []

        if semantic_first:
            # 策略1：按语义标点粗分（保留标点位置）
            # 使用正向预查，保留分隔符
            semantic_splits = re.split(r'([。！？；.,?!; \s\n])', text)
            # 重组：将分隔符拼回前一段
            segments = []
            current = ""
            for i, part in enumerate(semantic_splits):
                if i % 2 == 0:  # 文本段
                    current = part
                else:  # 分隔符
                    current += part
                    if len(current) > 10:  # 过滤过短片段
                        segments.append(current)
                    current = ""

            if current:
                segments.append(current)

            # 检查每个语义段长度
            for seg in segments:
                if len(seg) <= max_size:
                    chunks.append(seg.strip())
                else:
                    # 语义段仍太长，使用滑动窗口细切
                    chunks.extend(self._sliding_window(seg, max_size, overlap))
        else:
            # 策略2：直接使用滑动窗口（无语义保持）
            chunks = self._sliding_window(text, max_size, overlap)

        return [c for c in chunks if c.strip()]  # 过滤空串

    def _sliding_window(self, text: str, window_size: int, overlap: int) -> List[str]:
        """滑动窗口切分，确保上下文连续性"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + window_size, text_len)
            chunk = text[start:end]

            # 智能边界：避免在词中间切断（中文按字，英文按词）
            if end < text_len:
                # 尝试向后找最近的语义断点（标点或空格）
                look_ahead = min(20, text_len - end)  # 最多看20字符
                sub_text = text[end:end + look_ahead]

                # 找第一个标点或空格
                match = re.search(r'[。，！？；.,?!; \s]', sub_text)
                if match:
                    end += match.start() + 1

            chunks.append(text[start:end].strip())

            # 步进：窗口大小 - 重叠
            start += (window_size - overlap)

        return chunks

# if __name__ == "__main__":
#     test = """这个就是前面房型融合想更细会出现的问题。目前程序的逻辑是 房型名称及床型，在另外一个房型名称及床型（名称包含、房型一样）就会认为是相同房型，比如：泰迪珍藏君悦豪华大床房 ，君悦豪华大床房 房型名称，按照逻辑
#     君悦豪华大床房 会被认为是  泰迪珍藏君悦豪华大床房 的相同房型"""
#     OCRChuck()._split_long_text(test, 20, 4, True)
