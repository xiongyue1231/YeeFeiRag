#
# #模型下载
# from modelscope import snapshot_download
# model_dir = snapshot_download('PaddlePaddle/PaddleOCR-VL-1.5',cache_dir='./models/PaddlePaddle/PaddleOCR-VL-1.5')

#
# from paddleocr import PaddleOCR
#
# # 首次运行会自动下载模型（约 100MB）
# ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
#
# # 测试识别
# result = ocr.ocr('test.jpg', cls=True)
# for line in result[0]:
#     print(f"文字: {line[1][0]}, 置信度: {line[1][1]:.2f}")

# from modelscope import snapshot_download
# model_dir = snapshot_download('BAAI/bge-small-zh-v1.5',local_dir='./models/BAAI/bge-small-zh-v1.5')


#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-base',local_dir='./models/BAAI/bge-reranker-base')

from embedding import VecEmbedding
import re
from typing import List, Dict, Tuple, Union
import uuid

#
# class OCRChunker:
#     def __init__(self):
#         self.cleanSentence = []
#
#     def clean_sentences(
#             self,
#             sentences: List[Tuple[str, float]],  # (text, confidence) 元组列表
#             image_source: str,
#             max_chunk_size: int = 512,  # 每个chunk最大字符数
#             overlap_size: int = 50,  # 重叠区域大小（保持上下文）
#             semantic_chunking: bool = True  # 是否优先按语义（标点）切分
#     ) -> List[Dict]:
#         """
#         智能分块策略：
#         1. 短句（<= max_chunk_size）直接入库
#         2. 长句优先按语义标点切分（。！？；）
#         3. 超长语义段使用滑动窗口切分（带重叠）
#         """
#         self.cleanSentence = []
#         embedding = VecEmbedding()  # 假设这是你的嵌入模型
#
#         for sent_idx, (text, confidence) in enumerate(sentences):
#             # 基础清洗
#             cleaned_text = re.sub(r'\s+', '', text)
#
#             # 质量过滤（保留原逻辑）
#             if len(cleaned_text) <= 2 or confidence <= 0.5:
#                 continue
#
#             # ========== 分块策略分支 ==========
#             if len(text) <= max_chunk_size:
#                 # 短句：直接入库（单句=单chunk）
#                 self._add_chunk(
#                     text=text,
#                     vec=embedding.get_embedding(text).tolist(),
#                     image_source=image_source,
#                     sent_idx=sent_idx,
#                     chunk_idx=0,
#                     total_chunks=1,
#                     is_chunked=False,
#                     confidence=confidence
#                 )
#             else:
#                 # 长句：需要分块
#                 chunks = self._split_long_text(
#                     text,
#                     max_chunk_size,
#                     overlap_size,
#                     semantic_chunking
#                 )
#
#                 # 为每个子块生成embedding和独立ID
#                 for chunk_idx, chunk_text in enumerate(chunks):
#                     # 优化：如果chunk太短，跳过（避免噪音）
#                     if len(chunk_text.strip()) < 10:
#                         continue
#
#                     vec = embedding.get_embedding(chunk_text).tolist()
#
#                     self._add_chunk(
#                         text=chunk_text,
#                         vec=vec,
#                         image_source=image_source,
#                         sent_idx=sent_idx,
#                         chunk_idx=chunk_idx,
#                         total_chunks=len(chunks),
#                         is_chunked=True,
#                         parent_text=text,  # 保留原始长句
#                         confidence=confidence
#                     )
#
#         return self.cleanSentence
#
#     def _split_long_text(
#             self,
#             text: str,
#             max_size: int,
#             overlap: int,
#             semantic_first: bool
#     ) -> List[str]:
#         """
#         长文本分层切分：
#         1. 先尝试按语义切分（标点）
#         2. 对超长语义段使用滑动窗口
#         """
#         chunks = []
#
#         if semantic_first:
#             # 策略1：按语义标点粗分（保留标点位置）
#             # 使用正向预查，保留分隔符
#             semantic_splits = re.split(r'([。！？；\n])', text)
#             # 重组：将分隔符拼回前一段
#             segments = []
#             current = ""
#             for i, part in enumerate(semantic_splits):
#                 if i % 2 == 0:  # 文本段
#                     current = part
#                 else:  # 分隔符
#                     current += part
#                     if len(current) > 10:  # 过滤过短片段
#                         segments.append(current)
#                     current = ""
#             if current:
#                 segments.append(current)
#
#             # 检查每个语义段长度
#             for seg in segments:
#                 if len(seg) <= max_size:
#                     chunks.append(seg.strip())
#                 else:
#                     # 语义段仍太长，使用滑动窗口细切
#                     chunks.extend(self._sliding_window(seg, max_size, overlap))
#         else:
#             # 策略2：直接使用滑动窗口（无语义保持）
#             chunks = self._sliding_window(text, max_size, overlap)
#
#         return [c for c in chunks if c.strip()]  # 过滤空串
#
#     def _sliding_window(self, text: str, window_size: int, overlap: int) -> List[str]:
#         """滑动窗口切分，确保上下文连续性"""
#         chunks = []
#         start = 0
#         text_len = len(text)
#
#         while start < text_len:
#             end = min(start + window_size, text_len)
#             chunk = text[start:end]
#
#             # 智能边界：避免在词中间切断（中文按字，英文按词）
#             if end < text_len:
#                 # 尝试向后找最近的语义断点（标点或空格）
#                 look_ahead = min(20, text_len - end)  # 最多看20字符
#                 sub_text = text[end:end + look_ahead]
#
#                 # 找第一个标点或空格
#                 match = re.search(r'[。，！？；,\s]', sub_text)
#                 if match:
#                     end += match.start() + 1
#
#             chunks.append(text[start:end].strip())
#
#             # 步进：窗口大小 - 重叠
#             start += (window_size - overlap)
#
#         return chunks
#
#     def _add_chunk(
#             self,
#             text: str,
#             vec: List[float],
#             image_source: str,
#             sent_idx: int,
#             chunk_idx: int,
#             total_chunks: int,
#             is_chunked: bool,
#             confidence: float,
#             parent_text: str = None
#     ):
#         """统一添加chunk，保证元数据一致性"""
#         chunk_id = f"{image_source}_sent{sent_idx}_chunk{chunk_idx}"
#
#         data = {
#             "id": chunk_id,
#             "text": text,
#             "vector": vec,
#             "metadata": {
#                 "source_image": str(image_source),
#                 "sentence_index": int(sent_idx),
#                 "chunk_index": int(chunk_idx),
#                 "total_chunks": int(total_chunks),
#                 "is_chunked": is_chunked,  # 是否经过切分
#                 "chunk_type": "sentence_chunk" if is_chunked else "sentence",
#                 "confidence": float(confidence),
#                 "text_length": len(text),
#                 "has_parent": parent_text is not None,
#                 # 长文本分块时，保留原始句子的前100字符作为上下文线索
#                 "parent_snippet": parent_text[:100] + "..." if parent_text else None
#             }
#         }
#
#         self.cleanSentence.append(data)
#
#
# # ========== 使用示例 ==========
# if __name__ == "__main__":
#     # 模拟OCR输出：(文本, 置信度)
#     ocr_results = [
#         ("这是一句短文本。", 0.95),
#         ("这是一个超长的OCR识别结果，可能来自一张密集的文字图片，内容包含了很多重要的详细信息，"
#          "需要我们进行合理的切分处理以确保检索质量，同时要保持语义的连贯性和上下文的完整性，"
#          "避免在检索时出现信息割裂的情况。这是第二句补充内容！", 0.88),
#         ("短句2", 0.92),
#     ]
#
#     chunker = OCRChunker()
#     results = chunker.clean_sentences(
#         sentences=ocr_results,
#         image_source="doc_001_page_1",
#         max_chunk_size=50,  # 为了演示，设小一点
#         overlap_size=10,
#         semantic_chunking=True
#     )
#
#     # 打印结果结构
#     for r in results:
#         print(f"ID: {r['id']}")
#         print(f"Text: {r['text'][:30]}...")
#         print(f"Is Chunked: {r['metadata']['is_chunked']}")
#         print(f"Chunk Index: {r['metadata']['chunk_index']}/{r['metadata']['total_chunks']}")
#         print("-" * 40)