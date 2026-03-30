from chuck import OCRChuck
from file_handler import FileHandler
from milvus import MilvusStore
from pathlib import Path
from milvus import ContentType


class DocumentProcessor:
    SUPPORTED_TYPES = {
        # 文本类
        '.txt': 'text',
        '.csv': 'text',
        # 文档类
        '.pdf': 'document',
        '.docx': 'document',
        '.doc': 'document',
        '.excel': 'document',
        # 图片类（需要OCR）
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
    }

    def __init__(self):
        self.file_handler = FileHandler()
        self.chuck_handler = Chuck()
        self.store_to_milvus = MilvusStore()
        self.content_type = ContentType

    def detect_type(self, filename: str) -> str:
        """检测文件类型"""
        ext = Path(filename).suffix.lower()
        return self.SUPPORTED_TYPES.get(ext, 'unknown')

    def process_and_store(self, file_path: str):
        # 1. 提取内容（包含OCR）
        content = self.file_handler.extract_content(file_path)

        # 2. 文本分块
        data = self.chuck_handler.clean_sentences(content, file_path)

        # 3. 生成embedding（建议添加）
        # embeddings = await self.generate_embeddings(chunks)

        filetype = self.detect_type(file_path)
        self.store_to_milvus.init_collection(self.content_type[filetype])
        self.store_to_milvus.set_collection(self.content_type[filetype])
        # 4. 入库（统一在这里处理）
        self.store_to_milvus.add_document(data)

        return {"status": "success"
                # , "chunks": len(chunks)
                }


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_and_store("./imgtest/test.png")
