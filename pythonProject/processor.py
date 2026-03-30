from chuck import OCRChuck
from file_handler import FileHandler
from milvus import MilvusStore
from pathlib import Path
from milvus import ContentType
from content_type import ContentType


class DocumentProcessor:

    def __init__(self):
        self.file_handler = FileHandler()
        self.chuck_handler = OCRChuck()
        self.store_to_milvus = MilvusStore()

    def process_and_store(self, file_path: str):
        # 1. 提取内容（包含OCR）
        content = self.file_handler.extract_content(file_path)
        # 传入文件类型
        filetype = ContentType().detect_type(file_path)

        # 2. 文本分块
        data = self.chuck_handler.clean_sentences(content, file_path, filetype)

        # 创建milvus表
        self.store_to_milvus.init_collection(ContentType().SUPPORTED_TYPES[filetype])
        self.store_to_milvus.set_collection(ContentType().SUPPORTED_TYPES[filetype])
        # 3. 入库（统一在这里处理）
        self.store_to_milvus.add_document(data)

        return {"status": "success"
                # , "chunks": len(chunks)
                }


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_and_store("./imgtest/milvus.txt")
    # processor.process_and_store("./imgtest/test.png")
