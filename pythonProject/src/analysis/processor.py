from ..embed.chuck import OCRChuck
from file_handler import FileHandler
from ..database.milvus import MilvusManager
from pathlib import Path
from ..database.milvus import CollectionType
from content_type import ContentType


# 类似于BLL
class DocumentProcessor:

    def __init__(self):
        self.file_handler = FileHandler()
        self.chuck_handler = OCRChuck()
        self.store_to_milvus = MilvusManager()

    def process_and_store(self, file_path: str):
        # 1. 提取内容（包含OCR）
        content = self.file_handler.extract_content(file_path)
        # 传入文件类型
        filetype = ContentType().detect_type(file_path)

        # 2. 文本分块
        data = self.chuck_handler.clean_sentences(content, file_path, filetype)

        # 创建milvus表   默认采用一个，没必要通过文件类型去创建多个Collection     CollectionType[filetype]
        self.store_to_milvus.init_collection(CollectionType.document)
        self.store_to_milvus.set_collection(CollectionType.document)
        # 3. 入库（统一在这里处理）
        self.store_to_milvus.add_document(data)

        return {"status": "success"
                # , "chunks": len(chunks)
                }


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_and_store("./imgtest/生成式人工智能在企业信息管理中的应用与伦理挑战研究.docx")
    # processor.process_and_store("./imgtest/milvus.txt")
    # processor.process_and_store("./imgtest/test.png")
