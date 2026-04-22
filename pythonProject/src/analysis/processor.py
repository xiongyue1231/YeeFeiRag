from src.embed.chuck import OCRChuck
from file_handler import FileHandler
from src.database.milvus import MilvusManager
from content_type import ContentType
from src.app_config.loder import ConfigLoader


config_manager = ConfigLoader()


# 类似于BLL
class DocumentProcessor:

    def __init__(self):
        self.file_handler = FileHandler()
        self.chuck_handler = OCRChuck()
        self.store_to_milvus = MilvusManager()

    def process_and_store(self, file_path: str,file_type:str, collection_name: str, knowledge_id: int, document_id: int):
        if collection_name is None:
            collection_name = config_manager.config.milvus.collection_name
        # 1. 提取内容（包含OCR）
        content = self.file_handler.extract_content(file_path)
        # 获取文件hash
        source_hash = self.store_to_milvus.get_file_md5(file_path)
        # 删除旧数据
        self.store_to_milvus.delete_old_chunks_by_hash(collection_name, source_hash)
        # 2. 文本分块
        data = self.chuck_handler.clean_sentences(content, file_path, file_type, source_hash, knowledge_id, document_id)

        # 初始化Collection     CollectionType[filetype]
        # 以后如果有特殊要求，这里进行切换collection
        self.store_to_milvus.init_collection(collection_name)
        # 设置当前collection
        self.store_to_milvus.set_collection(collection_name)
        # 3. 入库（统一在这里处理）
        self.store_to_milvus.add_document(data, collection_name)

        return {"status": "success"
                # , "chunks": len(chunks)
                }


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_and_store("../../imgtest/新建 DOCX 文档.docx", "docx",
                                "current_collection_test", 1, 1)
    # processor.process_and_store("./imgtest/milvus.txt")
    # processor.process_and_store("./imgtest/test.png")
