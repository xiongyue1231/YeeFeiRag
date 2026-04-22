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

    def process_and_store(self, file_path: str):
        # 1. 提取内容（包含OCR）
        content = self.file_handler.extract_content(file_path)
        # 传入文件类型
        filetype = ContentType().detect_type(file_path)
        # 获取文件hash
        source_hash = self.store_to_milvus.get_file_md5(file_path)
        # 删除旧数据
        self.store_to_milvus.delete_old_chunks_by_hash(config_manager.config.milvus.collection_name, source_hash)
        # 2. 文本分块
        data = self.chuck_handler.clean_sentences(content, file_path, filetype, source_hash)

        # 初始化Collection     CollectionType[filetype]
        # 以后如果有特殊要求，这里进行切换collection
        self.store_to_milvus.init_collection(config_manager.config.milvus.collection_name)
        # 设置当前collection
        self.store_to_milvus.set_collection(config_manager.config.milvus.collection_name)
        # 3. 入库（统一在这里处理）
        self.store_to_milvus.add_document(data, config_manager.config.milvus.collection_name)

        return {"status": "success"
                # , "chunks": len(chunks)
                }


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_and_store("../../imgtest/生成式人工智能在企业信息管理中的应用与伦理挑战研究.docx")
    # processor.process_and_store("./imgtest/milvus.txt")
    # processor.process_and_store("./imgtest/test.png")
