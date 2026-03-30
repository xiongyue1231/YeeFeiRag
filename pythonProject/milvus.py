from paddleOcr import SimpleOcr
from pymilvus import MilvusClient,DataType
from chuck import OCRChuck
import google.protobuf
from paddleocr import PaddleOCR
from enum import Enum
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# print(f"protobuf 版本: {google.protobuf.__version__}")
#
# # gRPC 方式（推荐，性能更好）
# client = MilvusClient(uri="tcp://localhost:19530")
# print(client.get_server_version())

client = MilvusClient(host="localhost", port="19530")
print(client.get_server_version())


class ContentType(Enum):
    word = "text_collection"  # 文档数据
    image = "image_collection"  # 图片 OCR
    excel = "document_collection"  # 表格数据


class MilvusStore:
    def __init__(self):
        self.client = MilvusClient(host="localhost", port="19530")
        self.activate_collection = None

    def init_collection(self, content_type: ContentType):
        collection_name = content_type.value

        if self.client.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在")
            return collection_name

        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # 允许动态字段（可选）
        )

        # 主键字段
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
        # 向量字段
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=config["models"]["embedding_model"]["bge-small-zh-v1.5"]["dims"])
        # 文本字段
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)  # 存储原文
        index_params = client.prepare_index_params()

        # 为 vector 字段创建索引
        index_params.add_index(
            field_name="vector",  # 索引字段
            index_type="IVF_FLAT",  # 索引类型
            metric_type="IP",  # 度量类型（与 collection 一致）
            params={"nlist": 128}  # 索引参数
        )

        self.client.create_collection(collection_name,
                                      schema=schema,
                                      metric_type="IP",
                                      index_params=index_params
                                     )
        return collection_name

    def set_collection(self, content_type: ContentType):
        """切换当前操作的集合"""
        self.activate_collection = content_type.value
        print(f"切换到集合: {self.activate_collection}")

    def add_document(self, data, content_type: ContentType = None):
        """添加文档到指定集合"""
        collection = content_type.value if content_type else self.activate_collection

        if not collection:
            raise ValueError("请指定 content_type 或先调用 set_collection()")

        self.client.insert(collection, data=data)
        return collection
