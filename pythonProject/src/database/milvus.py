from pymilvus import MilvusClient, DataType, Collection
from enum import Enum
from typing import List, Dict, Any
from src.app_config.loder import ConfigLoader
import hashlib

config_manager = ConfigLoader()
# print(f"protobuf 版本: {google.protobuf.__version__}")
#
# # gRPC 方式（推荐，性能更好）
client = MilvusClient(uri="tcp://localhost:19530")
print(client.get_server_version())


# client = MilvusClient(host=config_manager.config.milvus.host, port=config_manager.config.milvus.port)
# print(client.get_server_version())

class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(host=config_manager.config.milvus.host, port=config_manager.config.milvus.port)
        self.activate_collection = None

    def init_collection(self, content_type: str):
        collection_name = content_type
        print()
        if self.client.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在")
            return collection_name

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # 允许动态字段（可选）
        )

        # 主键字段
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)

        model_name = config_manager.config.rag.embedding_model
        # 向量字段
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR,
                         dim=config_manager.config.models.embedding_model[model_name].dims)
        # 文本字段
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)  # 存储原文
        index_params = self.client.prepare_index_params()

        # 为 vector 字段创建索引
        index_params.add_index(
            field_name="vector",  # 索引字段
            index_type="HNSW",  # 索引类型
            metric_type="IP",  # 度量类型（与 collection 一致）
            params={"M": 16,  # 推荐 16-32
                    "efConstruction": 32  # 推荐 ≥ 2*M
                    }  # 索引参数
        )
        # 创建表
        self.client.create_collection(collection_name,
                                      schema=schema,
                                      metric_type="IP",
                                      index_params=index_params
                                      )
        return collection_name

    def set_collection(self, content_type: str):
        """切换当前操作的集合"""
        self.activate_collection = content_type
        print(f"切换到集合: {self.activate_collection}")

    def add_document(self, data, content_type: str):
        """添加文档到指定集合"""
        collection = content_type if content_type != '' else self.activate_collection

        if not collection:
            raise ValueError("请指定 content_type 或先调用 set_collection()")

        self.client.insert(collection, data=data)
        print(f"向集合 {collection} 添加了 {len(data)} 条数据")
        return collection

    def get_all_collections(self) -> List[str]:
        """获取所有 Collection 名称列表"""
        return self.client.list_collections()

    def search_all_collections(self, query, query_vector: List[float], top_k: int = 5) -> Dict[str, List[Dict]]:
        """在所有已加载的 Collection 中搜索（跨库搜索）"""
        results = {}

        for collection_name in self.get_all_collections():
            try:
                # 检查是否已加载
                if self.client.get_load_state(collection_name) != "Loaded":
                    self.client.load_collection(collection_name)

                # 如果query_length>0 则使用原始文本进行搜索，否则使用向量进行搜索
                if len(query) > 0:
                    hits = self.client.search(
                        collection_name=collection_name,
                        data=query,
                        anns_field="text",
                        limit=top_k,
                        output_fields=["text", "metadata"]  # 返回所有字段
                    )
                else:
                    # 执行搜索
                    hits = self.client.search(
                        collection_name=collection_name,
                        data=[query_vector],
                        anns_field="vector",
                        limit=top_k,
                        output_fields=["*"]  # 返回所有字段
                    )

                results[collection_name] = hits

            except Exception as e:
                print(f"搜索 {collection_name} 失败: {e}")
                results[collection_name] = []

        return results

    def get_file_md5(self, file_path: str) -> str:
        """
        计算文件的 MD5 值（根据文件内容，不是文件名）
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def delete_old_chunks_by_hash(self, collection_name: str, source_hash: str):
        """
        根据文件 hash 删除旧的向量数据（去重核心）
        """
        coll = Collection(collection_name)
        coll.load()  # 必须 load 才能删

        # 拼接删除表达式
        expr = f'source_hash == "{source_hash}"'

        # 执行删除
        coll.delete(expr)
        print(f"✅ 已删除旧数据，hash={source_hash}")


if __name__ == '__main__':
    milvus_manager = MilvusManager()
    milvus_manager.search_all_collections("人工智能", [])
