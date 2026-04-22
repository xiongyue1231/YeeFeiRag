from pymilvus import MilvusClient, DataType, Collection,Function,FunctionType,FieldSchema,CollectionSchema
from enum import Enum
from typing import List, Dict, Any
from src.app_config.loder import ConfigLoader
import hashlib

config_manager = ConfigLoader()
# # gRPC 方式（推荐，性能更好）
# client = MilvusClient(uri="tcp://localhost:19530")
# print(client.get_server_version())
# import pymilvus
# print(pymilvus.__version__)
# 1.milvus版本和pymilvus需要保持一致
# 2.milvus 2.5.x 以上版本才支持bm25，否则会报错

class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(host=config_manager.config.milvus.host, port=config_manager.config.milvus.port)
        self.activate_collection = None

    def init_collection(self, content_type: str):
        collection_name = content_type
        if self.client.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在")
            return collection_name

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=config_manager.config.models.embedding_model[config_manager.config.rag.embedding_model].dims
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                # 分词器，创建字段时，指定分词器， "type": "chinese"中英混合
                analyzer_params={
                    "type": "chinese"
                }
            ),
            FieldSchema(
                name="sparse_bm25",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                is_sparse=True,
            ),
        ]
        # bm25,新增的function函数会将text字段转换为稀疏向量，通过这个方法自动保存
        bm25_func = Function(
            name="bm25_text_emb",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_bm25"]
        )

        # 原生 schema（支持 function！）
        schema = CollectionSchema(
            fields=fields,
            auto_id=False,
            enable_dynamic_field=True
        )
        schema.add_function(bm25_func)

        index_params = self.client.prepare_index_params()

        # 为 vector 字段创建索引
        index_params.add_index(
            field_name="vector",  # 索引字段
            index_type="HNSW",  # 索引类型
            metric_type="IP",  # 度量类型（与 collection 一致）
            params={"M": 16,  # 推荐 16-32   邻居数
                    "efConstruction": 32  # 推荐 ≥ 2*M   从多少个中中选择最相似的
                    }  # 索引参数
        )
        # BM25 稀疏向量索引（SPARSE_WAND 适合 BM25）
        index_params.add_index(
            field_name="sparse_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )

        # 创建表
        self.client.create_collection(collection_name,
                                      schema=schema,
                                      index_params=index_params
                                      )
        return collection_name

    def set_collection(self, content_type: str):
        """切换当前操作的集合"""
        self.activate_collection = content_type
        print(f"切换到集合: {self.activate_collection}")

    def add_document(self, data, content_type: str):
        """添加文档到指定集合"""
        collection_name = content_type if content_type != '' else self.activate_collection

        if not collection_name:
            raise ValueError("请指定 content_type 或先调用 set_collection()")

        if isinstance(data, dict):
            data = [data]
        # for item in data:
        #     if "sparse_bm25" not in item:
        #         item["sparse_bm25"] = {}  # 或 []，根据 Milvus 版本调整

        self.client.insert(collection_name, data=data)
        print(f"向集合 {collection_name} 添加了 {len(data)} 条数据")
        self.client.flush(collection_name)
        return collection_name

    # collection_name 目前默认配置文件的collection_name
    def search_bm25(self, query: str, top_k: int = 5, collection_name: str = config_manager.config.milvus.collection_name) -> List[Dict]:
        """BM25 全文检索（纯文本关键词匹配）"""
        if self.client.get_load_state(collection_name) != "Loaded":
            self.client.load_collection(collection_name)
        search_params = {
            "params": {"drop_ratio_search": 0,"analyzer_name": "cn", "metric_type": "BM25"}
        }
        hits = self.client.search(
            collection_name=collection_name,
            data=[query],  # 直接传原始文本
            anns_field="sparse_bm25",
            limit=top_k,
            search_params=search_params,
            output_fields=["text", "source_hash", "metadata"]
        )
        print(f"【调试】BM25 查询 '{query}' 返回 hits 数量: {len(hits[0]) if hits else 0}")
        return hits[0] if hits else []

    # collection_name 目前默认配置文件的collection_name
    def search_dense(self, query_vector: List[float], top_k: int = 5, collection_name: str = config_manager.config.milvus.collection_name) -> List[Dict]:
        """稠密向量检索（语义匹配）"""
        if self.client.get_load_state(collection_name) != "Loaded":
            self.client.load_collection(collection_name)

        hits = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=top_k,
            output_fields=["text", "source_hash", "metadata"]
        )

        return hits[0] if hits else []

    def search_hybrid(self, collection_name: str, query: str, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """混合检索：BM25（关键词）+ 稠密向量（语义）"""
        if self.client.get_load_state(collection_name) != "Loaded":
            self.client.load_collection(collection_name)

        # 同时查两个字段
        hits = self.client.search(
            collection_name=collection_name,
            data=[query, query_vector],  # [BM25文本, 向量]
            anns_field=["sparse_bm25", "vector"],
            limit=top_k,
            output_fields=["text", "source_hash", "metadata"]
        )
        return hits[0] if hits else []

    def get_all_collections(self) -> List[str]:
        """获取所有 Collection 名称列表"""
        return self.client.list_collections()

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
        ✅ 最终无报错版：自动判断集合 + 正确参数名
        """
        # 1. 集合不存在直接跳过，不报错
        if not self.client.has_collection(collection_name):
            print(f"⚠️ 集合 {collection_name} 不存在，无需删除旧数据")
            return

        # 2. 加载集合
        try:
            if self.client.get_load_state(collection_name) != "Loaded":
                self.client.load_collection(collection_name)
        except Exception as e:
            print(f"⚠️ 集合 {collection_name} 加载失败: {str(e)}")
            return

        # 3. 执行删除
        try:
            filter_expr = f"source_hash == '{source_hash}'"
            # 正确写法：MilvusClient 用 filter=
            self.client.delete(
                collection_name=collection_name,
                filter=filter_expr
            )
            print(f"✅ 成功删除集合 {collection_name} 中 hash={source_hash} 的旧数据")
        except Exception as e:
            print(f"❌ 删除数据失败: {str(e)}")


# 在 __main__ 里测试
if __name__ == '__main__':
    client = MilvusClient(uri="tcp://localhost:19530")
    print(f"服务器版本: {client.get_server_version()}")

    # 测试分析器是否工作
    try:
        result = client.run_analyzer("人工智能生成内容的伦理挑战", analyzer_params={"type": "chinese"})
        print(f"分词结果: {result}")
    except Exception as e:
        print(f"分析器测试失败: {e}")