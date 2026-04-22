import yaml  # type: ignore
from typing import Union, List, Any, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.embed.embedding import VecEmbedding
from src.database.milvus import MilvusManager
from src.app_config.loder import ConfigLoader
from src.core.utils import create_llm_client
from src.prompts.templates import get_prompt_template
import os
from pathlib import Path

config_manager = ConfigLoader()
device = config_manager.config.deviceSettings.device

EMBEDDING_MODEL_PARAMS = {}
sorted_content = {}
sorted_records = {}


def load_rerank_model(model_name: str, model_path: str) -> None:
    global EMBEDDING_MODEL_PARAMS
    """
    加载重排序模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """

    current_dir = Path(__file__).parent.parent.resolve()
    model_path = current_dir / "models" / "BAAI" / "bge-reranker-base"
    model_path = str(model_path)
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(config_manager.config.deviceSettings.device)


if config_manager.config.rag.use_rerank:
    model_name = config_manager.config.rag.rerank_model
    model_info = config_manager.config.models.rerank_model[model_name]
    model_path = model_info.local_url

    print(f"加载重排序模型： {model_name} ")
    load_rerank_model(model_name, model_path)


class Rag:
    def __init__(self):
        self.rerank_model = config_manager.config.rag.rerank_model
        self.device = config_manager.config.deviceSettings.device  # 设备cpu还是gpu
        self.client = create_llm_client(config_manager.config.rag)
        self.llm_model = config_manager.config.rag.model
        self.Vec = VecEmbedding()
        self.use_rerank = config_manager.config.rag.use_rerank
        self.use_rrf = config_manager.config.rag.use_rrf
        self.milvus = MilvusManager()
        self.chunk_candidate = config_manager.config.rag.chunk_candidate

    def get_rank(self, text_pair) -> np.ndarray:
        """
        对文本对进行重排序
        :param text_pair: 待排序文本
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair,
                    padding=True,  # 填充
                    truncation=True,  # 截断
                    return_tensors='pt',  # 转换为张量
                    max_length=512,  # 最大长度
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores

        # raise NotImplemented

    # knowledge_id知识库ID
    def query_document(self, query: str, knowledge_id: int) -> List[str]:
        global sorted_content, sorted_records
        # 全文检索，指定一个知识库检索，bm25打分
        word_search_response = self.milvus.search_bm25(query, top_k=5)
        # 语义检索
        embedding_vector = self.Vec.get_embedding(query)  # 编码
        vector_search_response = self.milvus.search_dense(embedding_vector, top_k=5)

        if self.use_rrf:
            # rrf 融合排序算法
            # 排名越靠前，分数越高；多个来源都出现的结果，分数叠加。
            # 例如当前进行了bm25检索，然后进行向量检索，然后又通过向量检索，如果搜索出某个结果，2种算法都有score，那么最终结果就是2个score之和。
            k = 60
            fusion_score = {}
            search_id2record = {}
            for idx, record in enumerate(word_search_response):
                _id = record["id"]
                if _id not in fusion_score:
                    fusion_score[_id] = 1 / (idx + k)
                else:
                    fusion_score[_id] += 1 / (idx + k)

                if _id not in search_id2record:
                    search_id2record[_id] = record["entity"]

            for idx, record in enumerate(vector_search_response):
                _id = record["id"]
                if _id not in fusion_score:
                    fusion_score[_id] = 1 / (idx + k)
                else:
                    fusion_score[_id] += 1 / (idx + k)

                if _id not in search_id2record:
                    search_id2record[_id] = record["entity"]

            sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
            sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
            sorted_content = [x["text"] for x in sorted_records]

        if self.use_rerank:
            text_pair = []
            for chunk_content in sorted_content:
                text_pair.append([query, chunk_content])
            rerank_score = self.get_rank(text_pair)  # 重排序打分
            rerank_idx = np.argsort(rerank_score)[::-1]

            sorted_records = [sorted_records[x] for x in rerank_idx]
            sorted_content = [sorted_content[x] for x in rerank_idx]

        return sorted_records

    def chat_with_rag(
            self,
            knowledge_id: int,  # 知识库 哪一个知识库提问
            messages: List[Dict],
    ):
        # 用户的第一次提问用rag

        if len(messages) == 1:
            query = messages[0]["content"]
            print(f"【大模型改写用户提问】【原始：{query}】")
            # 对用户提问进行改写  目前使用大模型进行改写
            query = self.chat([{"role": "system", "content": get_prompt_template("rewriter")["system"]},
                               {"role": "user",
                                "content": get_prompt_template("rewriter")["user"].replace("{original_input}", query)}],0.1,0.1
                              ).content
            print(f"【大模型改写用户提问】【改写后：{query}】")
            related_records = self.query_document(query, knowledge_id)  # 检索到相关的文档
            print(related_records)
            related_document = '\n'.join([x["text"] for x in related_records])

            rag_system_prompt = get_prompt_template("basic_rag")["system"]
            rag_query = get_prompt_template("basic_rag")["user"].replace("{input}", query) \
                .replace("{all_document_str}", related_document)

            messages = self.chat(
                [{"role": "system", "content": rag_system_prompt},
                 {"role": "user", "content": rag_query}],
                0.5, 0.7
            ).content

        # 后序提问 直接大模型回答
        else:
            pass
            # normal_response = self.chat(
            #     messages,
            #     0.7, 0.9
            # ).content
            # messages.append({"role": "system", "content": normal_response})

        # messages.append({"role": "system", "content": rag_response})
        return messages


    def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
        """
        temperature 温度参数
        top_p 概率
        """
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message


if __name__ == "__main__":
    rag = Rag()
    res= rag.chat_with_rag(1, [{"role": "user", "content": "你好"}])
    print(res)
