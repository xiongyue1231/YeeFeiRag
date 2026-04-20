import yaml  # type: ignore
from typing import Union, List, Any, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from embedding import VecEmbedding
from milvus import MilvusManager
import datetime
from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()
device = config_manager.config.deviceSettings.device

EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}

BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：
{#RELATED_DOCUMENT#}


问题：{#QUESTION#}
'''


def load_rerank_model(model_name: str, model_path: str) -> None:
    """
    加载重排序模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to()


if  config_manager.config.rag.use_rerank:
    model_name = config_manager.config.rag.rerank_model
    model_info = config_manager.config.models.rerank_model[model_name]
    model_path =model_info.local_url

    print(f"Loading rerank model {model_name} from model_path...")
    load_rerank_model(model_name, model_path)


class Rag:
    def __init__(self):
        self.rerank_model = config_manager.config.rag.rerank_model
        self.device = config_manager.config.deviceSettings.device
        self.client = OpenAI(
            api_key=config_manager.config.rag.llm_api_key if config_manager.config.rag.is_llm else config_manager.config.rag.vllm_api_key,
            base_url=config_manager.config.rag.llm_base if config_manager.config.rag.is_llm else config_manager.config.rag.vllm_base
        )
        self.llm_model =config_manager.config.rag.llm_model if config_manager.config.rag.is_llm else config_manager.config.rag.vllm_model
        self.Vec = VecEmbedding()
        self.use_rerank =config_manager.config.rag.use_rerank
        self.milvus = MilvusManager()
        self.chunk_candidate =config_manager.config.rag.chunk_candidate

    def get_rank(self, text_pair) -> np.ndarray:
        """
        对文本对进行重排序
        :param text_pair: 待排序文本
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair, padding=True, truncation=True,
                    return_tensors='pt', max_length=512,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores

        # raise NotImplemented

    def query_document(self, query: str, knowledge_id: int) -> List[str]:

        # 全文检索，指定一个知识库检索，bm25打分
        word_search_response = self.milvus.search_all_collections(query, [], top_k=50)
        # 语义检索
        embedding_vector = self.Vec.get_embedding(query)  # 编码
        vector_search_response = self.milvus.search_all_collections('', embedding_vector, top_k=50)

        # rrf
        # 检索1 ：[a， b， c]
        # 检索2 ：[b， e， a]
        # a 1/60    b 1/61    c 1/62
        # b 1/60    e 1/61    a 1/62

        k = 60
        fusion_score = {}
        search_id2record = {}
        for idx, record in enumerate(word_search_response['hits']):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)

            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        for idx, record in enumerate(vector_search_response['hits']):
            _id = record["_id"]
            if _id not in fusion_score:
                fusion_score[_id] = 1 / (idx + k)
            else:
                fusion_score[_id] += 1 / (idx + k)

            if _id not in search_id2record:
                search_id2record[_id] = record["fields"]

        sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
        sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
        sorted_content = [x["chunk_content"] for x in sorted_records]

        if self.use_rerank:
            text_pair = []
            for chunk_content in sorted_content:
                text_pair.append([query, chunk_content])
            rerank_score = self.get_rank(text_pair)  # 重排序打分
            rerank_idx = np.argsort(rerank_score)[::-1]

            sorted_records = [sorted_records[x] for x in sorted_records]
            sorted_content = [sorted_content[x] for x in sorted_content]

        return sorted_records

    def chat_with_rag(
            self,
            knowledge_id: int,  # 知识库 哪一个知识库提问
            messages: List[Dict],
    ):
        # 用户的第一次提问用rag
        # 对用户提问进行改写
        if len(messages) == 1:
            query = messages[0]["content"]
            related_records = self.query_document(query, knowledge_id)  # 检索到相关的文档
            print(related_records)
            related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

            rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                .replace("{#QUESTION#}", query) \
                .replace("{#RELATED_DOCUMENT#}", related_document)

            rag_response = self.chat(
                [{"role": "user", "content": rag_query}],
                0.7, 0.9
            ).content
            messages.append({"role": "system", "content": rag_response})
        # 后序提问 直接大模型回答
        else:
            normal_response = self.chat(
                messages,
                0.7, 0.9
            ).content
            messages.append({"role": "system", "content": normal_response})

        # messages.append({"role": "system", "content": rag_response})
        return messages

    def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message

if __name__ == "__main__":
    rag = Rag()
    rag.chat_with_rag(1, [{"role": "user", "content": "请给我一个房间的描述"}])