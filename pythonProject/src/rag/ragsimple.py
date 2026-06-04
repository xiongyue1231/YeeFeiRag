from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
# from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
# 假设你原来的模块路径可导入
from src.embed.embedding import VecEmbedding
from src.database.milvus import MilvusManager
from src.app_config.loder import ConfigLoader
from src.prompts.templates import get_prompt_template
from src.core.utils import create_llm_langchain
from src.rag.ragbase import HybridRetriever
from operator import itemgetter
from langchain_community.chat_message_histories import RedisChatMessageHistory
from src.rag.MultiDialogueRag import (get_session_history as get_redis_session_history)
import json
# ---------- 加载配置 ----------
config_manager = ConfigLoader()
device = config_manager.config.deviceSettings.device


# ---------- 历史对话压缩工具函数 ----------
def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量（平均每个 token 约 1.3 个中文字符或 0.75 个英文单词）
    """
    # 简单的 token 估算方法，可根据实际需要替换为更准确的 tokenizer
    if not text:
        return 0
    # 混合文本的简单估算：平均每个 token 约 2 个字符
    return len(text) // 2


def compress_history_with_llm(messages: List[BaseMessage], llm) -> str:
    """
    使用 LLM 压缩历史对话
    """
    compress_prompt = ChatPromptTemplate.from_messages([
        ("system", "请将以下对话历史压缩成简洁的摘要，保留关键信息，不超过200字。"),
        ("user", "{history}")
    ])
    
    history_text = "\n".join([
        f"{'用户' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in messages
    ])
    
    chain = compress_prompt | llm | StrOutputParser()
    return chain.invoke({"history": history_text})


def compress_chat_history(
    chat_history: List[BaseMessage],
    llm,
    llm_max_token: int,
    max_recent_rounds: int = 2
) -> List[BaseMessage]:
    """
    压缩历史对话，保留最近完整的 N 轮对话，之前的历史进行压缩
    
    Args:
        chat_history: 完整的对话历史列表
        llm: LLM 实例
        llm_max_token: LLM 最大 token 限制
        max_recent_rounds: 保留的最近完整对话轮数，默认 2 轮
    
    Returns:
        压缩后的对话历史
    """
    if not chat_history:
        return []
    
    # 计算每轮对话（1个用户消息 + 1个AI消息为1轮）
    total_messages = len(chat_history)
    messages_to_keep = max_recent_rounds * 2  # 每轮2条消息
    
    # 如果历史消息较少，直接返回
    if total_messages <= messages_to_keep:
        return chat_history
    
    # 分割历史：需要压缩的部分 + 保留的部分
    messages_to_compress = chat_history[:-messages_to_keep]
    messages_to_keep_list = chat_history[-messages_to_keep:]
    
    # 估算压缩前的总 token 数
    total_tokens = sum([estimate_tokens(msg.content) for msg in chat_history])
    
    # 只有当总 token 超过阈值时才进行压缩
    if total_tokens <= llm_max_token:
        return chat_history
    
    # 使用 LLM 压缩旧历史
    compressed_summary = compress_history_with_llm(messages_to_compress, llm)
    
    # 创建压缩后的消息
    compressed_messages = [
        AIMessage(content=f"【历史对话摘要】{compressed_summary}")
    ]
    
    # 返回：压缩摘要 + 最近保留的对话
    return compressed_messages + messages_to_keep_list


# ---------- 问题改写器（可选）----------
def create_query_rewriter(llm):
    """返回一个 Runnable，用于改写用户最新问题"""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("rewriter")["system"]),
        ("user", get_prompt_template("rewriter")["user"])
    ])
    # 注意：模板中有 {original_input} 占位符
    return rewrite_prompt | llm | StrOutputParser()


# ---------- 构建多轮对话 RAG 链 ----------
def create_rag_chain(knowledge_id: int):
    """
    创建一个带对话历史的 RAG 链，支持多轮对话和历史压缩。
    """
    # 初始化 LLM 目前设置为 ChatOpenAI方式
    llm = create_llm_langchain(config_manager.config.rag)
    
    # 获取配置参数
    llm_max_token = config_manager.config.multi_dialogue_rag.llm_max_token

    # 初始化检索器
    retriever = HybridRetriever(
        vec_embedding=VecEmbedding(),
        milvus=MilvusManager(),
        use_rrf=config_manager.config.rag.use_rrf,
        use_rerank=config_manager.config.rag.use_rerank,
        chunk_candidate=config_manager.config.rag.chunk_candidate,
        knowledge_id=knowledge_id,
    )

    # 问题改写器（可选：每次对话都先改写用户问题以提升检索效果）
    query_rewriter = create_query_rewriter(llm)

    # 构建 RAG 提示模板（包含对话历史）
    system_prompt = get_prompt_template("basic_rag")["system"]
    user_template = get_prompt_template("basic_rag")["user"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # 对话历史占位符
        ("user", user_template),
    ])

    # 定义链的处理流程
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    # 历史对话压缩处理函数
    def process_chat_history(input_dict: dict) -> dict:
        """处理和压缩对话历史"""
        chat_history = input_dict.get("chat_history", [])
        # 压缩历史对话，保留最近2轮
        compressed_history = compress_chat_history(
            chat_history=chat_history,
            llm=llm,
            llm_max_token=llm_max_token,
            max_recent_rounds=2
        )
        return {
            **input_dict,
            "chat_history": compressed_history
        }

    # 输入字典需包含：question（原始问题）、chat_history（历史消息列表）
    chain = (
            RunnableLambda(process_chat_history)
            | {
                "rewritten_query": query_rewriter,  # 改写后的问题，用于检索
                "input": itemgetter("input"),  # 保留原始问题，可用于回答
                "chat_history": itemgetter("chat_history"),
            }
            | RunnablePassthrough.assign(docs=(lambda x: retriever.invoke(x["rewritten_query"])))
            | RunnablePassthrough.assign(all_document_str=lambda x: _format_docs(x["docs"]))
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def create_conversational_rag(knowledge_id: int):
    """
    返回带消息历史的对话 RAG 实例，可直接调用 .invoke() 进行多轮对话。
    Args:
        knowledge_id: 知识库 ID
        ttl: Redis 会话过期时间（秒），默认 3600
    """
    rag_chain = create_rag_chain(knowledge_id)

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_redis_session_history,
        input_messages_key="input",  # 输入中的用户问题键名
        history_messages_key="chat_history",
        output_messages_key="output",  # 输出回答的键名
    )

    return conversational_rag


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 初始化带多轮对话的 RAG
    conversational_rag = create_conversational_rag(knowledge_id=1)
    # 模拟多轮对话
    session_id = "user_123"
    # 第一轮
    response1 = conversational_rag.invoke(
        {"input": "这是毕业论文？"},
        config={"configurable": {"session_id": session_id}}
    )
    print("AI:", response1)

    # 第二轮，会携带历史上下文
    response2 = conversational_rag.invoke(
        {"input": "人工智能如何在企业中应用"},
        config={"configurable": {"session_id": session_id}}
    )
    print("AI:", response2)

    # 查看历史消息
    print("\n对话历史：")
    for msg in get_redis_session_history[session_id].messages:
        print(f"{msg.type}: {msg.content}")