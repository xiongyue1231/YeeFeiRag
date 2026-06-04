from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
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
from src.rag.MultiDialogueRag import get_session_history
# ---------- 加载配置 ----------
config_manager = ConfigLoader()
device = config_manager.config.deviceSettings.device


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
    创建一个带对话历史的 RAG 链，支持多轮对话。
    """
    # 初始化 LLM 目前设置为 ChatOpenAI方式
    llm = create_llm_langchain(config_manager.config.rag)

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

    # 输入字典需包含：question（原始问题）、chat_history（历史消息列表）
    chain = (
            {
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
    """
    rag_chain = create_rag_chain(knowledge_id)

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
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
    session_history = get_session_history(session_id).clear()
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
    for msg in get_session_history(session_id).messages:
        print(f"{msg.type}: {msg.content}")