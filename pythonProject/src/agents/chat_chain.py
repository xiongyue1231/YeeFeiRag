"""
闲聊链：意图路由判定为 chat 时使用。

跳过 问题改写 / 混合检索 / 重排序，直接 prompt + LLM 生成，
输入输出契约与 RAG 分支保持一致（dict 输入、str 输出）。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from src.prompts.templates import get_prompt_template


def create_chat_chain(llm):
    """
    构建闲聊链。

    Args:
        llm: 与 RAG 链共用的 LangChain LLM 实例

    Returns:
        Runnable：输入 {input, chat_history, ...}，输出 str
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("chat")["system"]),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", get_prompt_template("chat")["user"]),
    ])

    return (
        prompt
        | llm
        | StrOutputParser()
    )
