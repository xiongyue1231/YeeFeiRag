"""
意图路由器：规则层 + LLM 意图分类 + 失败兜底走 RAG

链路位置：process_chat_history 之后、RAG/闲聊分支之前。
输入/输出均为 dict，输出在原 state 基础上追加 "intent" 键（"chat" | "rag"）。
"""
import re
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from src.app_config.loder import ConfigLoader
from src.core.utils import create_llm_langchain
from src.prompts.templates import get_prompt_template

config_manager = ConfigLoader()

INTENT_CHAT = "chat"
INTENT_RAG = "rag"

# 规则层：明显闲聊（问候/感谢/道别/应答词/纯表情），零成本直接放行，不调用 LLM
_CHITCHAT_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|嗨|哈喽|在吗|早上好|上午好|下午好|晚上好|"
    r"谢谢|感谢|多谢|辛苦了|好的|好吧|嗯+|哦+|噢+|哈+|嘿+|"
    r"再见|拜拜|bye|goodbye)[啊呀哈哦!！?？。~～\s]*$",
    re.IGNORECASE,
)


def _rule_classify(text: str) -> Optional[str]:
    """规则层分类，命中返回意图，未命中返回 None"""
    if not text:
        return INTENT_CHAT  # 空输入没有检索价值，直接闲聊
    if _CHITCHAT_PATTERN.match(text):
        return INTENT_CHAT
    return None


def _create_router_llm():
    """创建路由专用 LLM：配置了 router_model 则用小模型，否则复用主模型"""
    rag_cfg = config_manager.config.rag
    router_model = (getattr(rag_cfg, "router_model", "") or "").strip()
    if router_model:
        # 同时覆盖 openai / vllm 两种 provider 的模型名，保证 model_copy 后可创建客户端
        llm_cfg = rag_cfg.model_copy(update={"llm_model": router_model, "vllm_model": router_model})
    else:
        llm_cfg = rag_cfg
    return create_llm_langchain(llm_cfg)


def create_intent_classifier():
    """
    构建意图分类节点（Runnable）。

    三层策略：
    1. 规则层：问候/感谢等明显闲聊直接判定，零成本
    2. LLM 层：带最近几轮历史做上下文分类
    3. 兜底：LLM 超时/异常时默认走 RAG（宁多检索，不出错）
    """
    llm = _create_router_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", get_prompt_template("intent_router")["system"]),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", get_prompt_template("intent_router")["user"]),
    ])
    classifier = prompt | llm

    def _classify(state: Dict[str, Any]) -> Dict[str, Any]:
        text = str(state.get("input", "")).strip()

        # 1. 规则层
        rule_result = _rule_classify(text)
        if rule_result is not None:
            print(f"【意图路由】规则命中 → {rule_result} | input: {text[:50]}", flush=True)
            return {**state, "intent": rule_result}

        # 2. LLM 层（带最近 4 条历史，解决"那第二个呢"这类指代问题）
        try:
            recent_history = state.get("chat_history", [])[-4:]
            raw = classifier.invoke({"input": text, "chat_history": recent_history})
            content = str(raw.content)
            # 思考型模型（如 qwen3）输出 <think>...</think> 前缀，只取思考后的最终结论
            if "</think>" in content:
                content = content.split("</think>")[-1]
            answer = content.strip().lower()
            # rag 优先：两个词都出现、或输出无法识别时，一律走 RAG（宁多检索不出错）
            if "rag" in answer:
                intent = INTENT_RAG
            elif "chat" in answer:
                intent = INTENT_CHAT
            else:
                intent = INTENT_RAG
            print(f"【意图路由】LLM 判定 → {intent} | 原始输出: {answer[:50]} | input: {text[:50]}", flush=True)
        except Exception as e:
            # 3. 兜底：分类失败默认走 RAG
            intent = INTENT_RAG
            print(f"【意图路由】LLM 分类失败，默认走 RAG | input: {text[:50]} | err: {e}", flush=True)

        return {**state, "intent": intent}

    return RunnableLambda(_classify)
