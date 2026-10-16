"""
意图路由器：规则层 + 本地意图分类模型（Intent_classification.py）+ 失败兜底走 RAG

链路位置：process_chat_history 之后、RAG/闲聊分支之前。
输入/输出均为 dict，输出在原 state 基础上追加 "intent" 键（"chat" | "rag"）。
"""
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableLambda

from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()

INTENT_CHAT = "chat"
INTENT_RAG = "rag"

# 项目根目录（pythonProject）：保证任意 cwd 下都能 import 根目录下的 Intent_classification
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Intent_classification as intent_service  # noqa: E402

# 规则层：明显闲聊（问候/感谢/道别/应答词/纯表情），零成本直接放行，不调用模型
_CHITCHAT_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|嗨|哈喽|在吗|早上好|上午好|下午好|晚上好|"
    r"谢谢|感谢|多谢|辛苦了|好的|好吧|嗯+|哦+|噢+|哈+|嘿+|"
    r"再见|拜拜|bye|goodbye)[啊呀哈哦!！?？。~～\s]*$",
    re.IGNORECASE,
)

# 对话元问题：追溯/复述/总结"对话本身"，答案在 chat_history 里而不是知识库里。
# 这类输入走 RAG 会命中一堆无关片段，而 RAG 提示词又禁止模型用历史作答，必然答非所问。
# 例如："刚才我问的什么" / "上面说了什么" / "重复一遍" / "总结一下对话"
# 注意：每条都要求句子在回溯词后基本结束，避免误伤"前面说的 nginx 怎么停止"这类
# 带指代但实质仍需检索的问题（那种交给改写器 + RAG 处理）。
_META_PATTERN = re.compile(
    r"|".join([
        # 刚才/上面/之前 （我）问/说/讲/提/聊 了 什么
        r"^(刚才|刚刚|之前|前面|上面|上次|上一轮)?\s*(我|我们)?\s*(问|说|讲|提|聊)(了|过)?的?"
        r"\s*(是)?\s*(什么|啥|哪些)(呢|吗)?[\?？。!！~～\s]*$",
        # 我 刚才 问 的 什么
        r"^(我|我们)\s*(刚才|刚刚|之前|前面|上面|上次|上一轮)\s*(问|说|讲|提|聊)(了|过)?的?"
        r"\s*(是)?\s*(什么|啥|哪些)(呢|吗)?[\?？。!！~～\s]*$",
        # 重复/再说/复述 一遍（我的问题）
        r"^(重复|再说|复述)(一下|一遍)?\s*(我|你|我们)?\s*(刚才|之前|上面|上次)?\s*(的)?"
        r"\s*(问题|话|内容)?[\?？。!！~～\s]*$",
        # 总结一下（我们的）对话
        r"^总结(一下|下)?\s*(我们|咱|这段)?\s*(的)?\s*(对话|聊天|内容|讨论)?[\?？。!！~～\s]*$",
        # 我们刚才聊了什么
        r"^(我们|咱)\s*(刚才|之前|前面|上面|上次)?\s*(聊|谈|讨论)(了|过)?"
        r"\s*(什么|啥|哪些)(呢|吗)?[\?？。!！~～\s]*$",
        # 对话历史 / 聊天记录
        r"^(对话|聊天)(历史|记录)[\?？。!！~～\s]*$",
    ])
)

# 带多少条历史做上下文分类（解决"那第二个呢"这类指代问题）
RECENT_HISTORY_SIZE = 4

# 意图分类节点缓存：整条链只允许构建一次（否则每个请求都会重复预加载 1.5B 模型）
_classifier_cache = None
_classifier_lock = threading.Lock()
# 模型层是否可用：预加载失败/冷却期内为 False，直接走兜底，不再逐请求重试
_model_usable = True
_model_path = ""
_model_device = "cpu"


def _rule_classify(text: str, has_history: bool = False) -> Optional[str]:
    """规则层分类，命中返回意图，未命中返回 None"""
    if not text:
        return INTENT_CHAT  # 空输入没有检索价值，直接闲聊
    if _CHITCHAT_PATTERN.match(text):
        return INTENT_CHAT
    # 对话元问题（"刚才我问的什么"）：答案在历史里，且必须确实有历史才成立，
    # 否则仍走检索分支，让提示词给出"资料中无法回答"的标准答复
    if has_history and _META_PATTERN.match(text):
        return INTENT_CHAT
    return None


def _resolve_model_path() -> str:
    """解析意图模型路径：配置中的相对路径基于项目根目录"""
    cfg_path = (
        getattr(config_manager.config.intent_classification, "model_path", "") or ""
    ).strip()
    if not cfg_path:
        return intent_service.MERGED_PATH
    path = Path(cfg_path)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def create_intent_classifier():
    """
    构建意图分类节点（Runnable），进程内只构建一次。

    三层策略：
    1. 规则层：问候/感谢等明显闲聊直接判定，零成本
    2. 模型层：调用 Intent_classification.py 的本地意图分类模型（带最近几轮历史）
    3. 兜底：模型未就绪/异常时默认走 RAG（宁多检索，不出错）
    """
    global _classifier_cache, _model_usable, _model_path, _model_device

    if _classifier_cache is not None:
        return _classifier_cache

    with _classifier_lock:
        if _classifier_cache is not None:
            return _classifier_cache

        model_path = _resolve_model_path()
        device = (
            getattr(config_manager.config.intent_classification, "device", "cpu") or "cpu"
        )
        cfg_enabled = getattr(config_manager.config.intent_classification, "enabled", True)

        if not cfg_enabled:
            # 未启用：完全不加载模型，请求直接走 RAG 分支
            _model_usable = False
            print("【意图路由】意图模型已关闭（intent_classification.enabled=False），全部走 RAG", flush=True)
        else:
            # 预加载模型（懒加载也行，这里提前暴露路径/显存问题）
            try:
                intent_service.load_model(model_path=model_path, device=device)
                _model_usable = True
            except Exception as e:
                _model_usable = False
                print(
                    f"【意图路由】意图模型不可用，后续请求直接走 RAG 兜底 | path: {model_path} | err: {e}",
                    flush=True,
                )

        _model_path = model_path
        _model_device = device
        _classifier_cache = RunnableLambda(_classify)

    return _classifier_cache


def _classify(state: Dict[str, Any]) -> Dict[str, Any]:
    """意图分类：规则层 → 模型层 → 兜底 RAG"""
    global _model_usable

    text = str(state.get("input", "")).strip()
    has_history = bool(state.get("chat_history"))

    # 1. 规则层
    rule_result = _rule_classify(text, has_history=has_history)
    if rule_result is not None:
        print(f"【意图路由】规则命中 → {rule_result} | input: {text[:50]}", flush=True)
        return {**state, "intent": rule_result}

    # 2. 模型层：本地微调模型分类（带最近 4 条历史，解决"那第二个呢"这类指代问题）
    if _model_usable:
        try:
            recent_history = state.get("chat_history", [])[-RECENT_HISTORY_SIZE:]
            raw, intent = intent_service.predict(
                text,
                recent_history,
                model_path=_model_path,
                device=_model_device,
            )
            if intent not in (INTENT_CHAT, INTENT_RAG):
                intent = INTENT_RAG
            print(
                f"【意图路由】本地模型判定 → {intent} | 原始输出: {raw[:50]} | input: {text[:50]}",
                flush=True,
            )
            return {**state, "intent": intent}
        except Exception as e:
            # 加载一次失败后本次及后续都走兜底，避免每个请求都卡在加载模型上
            _model_usable = False
            print(
                f"【意图路由】意图模型不可用，后续请求直接走 RAG 兜底 | input: {text[:50]} | err: {e}",
                flush=True,
            )

    # 3. 兜底：模型不可用时默认走 RAG
    return {**state, "intent": INTENT_RAG}
