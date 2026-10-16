"""
意图分类模块（LoRA 微调后的 Qwen）

两种用法：
1. 独立服务：python Intent_classification.py  ->  FastAPI: POST /predict
2. 进程内调用：from Intent_classification import predict / classify
   （router.py 的 LLM 层直接调用 classify()，不再走大模型接口）
"""
import os
import threading
import time
from typing import Optional, Sequence, Union

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 合并后的 LoRA 模型目录（可用环境变量 INTENT_MODEL_PATH 覆盖）
MERGED_PATH = os.environ.get(
    "INTENT_MODEL_PATH", os.path.join(BASE_DIR, "src", "lora", "qwen_merged")
)
# 推理设备（可用环境变量 INTENT_MODEL_DEVICE 覆盖，无 GPU 时自动回落 CPU）
DEFAULT_DEVICE = os.environ.get("INTENT_MODEL_DEVICE", "cpu")
# 加载失败后的冷却时间（秒）：显存/页面文件不足这类问题短期内不会自行恢复，
# 冷却期内直接判定不可用，避免每个请求都卡在"加载 1.5B 模型"上把整机拖垮
LOAD_FAIL_COOLDOWN = float(os.environ.get("INTENT_LOAD_FAIL_COOLDOWN", "300"))
# 单次加载超时（秒）：Windows 页面文件不足时 from_pretrained 的 mmap 可能长时间不返回，
# 超时后放弃本次加载，避免把请求/启动彻底卡死
LOAD_TIMEOUT = float(os.environ.get("INTENT_LOAD_TIMEOUT", "60"))

INSTRUCTION = (
    "你是一个对话意图分类器。判断用户输入的意图类别，只输出chat或者rag：\n"
    "- chat：闲聊、问候、感谢、道别、情感表达、天气如何等与知识查询无关的对话\n"
    "  结合对话历史判断指代不明确的输入（如\"那第二个呢\"）\n"
    "1. 只要涉及事实性知识诉求，一律归为 rag\n"
    "2. 无法确定时输出 rag"
)

MAX_NEW_TOKENS = 10

# 全局容器，lifespan / load_model 里填充
ml = {}
_ml_lock = threading.Lock()
# 最近一次加载失败的时间戳与原因（用于冷却，失败后不再逐请求重试）
_fail_at = 0.0
_fail_reason = ""

# 历史消息支持两种形态：dict（{'role','content'}）或 LangChain BaseMessage
HistoryItem = Union[dict, object]


def _load(model_path: str, device_: str):
    """真正执行模型加载（在独立线程里跑，配合外层超时控制）"""
    dtype = torch.float16 if device_.startswith("cuda") else torch.float32
    print(f"[IntentClassifier] loading: {model_path} (device={device_})", flush=True)
    if device_.startswith("cuda"):
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="cuda", trust_remote_code=True
        )
    else:
        # low_cpu_mem_usage：避免先随机初始化再拷权重，把峰值内存直接翻倍
        m = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
        ).to("cpu")
    m.eval()
    return m


def _load_with_timeout(model_path: str, device_: str):
    """
    带超时的模型加载。

    Windows 页面文件不足时 from_pretrained 的 mmap 可能一直不返回（不报错也不完成），
    这里用守护线程加载 + join 超时，超时即放弃，避免卡死请求或启动流程。
    """
    holder: dict = {}

    def _worker():
        try:
            holder["model"] = _load(model_path, device_)
        except BaseException as e:  # noqa: BLE001（需要把线程内异常带回主线程）
            holder["error"] = e

    t = threading.Thread(target=_worker, daemon=True, name="intent-model-loader")
    t.start()
    t.join(LOAD_TIMEOUT)

    if t.is_alive():
        _mark_failed(TimeoutError(f"加载超过 {LOAD_TIMEOUT}s 未完成"))
        raise RuntimeError(
            f"[IntentClassifier] 加载意图模型超过 {LOAD_TIMEOUT}s（页面文件/内存不足时会卡在 mmap），已放弃本次加载"
        )
    if "error" in holder:
        raise holder["error"]
    return holder["model"]


def _mark_failed(err: Exception) -> None:
    """记录加载失败时间戳，进入冷却期（失败原因如显存/页面文件不足短期内不会恢复）"""
    global _fail_at, _fail_reason
    _fail_at = time.time()
    _fail_reason = f"{type(err).__name__}: {str(err)[:200]}"


def is_ready() -> bool:
    """意图模型是否已成功加载（未加载 / 加载失败时为 False）"""
    return "model" in ml


def load_model(model_path: Optional[str] = None, device: Optional[str] = None):
    """
    加载意图分类模型（进程内只加载一次，线程安全；失败后进入冷却期不再重试）。

    :param model_path: 模型目录，默认 MERGED_PATH；已加载且路径一致时直接复用
    :param device: cuda / cpu，默认 DEFAULT_DEVICE（无 GPU 自动回落 cpu）
    :return: (tokenizer, model)
    :raise RuntimeError: 冷却期内或加载失败时抛出，调用方按"不可用"兜底
    """
    global _fail_at

    with _ml_lock:
        # 已加载：未指定路径或路径一致时直接复用
        if "model" in ml and (model_path is None or ml.get("path") == model_path):
            return ml["tokenizer"], ml["model"]

        # 冷却期内直接失败：不要在请求链路里反复加载大模型（这一步曾经把整机拖到超时）
        if _fail_at:
            waited = time.time() - _fail_at
            if waited < LOAD_FAIL_COOLDOWN:
                raise RuntimeError(
                    f"意图模型加载处于冷却期（剩余 {int(LOAD_FAIL_COOLDOWN - waited)}s）"
                    f" | 上次失败原因: {_fail_reason}"
                )

        model_path = model_path or MERGED_PATH
        device = device or DEFAULT_DEVICE

        if not os.path.exists(model_path):
            _mark_failed(FileNotFoundError(f"意图模型路径不存在: {model_path}"))
            raise FileNotFoundError(
                f"[IntentClassifier] 意图模型路径不存在: {model_path}\n"
                f"请先执行 LoRA 合并，或修改 src/app_config/models.py 的 intent_classification.model_path"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[IntentClassifier] CUDA 不可用，自动回落到 CPU", flush=True)
            device = "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        try:
            model = _load_with_timeout(model_path, device)
        except Exception as e:
            if not device.startswith("cuda"):
                _mark_failed(e)
                raise
            # 显存不足最常见：vLLM 已占满显卡时，回落到 CPU，避免拖垮同进程的其它模型
            print(
                f"[IntentClassifier] GPU 加载失败({type(e).__name__}: {str(e)[:120]})，自动回落到 CPU",
                flush=True,
            )
            torch.cuda.empty_cache()
            device = "cpu"
            try:
                model = _load_with_timeout(model_path, device)
            except Exception as e2:
                _mark_failed(e2)
                raise

        ml.clear()
        ml["tokenizer"] = tokenizer
        ml["model"] = model
        ml["path"] = model_path
        _fail_at = 0.0
        print("[IntentClassifier] 意图模型加载完成", flush=True)
        return tokenizer, model


def unload_model():
    """释放模型显存/内存"""
    with _ml_lock:
        ml.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_intent(raw: str) -> str:
    """把模型原始输出归一化为 chat / rag（rag 优先：宁多检索，不出错）"""
    answer = (raw or "").strip().lower()
    if "rag" in answer:
        return "rag"
    if "chat" in answer:
        return "chat"
    return "rag"


def _to_message_dicts(history: Optional[Sequence[HistoryItem]]) -> list:
    """兼容 dict 历史与 LangChain BaseMessage 历史，统一转成 role/content 列表"""
    messages: list = []
    for m in history or []:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role = getattr(m, "type", None)  # BaseMessage: human / ai / system
            content = getattr(m, "content", None)
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": str(content)})
    return messages


def _build_messages(user_input: str, history: Optional[Sequence[HistoryItem]]) -> list:
    messages: list = _to_message_dicts(history)
    messages.append(
        {"role": "user", "content": f"{INSTRUCTION}\n\n{user_input}"}
    )
    return messages


def predict(
    user_input: str,
    history: Optional[Sequence[HistoryItem]] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple:
    """
    执行意图分类。

    :return: (raw, intent)；raw 为模型原始输出，intent 为归一化后的 "chat" / "rag"
    """
    tokenizer, model = load_model(model_path, device)

    messages = _build_messages(user_input, history)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return raw, _normalize_intent(raw)


def classify(
    user_input: str,
    history: Optional[Sequence[HistoryItem]] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """对外暴露的分类方法：直接返回 "chat" 或 "rag"（router.py 调用入口）"""
    _, intent = predict(user_input, history, model_path, device)
    return intent


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


app = FastAPI(title="Intent Classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="用户输入")
    history: list[dict] | None = Field(
        default=None,
        description="可选对话历史，格式 [{'role':'user','content':'...'}, ...]",
    )


class PredictResponse(BaseModel):
    intent: str
    raw: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="model not ready")
    raw, intent = predict(req.user_input, req.history)
    return PredictResponse(intent=intent, raw=raw)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Intent_classification:app", host="0.0.0.0", port=8001, reload=False)
