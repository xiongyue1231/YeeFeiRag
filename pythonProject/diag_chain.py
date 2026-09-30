# -*- coding: utf-8 -*-
"""RAG 链路逐段诊断：Redis / vLLM LLM / Milvus"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("== 1. Redis 测试 ==", flush=True)
from src.app_config.loder import ConfigLoader
from redis import Redis

cfg = ConfigLoader()

r = Redis(host=cfg.config.redis.host, port=cfg.config.redis.port, db=0,
          socket_timeout=5, socket_connect_timeout=5)
t = time.time()
print(f"PING: {r.ping()}  耗时 {time.time() - t:.2f}s", flush=True)

key = "chat_history:user_123"
t = time.time()
key_type = r.type(key)
print(f"KEY {key} TYPE: {key_type}  耗时 {time.time() - t:.2f}s", flush=True)
if key_type in (b"hash", "hash"):
    t = time.time()
    n = r.hlen(key)
    print(f"HLEN: {n}  耗时 {time.time() - t:.2f}s", flush=True)
elif key_type in (b"string", "string"):
    print("!! 该 key 是 string 类型（旧版历史写入的 JSON），"
          "新版用 hgetall 读取会报 WRONGTYPE，建议删掉该 key", flush=True)

print("\n== 2. vLLM LLM 测试 ==", flush=True)
from src.core.utils import create_llm_langchain

llm = create_llm_langchain(cfg.config.rag)
print(f"base_url={llm.openai_api_base}  model={llm.model_name}", flush=True)
t = time.time()
try:
    resp = llm.invoke("回复：ok")
    print(f"LLM 返回: {resp.content!r}  耗时 {time.time() - t:.2f}s", flush=True)
except Exception as e:
    print(f"LLM 调用失败({time.time() - t:.2f}s): {type(e).__name__}: {e}", flush=True)

print("\n== 3. Milvus 测试 ==", flush=True)
from pymilvus import MilvusClient

t = time.time()
mc = MilvusClient(host=cfg.config.milvus.host, port=cfg.config.milvus.port)
print(f"collections: {mc.list_collections()}  耗时 {time.time() - t:.2f}s", flush=True)

print("\n诊断结束", flush=True)
