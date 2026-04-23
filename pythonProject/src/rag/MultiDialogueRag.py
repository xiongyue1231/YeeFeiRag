import json
import time
from typing import Optional
from redis import Redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()

class RedisChatMessageHistory(BaseChatMessageHistory):
    """基于 Redis 的聊天消息历史，支持 TTL 和会话内自增 seq_id"""

    def __init__(
            self,
            session_id: str,
            redis_client: Redis,
            ttl: Optional[int] = 3600,  # 默认 TTL 1小时
            key_prefix: str = "chat_history"
    ):
        self.session_id = session_id
        self.redis = redis_client
        self.ttl = ttl
        self.key_prefix = key_prefix

        # Redis key: chat_history:{session_id}
        self.history_key = f"{self.key_prefix}:{self.session_id}"
        # 序列号 key: chat_history:{session_id}:seq
        self.seq_key = f"{self.history_key}:seq"

    @property
    def messages(self) -> list[BaseMessage]:
        """获取所有消息，按 seq_id 排序"""
        # 获取所有消息（Hash 结构，field 为 seq_id）
        raw_messages = self.redis.hgetall(self.history_key)
        if not raw_messages:
            return []

        # 按 seq_id 排序并反序列化
        sorted_items = sorted(raw_messages.items(), key=lambda x: int(x[0]))
        message_dicts = [json.loads(msg.decode('utf-8')) for _, msg in sorted_items]

        return messages_from_dict(message_dicts)

    def add_message(self, message: BaseMessage) -> None:
        """添加消息，自动分配自增 seq_id"""
        # 获取下一个 seq_id（原子自增）
        seq_id = self.redis.incr(self.seq_key)

        # 序列化消息
        message_dict = messages_to_dict([message])[0]
        message_data = json.dumps(message_dict, ensure_ascii=False)

        # 使用 Pipeline 保证原子性
        pipe = self.redis.pipeline()

        # 存储消息到 Hash
        pipe.hset(self.history_key, str(seq_id), message_data)

        # 设置 TTL（如果配置了）
        if self.ttl:
            pipe.expire(self.history_key, self.ttl)
            pipe.expire(self.seq_key, self.ttl)

        pipe.execute()

    def clear(self) -> None:
        """清空会话历史"""
        pipe = self.redis.pipeline()
        pipe.delete(self.history_key)
        pipe.delete(self.seq_key)
        pipe.execute()

    def get_next_seq_id(self) -> int:
        """获取下一个 seq_id（不实际递增）"""
        current = self.redis.get(self.seq_key)
        return int(current.decode('utf-8')) + 1 if current else 1

    def get_message_by_seq(self, seq_id: int) -> Optional[BaseMessage]:
        """通过 seq_id 获取单条消息"""
        raw = self.redis.hget(self.history_key, str(seq_id))
        if not raw:
            return None
        return messages_from_dict([json.loads(raw.decode('utf-8'))])[0]


# 全局 Redis 连接和 store（替代原来的内存 store）
_redis_client: Optional[Redis] = None
_store: dict[str, RedisChatMessageHistory] = {}  # 本地缓存，减少 Redis 连接创建


def init_redis(host: str = config_manager.config.redis.host, port: int = config_manager.config.redis.port, db: int = 0, **kwargs):
    """初始化全局 Redis 连接"""
    global _redis_client
    _redis_client = Redis(host=host, port=port, db=db, decode_responses=False, **kwargs)
    return _redis_client


def get_session_history(
        session_id: str,
        ttl: Optional[int] = 3600,
        redis_client: Optional[Redis] = None
) -> BaseChatMessageHistory:
    """
    获取会话历史（Redis 持久化版本）

    Args:
        session_id: 会话 ID
        ttl: 会话过期时间（秒），默认 3600
        redis_client: 可选的 Redis 客户端，未提供则使用全局客户端

    Returns:
        RedisChatMessageHistory 实例
    """
    global _redis_client

    client = redis_client or _redis_client
    if client is None:
        raise RuntimeError(
            "Redis client not initialized. Call init_redis() first or pass redis_client."
        )

    # 复用已创建的实例（可选优化）
    if session_id not in _store:
        _store[session_id] = RedisChatMessageHistory(
            session_id=session_id,
            redis_client=client,
            ttl=ttl
        )

    return _store[session_id]