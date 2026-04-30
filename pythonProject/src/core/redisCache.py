import json
import redis
from typing import List, Tuple
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict, HumanMessage, AIMessage
from src.app_config.loder import ConfigLoader

config_manager = ConfigLoader()


class RedisChatMessageHistory(BaseChatMessageHistory):
    """
    使用 Redis 存储聊天历史的消息历史类。
    每条历史记录作为一个 JSON 字符串存储在 Redis 的字符串键中，键名为 f"chat_history:{session_id}"
    """

    def __init__(self, session_id: str, redis_client: redis.Redis, key_prefix: str = "chat_history:",
                 ttl: int = config_manager.config.redis.SESSION_TTL):
        self.session_id = session_id
        self.redis_client = redis_client
        self.key = f"{key_prefix}{session_id}"
        self.seq_key = f"{self.key}:seq"  # 计数器键
        self.ttl = ttl

    def _get_next_seq_id(self) -> int:
        """通过 Redis INCR 获取下一个序列号，并设置 TTL（若启用）"""
        next_id = self.redis_client.incr(self.seq_key)
        if self.ttl is not None:
            self.redis_client.expire(self.seq_key, self.ttl)
        return next_id

    @property
    def messages(self) -> List[BaseMessage]:
        """从 Redis 读取并返回消息列表"""

        # print(f" type: {self.redis_client.type(self.key)}")
        entries = self._get_all_entries()
        # 从每个条目中提取 message 字段
        msg_dicts = [entry["message"] for entry in entries]
        return messages_from_dict(msg_dicts)

    def get_messages_with_seq_id(self) -> List[Tuple[int, BaseMessage]]:
        """返回带有序号的消息列表，格式：[(seq_id, message), ...]"""
        entries = self._get_all_entries()
        result = []
        for entry in entries:
            msg = messages_from_dict([entry["message"]])[0]
            result.append((entry["seq_id"], msg))
        return result

    def _get_all_entries(self):
        """内部方法：读取完整的带 seq_id 的条目列表"""
        data = self.redis_client.get(self.key)
        if not data:
            return []
        return json.loads(data)

    def add_message(self, message: BaseMessage) -> None:
        seq_id = self._get_next_seq_id()
        msg_dict = message_to_dict(message)
        entry = {"seq_id": seq_id, "message": msg_dict}
        # 读取现有列表，追加新条目
        entries = self._get_all_entries()
        entries.append(entry)
        # 写回 Redis 并刷新 TTL
        self.redis_client.set(self.key, json.dumps(entries))
        if self.ttl is not None:
            self.redis_client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """清除当前会话的所有历史消息"""
        self.redis_client.delete(self.key)


# ---------- 初始化 Redis 客户端（根据实际环境配置连接参数） ----------
redis_client = redis.Redis(
    host=config_manager.config.redis.host,  # Redis 主机地址
    port=config_manager.config.redis.port,  # Redis 端口
    db=0,  # 使用的数据库编号
    decode_responses=True  # 自动将返回的字节串解码为字符串
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """工厂函数：根据 session_id 返回对应的 Redis 聊天历史实例"""
    return RedisChatMessageHistory(session_id=session_id, redis_client=redis_client)


if __name__ == "__main__":
    # 创建一个会话历史对象
    session_id = "user_123"
    session_history = get_session_history(session_id)
    session_history.clear()

    print("=== 添加消息 ===")
    session_history.add_message(HumanMessage(content="这是用户问题？"))
    session_history.add_message(AIMessage(content="这是AI回复。"))

    print("\n=== 仅消息内容（兼容原有逻辑）===")
    for msg in session_history.messages:
        print(f"{msg.type}: {msg.content}")

    print("\n=== 带 seq_id 的消息 ===")
    for seq_id, msg in session_history.get_messages_with_seq_id():
        print(f"seq_id={seq_id}, {msg.type}: {msg.content}")
