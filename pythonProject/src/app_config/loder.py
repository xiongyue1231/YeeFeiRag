"""
配置加载器（薄封装）

配置的唯一来源是 src/app_config/models.py 中的 Pydantic 模型默认值，
本模块不再解析 configs.yaml，只负责对外提供进程内唯一的配置实例（兼容旧调用方式）。
"""
from typing import Any, Dict, Optional, Union

from .models import AppConfig, build_config, get_config, set_config


class ConfigLoader:
    """配置加载器：读取 models.py 定义的配置单例"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 已废弃，仅为兼容旧调用保留。配置统一由 models.py 提供，传入路径会被忽略。
        """
        if config_path:
            print(
                f"[ConfigLoader] 配置已统一由 src/app_config/models.py 提供，忽略传入的配置文件: {config_path}",
                flush=True,
            )
        self._app_config: AppConfig = get_config()

    @property
    def config(self) -> AppConfig:
        return self._app_config

    @property
    def as_dict(self) -> dict:
        """返回当前配置的 dict 形式"""
        return self._app_config.model_dump()

    # -------------------------------------------------------------------------
    # 公共方法：change（仅内存生效，不再写回 YAML）
    # -------------------------------------------------------------------------
    def change(
            self,
            updates: Union[dict, list],
            save: bool = False,
            save_path: str = ""
    ) -> AppConfig:
        """
        在默认配置基础上覆盖部分配置项。

        Args:
            updates: 配置覆盖项，如 {"rag": {"temperature": 0.5}}
            save / save_path: 已废弃，保留仅为兼容旧调用
        """
        flat = dict(updates) if not isinstance(updates, dict) else updates
        new_config = build_config(flat)
        self._app_config = set_config(new_config)
        return self._app_config


# 便捷入口：其它模块可直接 `from src.app_config.loder import configs` 使用
config: AppConfig = get_config()
