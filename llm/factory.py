# llm/factory.py

from typing import Optional, Dict, Any
from .base import LLMClientInterface
from .openai_client import OpenAIClient
from .exceptions import LLMConfigException
from config.settings import settings


class LLMFactory:
    """LLM客户端工厂类"""

    _clients: Dict[str, type] = {
        'openai': OpenAIClient,
    }

    @classmethod
    def create_client(
            self,
            provider: Optional[str] = None,
            **kwargs
    ) -> LLMClientInterface:
        """
        创建LLM客户端

        Args:
            provider: LLM提供商 ('openai')
            **kwargs: 客户端配置参数

        Returns:
            LLMClientInterface: LLM客户端实例
        """
        provider = provider or settings.DEFAULT_LLM_PROVIDER

        if provider not in self._clients:
            available_providers = ', '.join(self._clients.keys())
            raise LLMConfigException(
                f"Unsupported LLM provider: {provider}. "
                f"Available providers: {available_providers}"
            )

        client_class = self._clients[provider]

        try:
            client = client_class(**kwargs)
            print(f"✅ LLM客户端创建成功: {provider}")
            return client
        except Exception as e:
            raise LLMConfigException(f"Failed to create {provider} client: {e}")

    @classmethod
    def get_available_providers(cls) -> list:
        """获取可用的LLM提供商列表"""
        return list(cls._clients.keys())

    @classmethod
    def register_client(cls, provider: str, client_class: type):
        """注册新的LLM客户端类"""
        cls._clients[provider] = client_class
