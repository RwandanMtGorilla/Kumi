"""
Embedding服务 - 处理多供应商embedding逻辑
"""
from typing import List, Optional, Callable
from config.settings import settings
from config.embedding_config import EmbeddingConfig
from vector_db.embedding_client import _OpenAIEmbeddingAPI
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding服务,支持多供应商配置"""

    def __init__(self):
        """初始化Embedding服务"""
        self.config: Optional[EmbeddingConfig] = None
        self._clients_cache = {}  # 缓存客户端实例

    def _ensure_config(self):
        """确保配置已加载"""
        if self.config is None:
            self.config = settings.get_embedding_config()

    def get_client(self, provider_name: str = None, model_name: str = None) -> _OpenAIEmbeddingAPI:
        """
        获取embedding客户端

        Args:
            provider_name: 供应商名称,None表示使用默认
            model_name: 模型名称,None表示使用默认

        Returns:
            _OpenAIEmbeddingAPI实例
        """
        self._ensure_config()

        # 使用默认值
        if provider_name is None or model_name is None:
            provider_name, model_name = self.config.get_default_model()

        # 检查缓存
        cache_key = f"{provider_name},{model_name}"
        if cache_key in self._clients_cache:
            return self._clients_cache[cache_key]

        # 获取模型信息
        model_info = self.config.get_model_info(provider_name, model_name)
        if not model_info:
            raise ValueError(f"未找到模型配置: provider={provider_name}, model={model_name}")

        # 创建客户端
        client = _OpenAIEmbeddingAPI(
            base_url=model_info['api_base_url'],
            token=model_info['api_key'],
            model=model_info['model']
        )

        # 缓存
        self._clients_cache[cache_key] = client
        logger.info(f"创建embedding客户端: {cache_key}")

        return client

    def encode_texts(
            self,
            texts: List[str],
            provider_name: str = None,
            model_name: str = None,
            batch_size: int = 20
    ) -> List[List[float]]:
        """
        批量编码文本为向量

        Args:
            texts: 文本列表
            provider_name: 供应商名称
            model_name: 模型名称
            batch_size: 批处理大小

        Returns:
            向量列表
        """
        client = self.get_client(provider_name, model_name)
        client.set_batch_size(batch_size)
        return client.encode_texts(texts)

    def encode_texts_with_progress(
            self,
            texts: List[str],
            progress_callback: Optional[Callable[[int, int, str], None]],
            provider_name: str = None,
            model_name: str = None,
            batch_size: int = 20
    ) -> List[List[float]]:
        """
        批量编码文本为向量(带进度回调)

        Args:
            texts: 文本列表
            progress_callback: 进度回调函数
            provider_name: 供应商名称
            model_name: 模型名称
            batch_size: 批处理大小

        Returns:
            向量列表
        """
        client = self.get_client(provider_name, model_name)
        client.set_batch_size(batch_size)
        return client.encode_texts_with_progress(texts, progress_callback)

    def test_connection(
            self,
            provider_name: str = None,
            model_name: str = None
    ) -> dict:
        """
        测试embedding API连接

        Args:
            provider_name: 供应商名称
            model_name: 模型名称

        Returns:
            {"success": bool, "message": str, "dimension": int}
        """
        try:
            logger.info(f"开始测试embedding连接: provider={provider_name}, model={model_name}")
            client = self.get_client(provider_name, model_name)
            result = client.test_connection()

            if result["success"]:
                logger.info(f"Embedding连接测试成功: provider={provider_name}, model={model_name}, dimension={result['dimension']}")
            else:
                logger.warning(f"Embedding连接测试失败: provider={provider_name}, model={model_name}, reason={result['message']}")

            return result
        except Exception as e:
            logger.error(f"测试连接时发生异常: provider={provider_name}, model={model_name}, error={e}", exc_info=True)
            return {
                "success": False,
                "message": f"测试连接失败: {str(e)}",
                "dimension": None
            }

    def get_available_models(self) -> List[dict]:
        """
        获取所有可用模型

        Returns:
            [
                {
                    "provider": "openrouter",
                    "model": "google/gemini-embedding-001",
                    "display_name": "openrouter,google/gemini-embedding-001"
                },
                ...
            ]
        """
        self._ensure_config()
        return self.config.get_all_models()

    def get_default_model(self) -> dict:
        """
        获取默认模型信息

        Returns:
            {
                "provider": "openrouter",
                "model": "google/gemini-embedding-001",
                "display_name": "openrouter,google/gemini-embedding-001"
            }
        """
        self._ensure_config()
        provider, model = self.config.get_default_model()
        return {
            "provider": provider,
            "model": model,
            "display_name": f"{provider},{model}"
        }

    def parse_model_identifier(self, model_identifier: str) -> tuple:
        """
        解析模型标识符

        Args:
            model_identifier: "provider,model" 格式

        Returns:
            (provider_name, model_name)
        """
        self._ensure_config()
        return self.config.parse_model_identifier(model_identifier)

    def generate_model_abbreviation(self, model_name: str) -> str:
        """
        生成模型缩写用于collection名称

        Args:
            model_name: 模型名称

        Returns:
            缩写字符串,如 "gem-1", "Qwe-6"
        """
        self._ensure_config()
        return EmbeddingConfig.generate_model_abbreviation(model_name)
