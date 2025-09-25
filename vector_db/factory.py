from typing import Optional, Dict
from .base import VectorDBInterface
from config.settings import settings
import threading


class _VectorDBFactory:
    """向量数据库工厂类 - 内部实现"""

    def __init__(self):
        self._clients: Dict[str, VectorDBInterface] = {}
        self._lock = threading.Lock()

    def create_client(self, db_type: Optional[str] = None) -> VectorDBInterface:
        """根据配置创建向量数据库客户端"""
        db_type = db_type or settings.vector_db_type.lower()

        # 检查缓存
        if db_type in self._clients:
            print(f"🔄 复用已缓存的 {db_type} 客户端")
            return self._clients[db_type]

        # 线程安全地创建新客户端
        with self._lock:
            # 双重检查
            if db_type in self._clients:
                return self._clients[db_type]

            # 创建新客户端
            client = self._create_new_client(db_type)
            self._clients[db_type] = client
            return client

    def _create_new_client(self, db_type: str) -> VectorDBInterface:
        """创建新的客户端实例"""
        if db_type == 'chroma':
            try:
                from .chroma_client import ChromaDBClient
                print(f"🆕 创建新的 ChromaDB 客户端")
                return ChromaDBClient()
            except ImportError as e:
                raise ImportError(f"使用ChromaDB需要安装相关依赖: pip install chromadb. 错误: {e}")

        elif db_type == 'milvus':
            try:
                from .milvus_client import MilvusDBClient
                print(f"🆕 创建新的 Milvus 客户端")
                return MilvusDBClient()
            except ImportError as e:
                raise ImportError(f"使用Milvus需要安装相关依赖: pip install pymilvus. 错误: {e}")
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")


# 模块级单例实例
_factory_instance = None
_factory_lock = threading.Lock()


class VectorDBFactory:
    """VectorDBFactory 的代理类，确保始终返回同一个工厂实例"""

    def __new__(cls):
        global _factory_instance
        if _factory_instance is None:
            with _factory_lock:
                if _factory_instance is None:
                    _factory_instance = _VectorDBFactory()
                    print("🔄 VectorDBFactory 单例已初始化")
        return _factory_instance

    @staticmethod
    def create_client(db_type: Optional[str] = None) -> VectorDBInterface:
        """静态方法保持原有调用方式"""
        factory = VectorDBFactory()
        return factory.create_client(db_type)
