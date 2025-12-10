from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any
import pandas as pd
from .base import VectorDBInterface
from .embedding_client import OpenAIEmbeddingAPI
from config.settings import settings


class MilvusDBClient(VectorDBInterface):
    """Milvus客户端实现"""

    def __init__(self, host: str = None, port: int = None, user: str = None, password: str = None):
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.user = user or settings.milvus_user
        self.password = password or settings.milvus_password

        # 连接到Milvus
        uri = f"http://{self.host}:{self.port}"
        self.client = MilvusClient(uri=uri, token=f"{self.user}:{self.password}")

        # 初始化embedding客户端
        self.embedding_api = OpenAIEmbeddingAPI()

        print(f"✅ Milvus客户端初始化完成")
        print(f"   地址: {uri}")
        print(f"   Embedding维度: {self.embedding_api.embedding_dim}")

    # 这里可以实现与原来Milvus相同的方法
    # 为了简洁，我只列出接口，具体实现可以参考原代码

    def create_collection(self, collection_name: str, **kwargs) -> bool:
        # 实现Milvus的集合创建逻辑
        pass

    def delete_collection(self, collection_name: str) -> bool:
        # 实现Milvus的集合删除逻辑
        pass

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return collections
        except Exception as e:
            print(f"❌ 获取Collections列表失败: {e}")
            return []

