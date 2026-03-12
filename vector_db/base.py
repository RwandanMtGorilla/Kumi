from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import pandas as pd


class VectorDBInterface(ABC):
    """向量数据库接口"""

    @abstractmethod
    def create_collection(self, collection_name: str, **kwargs) -> bool:
        """创建集合"""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        pass

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        pass

    @abstractmethod
    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> bool:
        """插入数据"""
        pass

    @abstractmethod
    def query_by_vector(self, collection_name: str, query_vector: List[float],
                        top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """向量查询"""
        pass

    @abstractmethod
    def query_by_ids(self, collection_name: str, ids: List[str],
                     include: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """根据ID查询

        Args:
            collection_name: 集合名称
            ids: 要查询的文档ID列表
            include: 要包含的字段列表，可选值: ['embeddings', 'documents', 'metadatas']
                     默认 None 表示返回全部字段

        Returns:
            数据列表
        """
        pass

    @abstractmethod
    def get_all_data(self, collection_name: str, limit: int = 1000, offset: int = 0,
                     include: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """获取所有数据

        Args:
            collection_name: 集合名称
            limit: 最大返回数量
            offset: 起始偏移量，用于分批读取
            include: 要包含的字段列表，可选值: ['embeddings', 'documents', 'metadatas']
                     默认 None 表示返回全部字段

        Returns:
            数据列表
        """
        pass

    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        pass

    @abstractmethod
    def get_collection_fields(self, collection_name: str) -> List[str]:
        """获取集合字段"""
        pass

    @abstractmethod
    def update_data(self, collection_name: str, ids: List[str],
                    embeddings: Optional[List[List[float]]] = None,
                    documents: Optional[List[str]] = None,
                    metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """更新数据

        Args:
            collection_name: 集合名称
            ids: 要更新的文档ID列表
            embeddings: 新的向量嵌入（可选）
            documents: 新的文档内容（可选）
            metadatas: 新的元数据（可选）

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def delete_by_ids(self, collection_name: str, ids: List[str]) -> bool:
        """根据ID删除数据

        Args:
            collection_name: 集合名称
            ids: 要删除的文档ID列表

        Returns:
            是否删除成功
        """
        pass
