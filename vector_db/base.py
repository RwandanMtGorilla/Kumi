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
    def query_by_ids(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID查询"""
        pass

    @abstractmethod
    def get_all_data(self, collection_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取所有数据"""
        pass

    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        pass

    @abstractmethod
    def get_collection_fields(self, collection_name: str) -> List[str]:
        """获取集合字段"""
        pass
