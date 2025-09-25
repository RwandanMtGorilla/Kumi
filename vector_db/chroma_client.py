import chromadb
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import numpy as np
from .base import VectorDBInterface
from .embedding_client import QwenEmbeddingAPI
from config.settings import settings


class ChromaDBClient(VectorDBInterface):
    """ChromaDB客户端实现"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or settings.chroma_host
        self.port = port or settings.chroma_port

        # 连接到ChromaDB
        self.client = chromadb.HttpClient(host=self.host, port=self.port)

        # 初始化embedding客户端
        self.embedding_api = QwenEmbeddingAPI()

        print(f"✅ ChromaDB客户端初始化完成")
        print(f"   地址: {self.host}:{self.port}")
        print(f"   Embedding维度: {self.embedding_api.embedding_dim}")

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1 = np.array(vec1, dtype=np.float32)
            vec2 = np.array(vec2, dtype=np.float32)

            # 计算点积
            dot_product = np.dot(vec1, vec2)

            # 计算向量范数
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)

            # 避免除零错误
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0

            # 计算余弦相似度
            cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

            # 确保结果在[-1, 1]范围内
            return float(np.clip(cosine_similarity, -1.0, 1.0))

        except Exception as e:
            print(f"❌ 计算余弦相似度失败: {e}")
            return 0.0

    def create_collection(self, collection_name: str, **kwargs) -> bool:
        """创建集合"""
        try:
            # 检查集合是否已存在
            try:
                self.client.delete_collection(name=collection_name)
                print(f"⚠️ Collection '{collection_name}' 已存在，已删除")
            except:
                pass

            # 创建新集合
            collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": f"Collection created at {datetime.now()}",
                    "embedding_dimension": self.embedding_api.embedding_dim,
                    **kwargs.get('metadata', {})
                }
            )

            print(f"✅ Collection '{collection_name}' 创建成功")
            return True

        except Exception as e:
            print(f"❌ 创建Collection失败: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(name=collection_name)
            print(f"✅ Collection '{collection_name}' 删除成功")
            return True
        except Exception as e:
            print(f"❌ 删除Collection失败: {e}")
            return False

    def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            return collection_names
        except Exception as e:
            print(f"❌ 获取Collections列表失败: {e}")
            return []

    def has_collection(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            self.client.get_collection(name=collection_name)
            return True
        except:
            return False

    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> bool:
        """插入数据"""
        try:
            collection = self.client.get_collection(name=collection_name)

            # 准备数据
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for item in data:
                # 生成ID
                item_id = item.get('id', str(uuid.uuid4()))
                ids.append(str(item_id))

                # 获取向量
                if 'embedding' in item:
                    embeddings.append(item['embedding'])
                elif 'dense_vector' in item:
                    embeddings.append(item['dense_vector'])
                else:
                    raise ValueError("数据中缺少embedding或dense_vector字段")

                # 获取文档文本
                if 'document' in item:
                    documents.append(item['document'])
                else:
                    # 如果没有document字段，使用所有文本字段组合
                    doc_parts = []
                    for key, value in item.items():
                        if key not in ['id', 'embedding', 'dense_vector'] and isinstance(value, str):
                            doc_parts.append(f"{key}: {value}")
                    documents.append(" | ".join(doc_parts))

                # 获取元数据
                metadata = {}
                for key, value in item.items():
                    if key not in ['id', 'embedding', 'dense_vector', 'document']:
                        # ChromaDB元数据只支持基本类型
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)
                metadatas.append(metadata)

            # 插入数据
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            print(f"✅ 成功插入 {len(data)} 条数据到 '{collection_name}'")
            return True

        except Exception as e:
            print(f"❌ 插入数据失败: {e}")
            return False

    def query_by_vector(self, collection_name: str, query_vector: List[float],
                        top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """向量查询"""
        try:
            collection = self.client.get_collection(name=collection_name)

            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances', 'embeddings']
            )

            # 格式化结果
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': "",
                        'metadata': {},
                        'distance': 0.0,
                        'similarity': 0.0,
                        'cosine_similarity': 0.0
                    }

                    # 安全地处理文档
                    if (results.get('documents') is not None and
                            len(results['documents']) > 0 and
                            i < len(results['documents'][0])):
                        doc = results['documents'][0][i]
                        result['document'] = doc if doc is not None else ""

                    # 安全地处理元数据
                    if (results.get('metadatas') is not None and
                            len(results['metadatas']) > 0 and
                            i < len(results['metadatas'][0])):
                        metadata = results['metadatas'][0][i]
                        result['metadata'] = metadata if metadata is not None else {}

                    # 安全地处理距离
                    if (results.get('distances') is not None and
                            len(results['distances']) > 0 and
                            i < len(results['distances'][0])):
                        distance = results['distances'][0][i]
                        result['distance'] = float(distance) if distance is not None else 0.0
                        # ChromaDB的距离是基于1-cosine_similarity的，所以相似度为1-distance
                        result['similarity'] = 1 - result['distance']

                    # 计算正确的余弦相似度
                    if (results.get('embeddings') is not None and
                            len(results['embeddings']) > 0 and
                            i < len(results['embeddings'][0])):
                        doc_embedding = results['embeddings'][0][i]
                        if doc_embedding is not None:
                            result['cosine_similarity'] = self._calculate_cosine_similarity(
                                query_vector, doc_embedding
                            )

                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ 向量查询失败: {e}")
            return []

    def query_by_ids(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        """根据ID查询"""
        try:
            collection = self.client.get_collection(name=collection_name)

            results = collection.get(
                ids=ids,
                include=['documents', 'metadatas', 'embeddings']
            )

            # 格式化结果
            formatted_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    result = {
                        'id': results['ids'][i],
                        'document': results['documents'][i] if results['documents'] else "",
                        'metadata': results['metadatas'][i] if results['metadatas'] else {},
                        'embedding': results['embeddings'][i] if results['embeddings'] else []
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ ID查询失败: {e}")
            return []

    def get_all_data(self, collection_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取所有数据"""
        try:
            collection = self.client.get_collection(name=collection_name)

            # ChromaDB没有直接的分页查询，需要通过其他方式获取
            results = collection.get(
                limit=limit,
                include=['documents', 'metadatas', 'embeddings']
            )

            # 格式化结果
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'])):
                    result = {
                        'id': results['ids'][i],
                        'document': "",
                        'embedding': [],
                        'dense_vector': []
                    }

                    # 安全地处理文档
                    if results.get('documents') is not None and i < len(results['documents']):
                        result['document'] = results['documents'][i] if results['documents'][i] is not None else ""

                    # 安全地处理嵌入向量
                    if results.get('embeddings') is not None and i < len(results['embeddings']):
                        embedding = results['embeddings'][i]
                        if embedding is not None:
                            result['embedding'] = embedding
                            result['dense_vector'] = embedding

                    # 安全地处理元数据字段
                    if results.get('metadatas') is not None and i < len(results['metadatas']):
                        metadata = results['metadatas'][i]
                        if metadata is not None and isinstance(metadata, dict):
                            for key, value in metadata.items():
                                # 避免覆盖系统字段
                                if key not in ['id', 'document', 'embedding', 'dense_vector']:
                                    result[key] = value

                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"❌ 获取所有数据失败: {e}")
            return []

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            collection = self.client.get_collection(name=collection_name)

            # 获取集合信息
            count_result = collection.count()

            return {
                "row_count": count_result,
                "name": collection_name,
                "metadata": collection.metadata or {}
            }

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {"error": str(e)}

    def get_collection_fields(self, collection_name: str) -> List[str]:
        """获取集合字段"""
        try:
            # 通过获取一条数据来推断字段
            sample_data = self.get_all_data(collection_name, limit=1)
            if sample_data:
                fields = list(sample_data[0].keys())
                # 过滤掉系统字段
                filtered_fields = [f for f in fields if f not in ['id', 'embedding', 'dense_vector']]
                return filtered_fields
            else:
                return []

        except Exception as e:
            print(f"❌ 获取字段失败: {e}")
            return []
