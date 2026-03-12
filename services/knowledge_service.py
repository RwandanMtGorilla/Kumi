from typing import List, Dict, Any, Optional
from vector_db.factory import VectorDBFactory
from services.embedding_service import EmbeddingService
from api.models import RetrievalRequest, RetrievalResponse, RetrievalRecord, MetadataFilter, ComparisonOperator
import logging
import asyncio
import pandas as pd
import re
import tempfile
import os
from pathlib import Path
from typing import Union
import traceback
import numpy as np
import time
import threading
import hashlib


logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识库服务"""

    def __init__(self):
        self.vector_client = VectorDBFactory.create_client()
        self.embedding_service = EmbeddingService()

    async def search(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        执行知识库检索
        """
        try:
            # 1. 向量化查询文本
            logger.info(f"正在向量化查询: {request.query}")
            query_embeddings = await self._vectorize_query(request.query)

            # 2. 向量搜索
            logger.info(f"正在搜索向量数据库: {request.knowledge_id}")
            search_results = await self._vector_search(
                collection_name=request.knowledge_id,
                query_vector=query_embeddings[0],  # 取第一个向量
                top_k=request.retrieval_setting.top_k * 2,  # 多取一些，后续过滤
            )

            # 3. 应用元数据筛选
            if request.metadata_condition:
                search_results = self._apply_metadata_filter(
                    search_results,
                    request.metadata_condition
                )

            # 4. 应用分数阈值筛选
            filtered_results = self._apply_score_threshold(
                search_results,
                request.retrieval_setting.score_threshold
            )

            # 5. 限制结果数量
            final_results = filtered_results[:request.retrieval_setting.top_k]

            # 6. 格式化为API响应
            records = self._format_results(final_results)

            return RetrievalResponse(records=records)

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

    async def _vectorize_query(
            self,
            query: str,
            provider_name: str = None,
            model_name: str = None
    ) -> List[List[float]]:
        """
        向量化查询文本

        Args:
            query: 查询文本
            provider_name: embedding供应商名称,None表示使用默认
            model_name: embedding模型名称,None表示使用默认
        """
        try:
            # 使用asyncio在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.embedding_service.encode_texts,
                [query],
                provider_name,
                model_name,
                1  # batch_size=1 for single query
            )
            return embeddings
        except Exception as e:
            logger.error(f"向量化失败: {e}")
            raise

    async def _vector_search(self, collection_name: str, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """执行向量搜索"""
        try:
            # 检查集合是否存在
            if not self.vector_client.has_collection(collection_name):
                raise ValueError(f"知识库 '{collection_name}' 不存在")

            # 执行搜索
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.vector_client.query_by_vector,
                collection_name, query_vector, top_k
            )
            return results
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise

    def _apply_metadata_filter(self, results: List[Dict[str, Any]], metadata_filter: MetadataFilter) -> List[
        Dict[str, Any]]:
        """应用元数据筛选"""
        try:
            filtered_results = []

            for result in results:
                metadata = result.get('metadata', {})

                # 检查所有条件
                condition_results = []
                for condition in metadata_filter.conditions:
                    condition_met = self._check_metadata_condition(metadata, condition)
                    condition_results.append(condition_met)

                # 根据逻辑操作符合并条件结果
                if metadata_filter.logical_operator.value == "and":
                    if all(condition_results):
                        filtered_results.append(result)
                else:  # or
                    if any(condition_results):
                        filtered_results.append(result)

            logger.info(f"元数据筛选: {len(results)} -> {len(filtered_results)}")
            return filtered_results

        except Exception as e:
            logger.error(f"元数据筛选失败: {e}")
            return results  # 筛选失败时返回原结果

    def _check_metadata_condition(self, metadata: Dict[str, Any], condition) -> bool:
        """检查单个元数据条件"""
        try:
            # 获取要检查的字段值
            values = []
            for field_name in condition.name:
                if field_name in metadata:
                    values.append(str(metadata[field_name]))

            if not values:
                # 字段不存在的情况
                return condition.comparison_operator in [ComparisonOperator.EMPTY, ComparisonOperator.IS_NOT]

            # 将所有值合并为一个字符串进行检查
            combined_value = " ".join(values).lower()
            condition_value = (condition.value or "").lower()

            # 根据操作符进行比较
            operator = condition.comparison_operator

            if operator == ComparisonOperator.CONTAINS:
                return condition_value in combined_value
            elif operator == ComparisonOperator.NOT_CONTAINS:
                return condition_value not in combined_value
            elif operator == ComparisonOperator.START_WITH:
                return combined_value.startswith(condition_value)
            elif operator == ComparisonOperator.END_WITH:
                return combined_value.endswith(condition_value)
            elif operator == ComparisonOperator.IS:
                return combined_value == condition_value
            elif operator == ComparisonOperator.IS_NOT:
                return combined_value != condition_value
            elif operator == ComparisonOperator.EMPTY:
                return not combined_value.strip()
            elif operator == ComparisonOperator.NOT_EMPTY:
                return bool(combined_value.strip())
            else:
                # 对于数值比较，尝试转换为数字
                try:
                    num_value = float(combined_value)
                    condition_num = float(condition_value)

                    if operator == ComparisonOperator.EQUAL:
                        return num_value == condition_num
                    elif operator == ComparisonOperator.NOT_EQUAL:
                        return num_value != condition_num
                    elif operator == ComparisonOperator.GREATER:
                        return num_value > condition_num
                    elif operator == ComparisonOperator.LESS:
                        return num_value < condition_num
                    elif operator == ComparisonOperator.GREATER_EQUAL:
                        return num_value >= condition_num
                    elif operator == ComparisonOperator.LESS_EQUAL:
                        return num_value <= condition_num
                except ValueError:
                    # 无法转换为数字，返回False
                    return False

            return False

        except Exception as e:
            logger.error(f"检查元数据条件失败: {e}")
            return False

    def _apply_score_threshold(self, results: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """应用分数阈值筛选"""
        try:
            # 使用相似度分数进行筛选
            filtered_results = []
            for result in results:
                # 优先使用cosine_similarity，其次是similarity，最后是1-distance
                score = result.get('cosine_similarity')
                if score is None:
                    score = result.get('similarity')
                if score is None:
                    distance = result.get('distance', 1.0)
                    score = 1 - distance

                if score >= threshold:
                    filtered_results.append(result)

            logger.info(f"分数阈值筛选({threshold}): {len(results)} -> {len(filtered_results)}")
            return filtered_results

        except Exception as e:
            logger.error(f"分数阈值筛选失败: {e}")
            return results

    def _format_results(self, results: List[Dict[str, Any]]) -> List[RetrievalRecord]:
        """格式化搜索结果为API响应格式"""
        records = []

        for result in results:
            try:
                # 获取文档内容
                content = result.get('document', '')
                if not content:
                    # 如果没有document字段，尝试从metadata中构建
                    metadata = result.get('metadata', {})
                    content_parts = []
                    for key, value in metadata.items():
                        if key not in ['title', 'path', 'id'] and isinstance(value, str):
                            content_parts.append(value)
                    content = ' '.join(content_parts) if content_parts else 'No content available'

                # 获取标题
                title = ''
                metadata = result.get('metadata', {})
                if 'title' in metadata:
                    title = metadata['title']
                elif 'filename' in metadata:
                    title = metadata['filename']
                elif 'path' in metadata:
                    title = metadata['path'].split('/')[-1] if '/' in metadata['path'] else metadata['path']
                else:
                    title = f"Document {result.get('id', 'Unknown')}"

                # 获取分数
                score = result.get('cosine_similarity')
                if score is None:
                    score = result.get('similarity')
                if score is None:
                    distance = result.get('distance', 1.0)
                    score = max(0, 1 - distance)  # 确保分数不为负

                # 确保分数在0-1范围内
                score = max(0, min(1, score))

                # 清理元数据（移除系统字段）
                clean_metadata = {}
                for key, value in metadata.items():
                    if key not in ['embedding', 'dense_vector']:
                        clean_metadata[key] = value

                # 确保文档 ID 存在于 metadata 中（用于 UMAP 等功能的 ID 映射）
                if 'id' in result:
                    clean_metadata['id'] = result['id']

                record = RetrievalRecord(
                    content=content,
                    score=score,
                    title=title,
                    metadata=clean_metadata if clean_metadata else None
                )

                records.append(record)

            except Exception as e:
                logger.error(f"格式化结果失败: {e}, result: {result}")
                continue

        return records

    async def list_collections(self) -> List[str]:
        """列出所有知识库集合"""
        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                None, self.vector_client.list_collections
            )
            return collections
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            raise

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None, self.vector_client.get_collection_stats, collection_name
            )
            return stats
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            raise


    async def delete_collection(self, collection_name: str) -> bool:
        """删除知识库集合"""
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self.vector_client.delete_collection, collection_name
            )
            logger.info(f"集合 '{collection_name}' 删除{'成功' if success else '失败'}")
            return success
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    async def process_and_vectorize_file(
            self,
            file_content: bytes,
            filename: str,
            embedding_template: str,
            document_template: str,  # 新增document模板参数
            collection_name: str,
            batch_size: int = 5  # 默认值改为5
    ) -> Dict[str, Any]:
        """
        处理上传的文件并进行向量化存储
        """
        try:
            logger.info(f"开始处理文件: {filename}")

            # 1. 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # 2. 读取数据文件
                df = await self._read_data_file(temp_file_path, filename)

                # 3. 验证embedding模板字段
                embedding_fields = re.findall(r'\{([^}]+)\}', embedding_template)
                missing_embedding_fields = [field for field in embedding_fields if field not in df.columns]
                if missing_embedding_fields:
                    return {
                        "success": False,
                        "message": f"Embedding模板中的字段在数据中未找到: {missing_embedding_fields}",
                        "data": {"available_fields": list(df.columns)}
                    }

                # 4. 验证document模板字段
                document_fields = re.findall(r'\{([^}]+)\}', document_template)
                missing_document_fields = [field for field in document_fields if field not in df.columns]
                if missing_document_fields:
                    return {
                        "success": False,
                        "message": f"Document模板中的字段在数据中未找到: {missing_document_fields}",
                        "data": {"available_fields": list(df.columns)}
                    }

                # 5. 创建或获取collection
                if not self.vector_client.has_collection(collection_name):
                    self.vector_client.create_collection(collection_name)

                # 6. 生成embedding文本和document文本
                embedding_texts = []
                document_texts = []

                for idx, row in df.iterrows():
                    # 清理行数据
                    row_dict = {}
                    for column, value in row.items():
                        if pd.isna(value) or value in [np.inf, -np.inf]:
                            row_dict[column] = ""
                        else:
                            row_dict[column] = str(value) if not isinstance(value, (str, int, float, bool)) else value

                    # 生成embedding文本
                    embedding_text = self._parse_embedding_template(embedding_template, row_dict)
                    embedding_texts.append(embedding_text)

                    # 生成document文本
                    document_text = self._parse_embedding_template(document_template, row_dict)
                    document_texts.append(document_text)

                logger.info(f"生成了 {len(embedding_texts)} 个embedding文本和 {len(document_texts)} 个document文本")

                # 7. 向量化
                loop = asyncio.get_event_loop()
                vectors = await loop.run_in_executor(
                    None, self.embedding_client.encode_texts, embedding_texts
                )

                # 8. 分批插入数据
                total_batches = (len(df) + batch_size - 1) // batch_size
                inserted_count = 0

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(df))

                    logger.info(f"处理批次 {batch_idx + 1}/{total_batches} (记录 {start_idx + 1}-{end_idx})")

                    # 准备批次数据
                    batch_entities = []
                    for i in range(start_idx, end_idx):
                        entity = {}

                        # 添加所有原始字段
                        for column in df.columns:
                            value = df.iloc[i][column]
                            if pd.isna(value):
                                entity[column] = ""
                            else:
                                entity[column] = str(value) if not isinstance(value, (str, int, float, bool)) else value

                        # 添加向量和文档
                        entity["embedding"] = vectors[i]
                        entity["dense_vector"] = vectors[i]
                        entity["document"] = document_texts[i]  # 使用document模板生成的文本
                        entity["source_file"] = filename
                        entity["upload_time"] = time.time()

                        batch_entities.append(entity)

                    # 插入数据
                    success = self.vector_client.insert_data(collection_name, batch_entities)
                    if success:
                        inserted_count += len(batch_entities)
                        logger.info(f"批次 {batch_idx + 1} 插入成功，记录数: {len(batch_entities)}")
                    else:
                        logger.error(f"批次 {batch_idx + 1} 插入失败")

                # 9. 获取最终统计信息
                stats = self.vector_client.get_collection_stats(collection_name)

                return {
                    "success": True,
                    "message": "向量化存储完成",
                    "data": {
                        "collection_name": collection_name,
                        "file_name": filename,
                        "total_records": len(df),
                        "inserted_records": inserted_count,
                        "embedding_template_used": embedding_template,
                        "document_template_used": document_template,
                        "batch_size": batch_size,
                        "total_batches": total_batches,
                        "collection_stats": stats
                    }
                }

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"向量化处理失败: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"处理失败: {str(e)}"
            }

    async def _read_data_file(self, file_path: str, filename: str) -> pd.DataFrame:
        """读取数据文件"""
        file_path = Path(file_path)

        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                # 尝试不同编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("无法解码CSV文件，请检查文件编码")
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")

            logger.info(f"成功读取文件: {filename}, 数据形状: {df.shape}, 列名: {list(df.columns)}")
            return df

        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise

    def _parse_embedding_template(self, template: str, row_data: Dict) -> str:
        """解析embedding模板"""
        try:
            pattern = r'\{([^}]+)\}'
            matches = re.findall(pattern, template)

            formatted_text = template
            for field_name in matches:
                if field_name in row_data:
                    value = row_data[field_name]

                    # 处理特殊值
                    if value is None or pd.isna(value):
                        value = ""
                    elif value in [np.inf, -np.inf]:
                        value = "∞" if value == np.inf else "-∞"
                    elif not isinstance(value, str):
                        value = str(value)

                    # 清理可能的 'nan' 字符串
                    if value in ['nan', 'NaN', 'None']:
                        value = ""

                    formatted_text = formatted_text.replace(f'{{{field_name}}}', value)
                else:
                    formatted_text = formatted_text.replace(f'{{{field_name}}}', "")

            return formatted_text.strip()
        except Exception as e:
            logger.error(f"模板解析失败: {e}")
            return str(row_data)

    def _compute_document_hash(self, text: str) -> str:
        """计算 document 文本的 MD5 hash，用于去重"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    async def get_existing_values_for_dedup(
            self,
            collection_name: str,
            dedup_field: str = None
    ) -> set:
        """
        获取知识库中已存在的去重值集合

        Args:
            collection_name: 集合名称
            dedup_field: 去重字段名。如果为 None，则使用 document 文本的 hash

        Returns:
            已存在值的集合（hash 值或字段值）
        """
        if not self.vector_client.has_collection(collection_name):
            return set()

        loop = asyncio.get_event_loop()
        all_data = await loop.run_in_executor(
            None, self.vector_client.get_all_data, collection_name, None
        )

        existing_values = set()
        for doc in all_data:
            if dedup_field:
                # 使用指定字段值
                field_value = doc.get(dedup_field)
                if field_value is not None and field_value != '':
                    existing_values.add(str(field_value))
            else:
                # 使用 document 文本的 hash
                doc_text = doc.get('document', '')
                if doc_text:
                    existing_values.add(self._compute_document_hash(doc_text))

        return existing_values

    async def get_file_preview(self, file_content: bytes, filename: str, max_rows: int = 5) -> Dict[str, Any]:
        """预览文件内容，用于前端显示"""
        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # 读取数据
                df = await self._read_data_file(temp_file_path, filename)

                # 清理数据中的 NaN、无穷大值等 JSON 不兼容的值
                df_cleaned = self._clean_dataframe_for_json(df.copy())

                # 获取预览数据
                preview_data = df_cleaned.head(max_rows).to_dict('records')

                return {
                    "success": True,
                    "data": {
                        "columns": list(df.columns),
                        "total_rows": len(df),
                        "preview_rows": len(preview_data),
                        "preview_data": preview_data,
                        "file_info": {
                            "name": filename,
                            "size": len(file_content),
                            "type": Path(filename).suffix
                        }
                    }
                }

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"文件预览失败: {e}")
            return {
                "success": False,
                "message": f"预览失败: {str(e)}"
            }

    def _clean_dataframe_for_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理 DataFrame 中不兼容 JSON 的值"""
        try:
            # 处理每一列的数据
            for column in df.columns:
                # 替换 NaN 值为空字符串
                df[column] = df[column].fillna("")

                # 处理数值列中的无穷大值
                if df[column].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # 将无穷大值替换为字符串
                    df[column] = df[column].replace([np.inf, -np.inf], ["∞", "-∞"])

                    # 确保 NaN 被处理
                    df[column] = df[column].fillna("")

                # 将所有值转换为字符串，确保 JSON 兼容性
                df[column] = df[column].astype(str)

                # 处理可能的 'nan' 字符串
                df[column] = df[column].replace(['nan', 'NaN', 'None'], "")

            return df

        except Exception as e:
            logger.error(f"清理数据失败: {e}")
            # 如果清理失败，尝试简单的字符串转换
            for column in df.columns:
                df[column] = df[column].astype(str).replace(['nan', 'NaN', 'None'], "")
            return df

    def update_progress(self, task_id: str, progress_storage: dict,
                        stage: str, stage_number: int, total_stages: int,
                        current_batch: int = 0, total_batches: int = 0,
                        message: str = "", status: str = "processing",
                        stage_completed: bool = False, batch_completed: bool = False,
                        sub_progress: float = 0.0):  # 新增：子阶段进度（0.0-1.0）
        """更新任务进度"""
        if task_id not in progress_storage:
            return

        # 计算总体进度百分比
        if stage_completed:
            # 当前阶段已完成
            completed_stages = stage_number
            batch_progress = 0
            sub_stage_progress = 0
        else:
            # 当前阶段未完成，计算已完成的阶段
            completed_stages = stage_number - 1

            # 在当前阶段中的进度计算
            if total_batches > 0:
                # 有批次信息的情况
                completed_batches = current_batch if batch_completed else max(0, current_batch - 1)
                # sub_progress 表示当前未完成 batch 内部的进度 (0.0-1.0)
                inner_progress = sub_progress if not batch_completed and sub_progress > 0 else 0
                batch_progress = (completed_batches + inner_progress) / total_batches / total_stages
                sub_stage_progress = 0
            else:
                # 没有批次信息但有子进度的情况
                batch_progress = 0
                # 子阶段进度贡献到当前阶段
                sub_stage_progress = sub_progress / total_stages

        stage_progress = completed_stages / total_stages
        progress_percent = min(100, (stage_progress + batch_progress + sub_stage_progress) * 100)

        progress_storage[task_id].update({
            "status": status,
            "stage": stage,
            "stage_number": stage_number,
            "total_stages": total_stages,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "progress_percent": round(progress_percent, 1),
            "message": message
        })

        logger.info(
            f"任务 {task_id} 进度更新: {stage} ({stage_number}/{total_stages}) - 批次 {current_batch}/{total_batches} - {progress_percent}% (已完成: {completed_stages}阶段, 子进度: {sub_progress:.2f})")

    def sanitize_for_json(self, data):
        """清理数据中的NumPy类型，确保JSON兼容性"""
        if isinstance(data, dict):
            return {k: self.sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()  # 转换为Python原生数值类型
        elif isinstance(data, np.ndarray):
            return data.tolist()  # 转换为Python列表
        elif isinstance(data, np.bool_):
            return bool(data)
        elif pd.isna(data):
            return None
        else:
            return data

    def _split_into_chunks(self, texts: List[str], num_chunks: int) -> List[List[str]]:
        """将文本列表均分为多个块，用于并发Worker处理"""
        if num_chunks <= 1 or len(texts) == 0:
            return [texts]

        chunk_size = (len(texts) + num_chunks - 1) // num_chunks
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            if chunk:  # 只添加非空块
                chunks.append(chunk)
        return chunks

    async def _vectorize_single_text(
            self,
            text: str,
            embedding_provider: str = None,
            embedding_model: str = None
    ) -> List[float]:
        """单条文本向量化（用于容错处理）"""
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            self.embedding_service.encode_texts,
            [text],
            embedding_provider,
            embedding_model,
            1  # batch_size=1
        )
        return vectors[0] if vectors else None

    async def process_and_vectorize_file_async(
            self,
            task_id: str,
            file_content: bytes,
            filename: str,
            embedding_template: str,
            document_template: str,
            collection_name: str,
            batch_size: int,
            num_workers: int,
            progress_storage: dict,
            embedding_provider: str = None,
            embedding_model: str = None,
            enable_incremental: bool = True,
            dedup_field: str = None,
            insert_batch_multiplier: int = 10,
            enable_column_update: bool = False
    ):
        """
        异步处理文件向量化（带进度追踪，支持并发Worker，支持增量上传和容错处理）

        Args:
            enable_incremental: 是否启用增量上传（跳过已存在的记录）
            dedup_field: 去重字段名。如果为 None，则使用 document 文本的 hash 进行去重
            enable_column_update: 是否启用列更新（更新已存在记录的 metadata）
        """
        try:
            logger.info(f"任务 {task_id} 开始,收到参数：batch_size={batch_size}, num_workers={num_workers}, "
                       f"insert_batch_multiplier={insert_batch_multiplier}, "
                       f"enable_incremental={enable_incremental}, dedup_field={dedup_field}, "
                       f"enable_column_update={enable_column_update}")
            total_stages = 5  # 阶段5合并了向量化和存储

            # 初始化失败记录收集器
            failed_records = []
            skipped_duplicates = 0
            embedding_failed_count = 0
            storage_failed_count = 0
            updated_count = 0  # 列更新的记录数

            # 阶段1: 读取文件 - 开始
            self.update_progress(task_id, progress_storage, "读取文件数据", 1, total_stages,
                                 message="正在解析文件格式和内容...")

            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # 读取数据文件
                df = await self._read_data_file(temp_file_path, filename)

                # 阶段1: 读取文件 - 完成
                self.update_progress(task_id, progress_storage, "读取文件数据", 1, total_stages,
                                     message=f"文件解析完成，共 {len(df)} 行数据，{len(df.columns)} 个字段",
                                     stage_completed=True)

                # 阶段2: 验证模板 - 开始
                self.update_progress(task_id, progress_storage, "验证模板字段", 2, total_stages,
                                     message="正在验证模板中的字段是否存在...")

                # 验证embedding模板字段
                embedding_fields = re.findall(r'\{([^}]+)\}', embedding_template)
                missing_embedding_fields = [field for field in embedding_fields if field not in df.columns]
                if missing_embedding_fields:
                    progress_storage[task_id].update({
                        "status": "error",
                        "error": f"Embedding模板中的字段在数据中未找到: {missing_embedding_fields}",
                        "result": {"available_fields": list(df.columns)}
                    })
                    return

                # 验证document模板字段
                document_fields = re.findall(r'\{([^}]+)\}', document_template)
                missing_document_fields = [field for field in document_fields if field not in df.columns]
                if missing_document_fields:
                    progress_storage[task_id].update({
                        "status": "error",
                        "error": f"Document模板中的字段在数据中未找到: {missing_document_fields}",
                        "result": {"available_fields": list(df.columns)}
                    })
                    return

                # 阶段2: 验证模板 - 完成
                self.update_progress(task_id, progress_storage, "验证模板字段", 2, total_stages,
                                     message="模板验证通过，所有字段都存在于数据中",
                                     stage_completed=True)

                # ========== 阶段3: 去重检查（新增） ==========
                # 先生成所有记录的 document 文本，用于去重判断
                all_row_data = []  # 存储 (原始索引, row_dict, document_text)
                for idx, row in df.iterrows():
                    row_dict = {}
                    for column, value in row.items():
                        if pd.isna(value) or value in [np.inf, -np.inf]:
                            row_dict[column] = ""
                        else:
                            row_dict[column] = str(value) if not isinstance(value, (str, int, float, bool)) else value
                    document_text = self._parse_embedding_template(document_template, row_dict)
                    all_row_data.append((idx, row_dict, document_text))

                # 执行去重检查
                rows_to_process = []  # 需要处理的记录：(原始索引, row_dict, document_text)
                original_indices = []  # 保存原始DataFrame索引，用于后续存储
                records_to_update = []  # 列更新模式下需要更新的记录：(id, row_dict)

                if enable_incremental:
                    self.update_progress(task_id, progress_storage, "去重检查", 3, total_stages,
                                        message="正在获取知识库已有数据进行比对...")

                    # 获取已存在的值集合
                    existing_values = await self.get_existing_values_for_dedup(collection_name, dedup_field)

                    self.update_progress(task_id, progress_storage, "去重检查", 3, total_stages,
                                        message=f"已获取 {len(existing_values)} 条已有记录，正在比对...")

                    # 比对去重
                    for idx, row_dict, document_text in all_row_data:
                        if dedup_field:
                            # 使用指定字段值
                            dedup_value = str(row_dict.get(dedup_field, ''))
                        else:
                            # 使用 document 文本的 hash
                            dedup_value = self._compute_document_hash(document_text)

                        if dedup_value in existing_values:
                            if enable_column_update and dedup_field == 'id':
                                # 列更新模式：收集需要更新 metadata 的记录
                                records_to_update.append((dedup_value, row_dict))
                            else:
                                # 普通模式：跳过重复记录
                                skipped_duplicates += 1
                        else:
                            rows_to_process.append((idx, row_dict, document_text))
                            original_indices.append(idx)
                            # 将新值加入集合，避免文件内重复
                            existing_values.add(dedup_value)

                    # 执行列更新
                    if records_to_update:
                        self.update_progress(task_id, progress_storage, "去重检查", 3, total_stages,
                                            message=f"正在更新 {len(records_to_update)} 条已有记录的 metadata...")

                        update_batch_size = 100
                        for batch_start in range(0, len(records_to_update), update_batch_size):
                            batch_end = min(batch_start + update_batch_size, len(records_to_update))
                            batch = records_to_update[batch_start:batch_end]

                            ids_to_update = [item[0] for item in batch]
                            metadatas_to_update = []

                            for record_id, row_dict in batch:
                                # 清理 metadata，移除系统字段
                                clean_metadata = {}
                                for k, v in row_dict.items():
                                    if k not in ['id', 'embedding', 'dense_vector', 'document']:
                                        # 确保值是 ChromaDB 支持的基本类型
                                        if isinstance(v, (str, int, float, bool)):
                                            clean_metadata[k] = v
                                        elif v is None or (isinstance(v, float) and (pd.isna(v) or v in [np.inf, -np.inf])):
                                            clean_metadata[k] = ""
                                        else:
                                            clean_metadata[k] = str(v)
                                # 添加更新时间戳和来源文件
                                clean_metadata['upload_time'] = time.time()
                                clean_metadata['source_file'] = filename
                                metadatas_to_update.append(clean_metadata)

                            try:
                                success = self.vector_client.update_data(
                                    collection_name,
                                    ids_to_update,
                                    embeddings=None,  # 不更新向量
                                    documents=None,   # 不更新 document
                                    metadatas=metadatas_to_update
                                )
                                if success:
                                    updated_count += len(batch)
                            except Exception as e:
                                logger.warning(f"任务 {task_id} 批量更新 metadata 失败: {e}")
                                # 继续处理其他批次

                    self.update_progress(task_id, progress_storage, "去重检查", 3, total_stages,
                                        message=f"去重完成：跳过 {skipped_duplicates} 条，更新 {updated_count} 条，待新增 {len(rows_to_process)} 条",
                                        stage_completed=True)

                    # 更新进度存储中的计数
                    progress_storage[task_id]["skipped_duplicates"] = skipped_duplicates
                    progress_storage[task_id]["updated_records"] = updated_count
                else:
                    # 不启用增量上传，处理所有记录
                    self.update_progress(task_id, progress_storage, "去重检查", 3, total_stages,
                                        message="增量上传已禁用，将处理所有记录",
                                        stage_completed=True)
                    rows_to_process = all_row_data
                    original_indices = [idx for idx, _, _ in all_row_data]

                # 如果没有新记录需要处理
                if not rows_to_process:
                    stats = self.vector_client.get_collection_stats(collection_name) if self.vector_client.has_collection(collection_name) else {}
                    message = "所有记录已存在于知识库中，无新增数据"
                    if updated_count > 0:
                        message = f"列更新完成：更新 {updated_count} 条记录，无新增数据"
                    progress_storage[task_id].update({
                        "status": "completed",
                        "stage": "处理完成",
                        "stage_number": total_stages,
                        "total_stages": total_stages,
                        "progress_percent": 100,
                        "message": message,
                        "result": {
                            "collection_name": collection_name,
                            "file_name": filename,
                            "total_records": len(df),
                            "skipped_duplicates": skipped_duplicates,
                            "updated_records": updated_count,
                            "inserted_records": 0,
                            "embedding_failed_count": 0,
                            "storage_failed_count": 0,
                            "collection_stats": stats
                        }
                    })
                    return

                # 阶段4: 准备向量数据库 - 开始
                self.update_progress(task_id, progress_storage, "准备向量数据库", 4, total_stages,
                                     message="正在创建或连接collection...")

                # 创建或获取collection，并管理 metadata
                if not self.vector_client.has_collection(collection_name):
                    # 非增量上传：创建新 Collection 并保存 metadata
                    # 生成默认的 display_name 作为 description（将下划线替换为空格，首字母大写）
                    default_description = collection_name.replace("_", " ").title()
                    collection_metadata = {
                        "embedding_template": embedding_template,
                        "document_template": document_template,
                        "source_files": filename,  # 初始只有一个文件
                        "user_description": default_description  # 用户可编辑的描述，默认为格式化后的库名
                    }
                    self.vector_client.create_collection(collection_name, metadata=collection_metadata)
                    logger.info(f"任务 {task_id} 创建新 Collection '{collection_name}'，保存 metadata: {collection_metadata}")
                else:
                    # 增量上传：更新 source_files
                    if enable_incremental:
                        try:
                            stats = self.vector_client.get_collection_stats(collection_name)
                            current_metadata = stats.get("metadata", {})
                            current_files = current_metadata.get("source_files", "")

                            # 追加新文件名（避免重复）
                            file_list = [f.strip() for f in current_files.split(",") if f.strip()]
                            if filename not in file_list:
                                file_list.append(filename)
                                new_source_files = ",".join(file_list)
                                self.vector_client.update_collection_metadata(
                                    collection_name,
                                    {"source_files": new_source_files}
                                )
                                logger.info(f"任务 {task_id} 更新 Collection '{collection_name}' 的 source_files: {new_source_files}")
                        except Exception as metadata_error:
                            logger.warning(f"任务 {task_id} 更新 metadata 失败（不影响上传）: {metadata_error}")

                # 计算待处理记录数
                records_to_process_count = len(rows_to_process)

                # 阶段4: 准备向量数据库 - 完成
                self.update_progress(task_id, progress_storage, "准备向量数据库", 4, total_stages,
                                     message=f"Collection准备完成，共 {records_to_process_count} 条记录待处理",
                                     stage_completed=True)

                # 阶段5: 向量化并存储（边向量化边插入，减少堆积，防止中途出错前面白跑）
                self.update_progress(task_id, progress_storage, "向量化并存储", 5, total_stages,
                                     message="正在准备embedding文本...")

                # 从 rows_to_process 中提取 embedding 文本和 document 文本
                embedding_texts = []
                document_texts = []
                processed_row_data = []  # 保存处理过的行数据，用于后续存储

                for idx, row_dict, document_text in rows_to_process:
                    # 生成 embedding 文本
                    embedding_text = self._parse_embedding_template(embedding_template, row_dict)
                    embedding_texts.append(embedding_text)
                    document_texts.append(document_text)
                    processed_row_data.append((idx, row_dict))

                # 【重要】记录使用的embedding模型
                logger.info(f"任务 {task_id} 使用embedding模型: provider={embedding_provider}, model={embedding_model}, batch_size={batch_size}, num_workers={num_workers}")

                # 计算插入批次信息
                insert_batch_size = batch_size * insert_batch_multiplier
                total_insert_batches = (records_to_process_count + insert_batch_size - 1) // insert_batch_size if records_to_process_count > 0 else 0
                inserted_count = 0

                embedding_start_time = time.time()
                loop = asyncio.get_event_loop()

                self.update_progress(task_id, progress_storage, "向量化并存储", 5, total_stages,
                                     1, total_insert_batches,
                                     f"共 {records_to_process_count} 条记录，分 {total_insert_batches} 批处理（每批 {insert_batch_size} 条），使用 {num_workers} 个Worker")

                # 按 insert_batch_size 分批，每批内向量化后立即插入数据库
                for insert_batch_idx in range(total_insert_batches):
                    chunk_start = insert_batch_idx * insert_batch_size
                    chunk_end = min((insert_batch_idx + 1) * insert_batch_size, records_to_process_count)
                    current_insert_batch = insert_batch_idx + 1

                    chunk_embedding_texts = embedding_texts[chunk_start:chunk_end]
                    chunk_document_texts = document_texts[chunk_start:chunk_end]
                    chunk_row_data = processed_row_data[chunk_start:chunk_end]

                    # --- 向量化当前 chunk ---
                    self.update_progress(task_id, progress_storage, "向量化并存储", 5, total_stages,
                                         current_insert_batch, total_insert_batches,
                                         f"批次 {current_insert_batch}/{total_insert_batches} 正在向量化 ({chunk_end - chunk_start} 条)...",
                                         sub_progress=0.0)

                    chunk_vectors = None

                    if num_workers <= 1:
                        # 单Worker模式（带容错）
                        def make_embedding_callback(ib_idx, total_ib):
                            def callback(completed_batches: int, total_embedding_batches: int, message: str):
                                embedding_progress = (completed_batches / total_embedding_batches) if total_embedding_batches > 0 else 0
                                # sub_progress 表示当前插入批次内 embedding 的进度 (0.0-0.9)，预留0.1给插入
                                self.update_progress(
                                    task_id, progress_storage, "向量化并存储", 5, total_stages,
                                    ib_idx + 1, total_ib,
                                    f"批次 {ib_idx+1}/{total_ib} 向量化: {completed_batches}/{total_embedding_batches} - {message}",
                                    sub_progress=embedding_progress * 0.9
                                )
                            return callback

                        try:
                            chunk_vectors = await loop.run_in_executor(
                                None,
                                self.embedding_service.encode_texts_with_progress,
                                chunk_embedding_texts,
                                make_embedding_callback(insert_batch_idx, total_insert_batches),
                                embedding_provider,
                                embedding_model,
                                batch_size
                            )
                        except Exception as batch_error:
                            # 批量失败，降级为逐条向量化
                            logger.warning(f"任务 {task_id} 批次 {current_insert_batch} 批量向量化失败: {batch_error}，尝试逐条处理")
                            chunk_vectors = []
                            for i, text in enumerate(chunk_embedding_texts):
                                try:
                                    vec = await self._vectorize_single_text(text, embedding_provider, embedding_model)
                                    chunk_vectors.append(vec)
                                except Exception as single_error:
                                    logger.warning(f"任务 {task_id} 记录 {chunk_start + i} 向量化失败: {single_error}")
                                    embedding_failed_count += 1
                                    chunk_vectors.append(None)
                                    original_idx = chunk_row_data[i][0]
                                    failed_records.append({
                                        "row_index": original_idx,
                                        "reason": "embedding_failed",
                                        "error": str(single_error),
                                        "data": chunk_row_data[i][1]
                                    })
                                # 更新进度
                                if (i + 1) % 10 == 0 or (i + 1) == len(chunk_embedding_texts):
                                    progress = (i + 1) / len(chunk_embedding_texts)
                                    self.update_progress(
                                        task_id, progress_storage, "向量化并存储", 5, total_stages,
                                        current_insert_batch, total_insert_batches,
                                        f"批次 {current_insert_batch}/{total_insert_batches} 逐条向量化: {i + 1}/{len(chunk_embedding_texts)}",
                                        sub_progress=progress * 0.9
                                    )
                    else:
                        # 多Worker并发模式
                        text_chunks = self._split_into_chunks(chunk_embedding_texts, num_workers)
                        actual_workers = len(text_chunks)

                        # 进度聚合器
                        worker_progress = {i: {"completed": 0, "total": 0} for i in range(actual_workers)}
                        progress_lock = threading.Lock()

                        def make_worker_callback(worker_id, ib_idx, total_ib):
                            def callback(completed, total, message):
                                with progress_lock:
                                    worker_progress[worker_id] = {"completed": completed, "total": total}
                                    total_completed = sum(wp["completed"] for wp in worker_progress.values())
                                    total_batches_all = sum(wp["total"] for wp in worker_progress.values())
                                    if total_batches_all > 0:
                                        embedding_progress = total_completed / total_batches_all
                                        self.update_progress(
                                            task_id, progress_storage, "向量化并存储", 5, total_stages,
                                            ib_idx + 1, total_ib,
                                            f"批次 {ib_idx+1}/{total_ib} Worker进度: {total_completed}/{total_batches_all} ({actual_workers} Workers)",
                                            sub_progress=embedding_progress * 0.9
                                        )
                            return callback

                        async def encode_chunk(worker_id, texts, start_offset):
                            """单个Worker的编码任务（带容错）"""
                            callback = make_worker_callback(worker_id, insert_batch_idx, total_insert_batches)
                            try:
                                result = await loop.run_in_executor(
                                    None,
                                    self.embedding_service.encode_texts_with_progress_concurrent,
                                    texts,
                                    callback,
                                    embedding_provider,
                                    embedding_model,
                                    batch_size
                                )
                                return (worker_id, result, [])
                            except Exception as batch_error:
                                logger.warning(f"任务 {task_id} 批次 {current_insert_batch} Worker {worker_id} 批量向量化失败: {batch_error}，尝试逐条处理")
                                vectors_fallback = []
                                failed_indices = []
                                for i, text in enumerate(texts):
                                    global_idx = start_offset + i
                                    try:
                                        vec = await self._vectorize_single_text(text, embedding_provider, embedding_model)
                                        vectors_fallback.append(vec)
                                    except Exception as single_error:
                                        logger.warning(f"任务 {task_id} Worker {worker_id} 记录 {global_idx} 向量化失败: {single_error}")
                                        vectors_fallback.append(None)
                                        failed_indices.append(global_idx)
                                return (worker_id, vectors_fallback, failed_indices)

                        # 计算每个chunk的起始偏移量
                        chunk_offsets = []
                        offset_val = 0
                        for chunk in text_chunks:
                            chunk_offsets.append(offset_val)
                            offset_val += len(chunk)

                        tasks = [encode_chunk(i, chunk, chunk_offsets[i]) for i, chunk in enumerate(text_chunks)]
                        results = await asyncio.gather(*tasks)

                        # 按原始顺序合并结果，并收集失败记录
                        chunk_vectors = []
                        for worker_id, result, failed_indices in sorted(results, key=lambda x: x[0]):
                            chunk_vectors.extend(result)
                            for local_idx in failed_indices:
                                embedding_failed_count += 1
                                original_idx = chunk_row_data[local_idx][0]
                                failed_records.append({
                                    "row_index": original_idx,
                                    "reason": "embedding_failed",
                                    "error": "向量化失败",
                                    "data": chunk_row_data[local_idx][1]
                                })

                    # --- 过滤失败记录并构建 entity ---
                    self.update_progress(task_id, progress_storage, "向量化并存储", 5, total_stages,
                                         current_insert_batch, total_insert_batches,
                                         f"批次 {current_insert_batch}/{total_insert_batches} 正在插入数据库...",
                                         sub_progress=0.9)

                    batch_entities = []
                    for i in range(len(chunk_vectors)):
                        if chunk_vectors[i] is None:
                            continue  # 跳过 embedding 失败的记录
                        try:
                            original_idx, row_dict = chunk_row_data[i]
                            entity = {}
                            for key, value in row_dict.items():
                                entity[key] = self.sanitize_for_json(value)
                            entity["embedding"] = self.sanitize_for_json(chunk_vectors[i])
                            entity["dense_vector"] = self.sanitize_for_json(chunk_vectors[i])
                            entity["document"] = chunk_document_texts[i]
                            entity["source_file"] = filename
                            entity["upload_time"] = time.time()
                            entity = self.sanitize_for_json(entity)
                            batch_entities.append((chunk_start + i, entity))
                        except Exception as e:
                            logger.warning(f"任务 {task_id} 记录 {chunk_start + i} 准备失败: {e}")
                            storage_failed_count += 1
                            failed_records.append({
                                "row_index": original_idx if 'original_idx' in dir() else chunk_start + i,
                                "reason": "prepare_failed",
                                "error": str(e),
                                "data": row_dict if 'row_dict' in dir() else {}
                            })

                    # --- 立即插入数据库 ---
                    if batch_entities:
                        entities_to_insert = [entity for _, entity in batch_entities]
                        try:
                            success = self.vector_client.insert_data(collection_name, entities_to_insert)
                            if success:
                                inserted_count += len(entities_to_insert)
                                logger.info(f"任务 {task_id} 批次 {current_insert_batch} 插入成功，记录数: {len(entities_to_insert)}")
                            else:
                                # 批量插入失败，尝试逐条插入
                                logger.warning(f"任务 {task_id} 批次 {current_insert_batch} 批量插入失败，尝试逐条插入")
                                for idx, entity in batch_entities:
                                    try:
                                        single_success = self.vector_client.insert_data(collection_name, [entity])
                                        if single_success:
                                            inserted_count += 1
                                        else:
                                            original_idx = processed_row_data[idx][0]
                                            storage_failed_count += 1
                                            failed_records.append({
                                                "row_index": original_idx,
                                                "reason": "storage_failed",
                                                "error": "插入失败",
                                                "data": processed_row_data[idx][1]
                                            })
                                    except Exception as e:
                                        original_idx = processed_row_data[idx][0]
                                        storage_failed_count += 1
                                        failed_records.append({
                                            "row_index": original_idx,
                                            "reason": "storage_failed",
                                            "error": str(e),
                                            "data": processed_row_data[idx][1]
                                        })
                        except Exception as e:
                            # 批量插入异常，尝试逐条插入
                            logger.warning(f"任务 {task_id} 批次 {current_insert_batch} 批量插入异常: {e}，尝试逐条插入")
                            for idx, entity in batch_entities:
                                try:
                                    single_success = self.vector_client.insert_data(collection_name, [entity])
                                    if single_success:
                                        inserted_count += 1
                                    else:
                                        original_idx = processed_row_data[idx][0]
                                        storage_failed_count += 1
                                        failed_records.append({
                                            "row_index": original_idx,
                                            "reason": "storage_failed",
                                            "error": "插入失败",
                                            "data": processed_row_data[idx][1]
                                        })
                                except Exception as single_e:
                                    original_idx = processed_row_data[idx][0]
                                    storage_failed_count += 1
                                    failed_records.append({
                                        "row_index": original_idx,
                                        "reason": "storage_failed",
                                        "error": str(single_e),
                                        "data": processed_row_data[idx][1]
                                    })

                    # 当前插入批次完成
                    self.update_progress(task_id, progress_storage, "向量化并存储", 5, total_stages,
                                         current_insert_batch, total_insert_batches,
                                         f"第 {current_insert_batch}/{total_insert_batches} 批完成（已入库 {inserted_count} 条）",
                                         batch_completed=True)

                    # 添加小延迟，让前端能看到进度变化
                    await asyncio.sleep(0.1)

                embedding_api_duration = time.time() - embedding_start_time

                # 获取最终统计信息
                stats = self.vector_client.get_collection_stats(collection_name) if self.vector_client.has_collection(collection_name) else {}

                # 如果所有记录都处理失败
                if inserted_count == 0 and embedding_failed_count > 0:
                    progress_storage[task_id].update({
                        "status": "completed",
                        "stage": "处理完成",
                        "stage_number": total_stages,
                        "total_stages": total_stages,
                        "progress_percent": 100,
                        "message": "所有记录向量化失败，没有数据可存储",
                        "result": {
                            "collection_name": collection_name,
                            "file_name": filename,
                            "total_records": len(df),
                            "skipped_duplicates": skipped_duplicates,
                            "embedding_failed_count": embedding_failed_count,
                            "storage_failed_count": storage_failed_count,
                            "inserted_records": 0,
                            "failed_records": failed_records[:100] if len(failed_records) > 100 else failed_records,
                            "total_failed_records": len(failed_records),
                            "collection_stats": stats
                        }
                    })
                    return

                # 阶段5: 向量化并存储 - 完成（所有阶段完成）
                progress_storage[task_id].update({
                    "status": "completed",
                    "stage": "处理完成",
                    "stage_number": total_stages,
                    "total_stages": total_stages,
                    "current_batch": total_insert_batches,
                    "total_batches": total_insert_batches,
                    "progress_percent": 100,
                    "message": "向量化处理已完成",
                    "result": {
                        "collection_name": collection_name,
                        "file_name": filename,
                        "total_records": len(df),
                        "skipped_duplicates": skipped_duplicates,
                        "updated_records": updated_count,
                        "embedding_failed_count": embedding_failed_count,
                        "storage_failed_count": storage_failed_count,
                        "inserted_records": inserted_count,
                        "failed_records": failed_records[:100] if len(failed_records) > 100 else failed_records,
                        "total_failed_records": len(failed_records),
                        "embedding_template_used": embedding_template,
                        "document_template_used": document_template,
                        "embedding_provider": embedding_provider or "default",
                        "embedding_model": embedding_model or "default",
                        "batch_size": batch_size,
                        "insert_batch_size": insert_batch_size,
                        "total_batches": total_insert_batches,
                        "collection_stats": stats,
                        "embedding_api_duration": round(embedding_api_duration, 2)
                    }
                })

                logger.info(f"任务 {task_id} 处理完成")

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"任务 {task_id} 处理失败: {e}")
            traceback.print_exc()
            progress_storage[task_id].update({
                "status": "error",
                "error": f"处理失败: {str(e)}",
                "message": f"处理过程中发生错误: {str(e)}"
            })

    async def get_collection_documents(
        self,
        collection_name: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取集合中的文档列表（分页）

        Args:
            collection_name: 集合名称
            page: 页码（从1开始）
            page_size: 每页数量

        Returns:
            包含 documents、total、pages 的字典
        """
        try:
            loop = asyncio.get_event_loop()

            # 获取总数
            stats = await loop.run_in_executor(
                None, self.vector_client.get_collection_stats, collection_name
            )
            total = stats.get("row_count", 0)

            # 计算分页
            total_pages = (total + page_size - 1) // page_size if total > 0 else 1
            offset = (page - 1) * page_size

            # 获取数据 - ChromaDB 不支持 offset，需要获取更多数据然后截取
            limit = page * page_size
            all_data = await loop.run_in_executor(
                None, self.vector_client.get_all_data, collection_name, limit
            )

            # 截取当前页数据
            documents = all_data[offset:offset + page_size]

            # 清理数据，移除 embedding 以减少传输量
            cleaned_documents = []
            for doc in documents:
                cleaned_doc = {k: v for k, v in doc.items()
                             if k not in ['embedding', 'dense_vector']}
                cleaned_documents.append(cleaned_doc)

            return {
                "documents": cleaned_documents,
                "total": total,
                "pages": total_pages
            }

        except Exception as e:
            logger.error(f"获取文档列表失败: {e}")
            raise

    async def get_document_by_id(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """根据ID获取单个文档的完整信息

        Args:
            collection_name: 集合名称
            document_id: 文档ID

        Returns:
            文档数据字典，不存在则返回 None
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, self.vector_client.query_by_ids, collection_name, [document_id]
            )

            if results:
                doc = results[0]
                # 将 metadata 字段展开到顶层（保持与 get_all_data 一致的格式）
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    for key, value in doc['metadata'].items():
                        if key not in doc:
                            doc[key] = value
                    del doc['metadata']

                # 清理数据以确保 JSON 兼容性（移除 embedding 向量，转换 NumPy 类型）
                # 移除 embedding 和 dense_vector 字段（前端不需要）
                doc.pop('embedding', None)
                doc.pop('dense_vector', None)

                # 清理其他字段中的 NumPy 类型
                doc = self.sanitize_for_json(doc)

                return doc
            return None

        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None

    async def delete_document_from_collection(
        self,
        collection_name: str,
        document_id: str
    ) -> bool:
        """从集合中删除单个文档

        Args:
            collection_name: 集合名称
            document_id: 文档ID

        Returns:
            是否删除成功
        """
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self.vector_client.delete_by_ids, collection_name, [document_id]
            )

            logger.info(f"文档 {document_id} 从 {collection_name} 删除{'成功' if success else '失败'}")
            return success

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    async def update_document_in_collection(
        self,
        collection_name: str,
        document_id: str,
        document_content: str,
        metadata: Dict[str, Any],
        re_vectorize: bool = False,
        embedding_template: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新集合中的文档

        Args:
            collection_name: 集合名称
            document_id: 文档ID
            document_content: 新的文档内容
            metadata: 新的元数据
            re_vectorize: 是否重新向量化
            embedding_template: 重新向量化时使用的模板（可选）
            embedding_provider: embedding供应商（可选）
            embedding_model: embedding模型（可选）

        Returns:
            {"success": bool, "message": str}
        """
        try:
            loop = asyncio.get_event_loop()

            # 准备更新数据
            new_embedding = None

            if re_vectorize:
                # 生成用于向量化的文本
                if embedding_template:
                    # 使用模板生成文本，结合 metadata
                    embedding_text = self._parse_embedding_template(embedding_template, metadata)
                else:
                    # 使用 document 内容作为向量化文本
                    embedding_text = document_content

                logger.info(f"重新向量化文档 {document_id}，使用文本: {embedding_text[:100]}...")

                # 调用 embedding 服务
                embeddings = await loop.run_in_executor(
                    None,
                    self.embedding_service.encode_texts,
                    [embedding_text],
                    embedding_provider,
                    embedding_model,
                    1  # batch_size
                )
                new_embedding = embeddings[0]

            # 清理 metadata，移除系统字段
            clean_metadata = {k: v for k, v in metadata.items()
                            if k not in ['id', 'embedding', 'dense_vector', 'document']}

            # 执行更新
            success = await loop.run_in_executor(
                None,
                self.vector_client.update_data,
                collection_name,
                [document_id],
                [new_embedding] if new_embedding else None,
                [document_content],
                [clean_metadata]
            )

            if success:
                msg = "文档更新成功"
                if re_vectorize:
                    msg += "（已重新向量化）"
                return {
                    "success": True,
                    "message": msg
                }
            else:
                return {
                    "success": False,
                    "message": "文档更新失败"
                }

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"更新失败: {str(e)}"
            }
