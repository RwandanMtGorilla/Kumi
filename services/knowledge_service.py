from typing import List, Dict, Any
from vector_db.factory import VectorDBFactory
from vector_db.embedding_client import QwenEmbeddingAPI
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


logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识库服务"""

    def __init__(self):
        self.vector_client = VectorDBFactory.create_client()
        self.embedding_client = QwenEmbeddingAPI()

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

    async def _vectorize_query(self, query: str) -> List[List[float]]:
        """向量化查询文本"""
        try:
            # 使用asyncio在线程池中运行同步函数
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.embedding_client.encode_texts, [query]
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
                # 有批次信息的情况（如阶段5存储数据库）
                completed_batches = current_batch if batch_completed else max(0, current_batch - 1)
                batch_progress = completed_batches / total_batches / total_stages
                sub_stage_progress = 0
            else:
                # 没有批次信息但有子进度的情况（如阶段4的embedding）
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

    async def process_and_vectorize_file_async(
            self,
            task_id: str,
            file_content: bytes,
            filename: str,
            embedding_template: str,
            document_template: str,
            collection_name: str,
            batch_size: int,
            progress_storage: dict
    ):
        """异步处理文件向量化（带进度追踪）"""
        try:
            logger.info(f"任务 {task_id} 开始,收到参数：batch_size={batch_size}")
            total_stages = 5

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

                # 阶段3: 准备向量数据库 - 开始
                self.update_progress(task_id, progress_storage, "准备向量数据库", 3, total_stages,
                                     message="正在创建或连接collection...")

                # 创建或获取collection
                if not self.vector_client.has_collection(collection_name):
                    self.vector_client.create_collection(collection_name)

                # 计算批次信息
                total_batches = (len(df) + batch_size - 1) // batch_size

                # 阶段3: 准备向量数据库 - 完成
                self.update_progress(task_id, progress_storage, "准备向量数据库", 3, total_stages,
                                     message=f"Collection准备完成，将分 {total_batches} 个批次处理",
                                     stage_completed=True)

                # 阶段4: 生成向量嵌入 - 开始
                self.update_progress(task_id, progress_storage, "生成向量嵌入", 4, total_stages,
                                     message="正在生成embedding文本和document文本...")

                # 生成embedding文本和document文本
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

                    # 生成文本
                    embedding_text = self._parse_embedding_template(embedding_template, row_dict)
                    embedding_texts.append(embedding_text)

                    document_text = self._parse_embedding_template(document_template, row_dict)
                    document_texts.append(document_text)

                # 更新进度：文本生成完成，开始计算向量
                self.update_progress(task_id, progress_storage, "生成向量嵌入", 4, total_stages,
                                     message="正在计算向量嵌入...", sub_progress=0.1)  # 文本生成完成，给予10%的子进度

                # 【重要】设置embedding API的批处理大小
                logger.info(f"任务 {task_id} 设置embedding API批处理大小为: {batch_size}")
                self.embedding_client.set_batch_size(batch_size)

                # 定义embedding进度回调函数
                def embedding_progress_callback(completed_batches: int, total_embedding_batches: int, message: str):
                    """Embedding进度回调函数"""
                    # 计算当前阶段内的子进度（0.1 到 1.0，因为文本生成已经占用了0.1）
                    embedding_progress = 0.1 + (
                                completed_batches / total_embedding_batches * 0.9) if total_embedding_batches > 0 else 0.1

                    # 更新进度，传入子进度参数
                    self.update_progress(
                        task_id, progress_storage, "生成向量嵌入", 4, total_stages,
                        message=f"向量化进度: {completed_batches}/{total_embedding_batches} 批次 - {message}",
                        sub_progress=embedding_progress
                    )

                # 向量化（使用带进度的方法）
                loop = asyncio.get_event_loop()
                vectors = await loop.run_in_executor(
                    None,
                    self.embedding_client.encode_texts_with_progress,
                    embedding_texts,
                    embedding_progress_callback
                )

                # 阶段4: 生成向量嵌入 - 完成
                self.update_progress(task_id, progress_storage, "生成向量嵌入", 4, total_stages,
                                     message=f"向量嵌入生成完成，共 {len(vectors)} 个向量",
                                     stage_completed=True)

                # 阶段5: 存储到数据库 - 开始
                inserted_count = 0

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(df))
                    current_batch = batch_idx + 1

                    # 开始处理当前批次
                    self.update_progress(task_id, progress_storage, "存储到向量数据库", 5, total_stages,
                                         current_batch, total_batches,
                                         f"正在处理第 {current_batch}/{total_batches} 批次 (记录 {start_idx + 1}-{end_idx})")

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
                                # 使用sanitize_for_json处理值
                                entity[column] = self.sanitize_for_json(value)

                        # 添加向量和文档 - 确保向量也被正确处理
                        entity["embedding"] = self.sanitize_for_json(vectors[i])
                        entity["dense_vector"] = self.sanitize_for_json(vectors[i])
                        entity["document"] = document_texts[i]
                        entity["source_file"] = filename
                        entity["upload_time"] = time.time()

                        # 对整个实体进行最终清理
                        entity = self.sanitize_for_json(entity)
                        batch_entities.append(entity)

                    # 插入数据
                    success = self.vector_client.insert_data(collection_name, batch_entities)
                    if success:
                        inserted_count += len(batch_entities)
                        logger.info(f"任务 {task_id} 批次 {current_batch} 插入成功，记录数: {len(batch_entities)}")

                        # 当前批次完成
                        self.update_progress(task_id, progress_storage, "存储到向量数据库", 5, total_stages,
                                             current_batch, total_batches,
                                             f"第 {current_batch}/{total_batches} 批次处理完成 (记录 {start_idx + 1}-{end_idx})",
                                             batch_completed=True)
                    else:
                        logger.error(f"任务 {task_id} 批次 {current_batch} 插入失败")
                        # 即使失败也要更新进度，但不标记为已完成
                        self.update_progress(task_id, progress_storage, "存储到向量数据库", 5, total_stages,
                                             current_batch, total_batches,
                                             f"第 {current_batch}/{total_batches} 批次处理失败")

                    # 添加小延迟，让前端能看到进度变化
                    await asyncio.sleep(0.1)

                # 获取最终统计信息
                stats = self.vector_client.get_collection_stats(collection_name)

                # 阶段5: 存储到数据库 - 完成（所有阶段完成）
                progress_storage[task_id].update({
                    "status": "completed",
                    "stage": "处理完成",
                    "stage_number": total_stages,
                    "total_stages": total_stages,
                    "current_batch": total_batches,
                    "total_batches": total_batches,
                    "progress_percent": 100,
                    "message": "向量化处理已完成",
                    "result": {
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
