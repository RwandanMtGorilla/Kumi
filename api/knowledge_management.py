from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from services.knowledge_service import KnowledgeService
from config.settings import settings
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import BackgroundTasks
import uuid
from io import BytesIO
from openpyxl import Workbook

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 获取项目根目录的绝对路径
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"

# 模板配置
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 初始化知识库服务
knowledge_service = KnowledgeService()


def get_user_from_request(request: Request) -> dict:
    """从request中获取用户信息（由中间件设置）"""
    return getattr(request.state, 'user', {})


def is_ajax_request(request: Request) -> bool:
    """检查是否为AJAX请求"""
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def get_template_context(request: Request, active_page: str = "") -> dict:
    """获取模板上下文"""
    user = get_user_from_request(request)
    return {
        "request": request,
        "user": user,
        "active_page": active_page,
        "is_ajax": is_ajax_request(request)
    }


@router.get("/knowledge", response_class=HTMLResponse)
async def knowledge_dashboard(request: Request):
    """知识库主页面"""
    try:
        context = get_template_context(request, "knowledge")
        return templates.TemplateResponse("pages/knowledge/dashboard.html", context)
    except Exception as e:
        logger.error(f"知识库主页面加载失败: {e}")
        raise HTTPException(status_code=500, detail="页面加载失败")


@router.get("/api/knowledge/collections")
async def get_collections(request: Request):
    """获取所有知识库集合"""
    try:
        # 用户信息从中间件获取，这里可以记录日志
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求获取知识库集合列表")

        collections = await knowledge_service.list_collections()

        # 获取每个集合的详细信息
        collections_info = []
        for collection_name in collections:
            try:
                stats = await knowledge_service.get_collection_stats(collection_name)
                metadata = stats.get("metadata", {})

                # 解析 source_files 为列表（用于前端文件名匹配）
                source_files_str = metadata.get("source_files", "")
                source_files_list = [f.strip() for f in source_files_str.split(",") if f.strip()]

                # 获取 user_description，如果不存在则使用 display_name 作为默认值
                default_display = collection_name.replace("_", " ").title()
                user_description = metadata.get("user_description", default_display)

                collections_info.append({
                    "name": collection_name,
                    "stats": stats,
                    "display_name": default_display,
                    "user_description": user_description,
                    "document_count": stats.get("row_count", stats.get("count", 0)),
                    "created_time": stats.get("created_time", "未知"),
                    "last_updated": stats.get("last_updated", "未知"),
                    "source_files": source_files_list,
                    "embedding_template": metadata.get("embedding_template", ""),
                    "document_template": metadata.get("document_template", "")
                })
            except Exception as e:
                logger.warning(f"获取集合 {collection_name} 统计信息失败: {e}")
                default_display = collection_name.replace("_", " ").title()
                collections_info.append({
                    "name": collection_name,
                    "stats": {},
                    "display_name": default_display,
                    "user_description": default_display,
                    "document_count": 0,
                    "created_time": "未知",
                    "last_updated": "未知",
                    "source_files": [],
                    "embedding_template": "",
                    "document_template": "",
                    "error": str(e)
                })

        return JSONResponse({
            "status": "success",
            "data": {
                "collections": collections_info,
                "total": len(collections_info)
            }
        })

    except Exception as e:
        logger.error(f"获取知识库集合失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取知识库集合失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/collections/{collection_name}")
async def get_collection_details(collection_name: str, request: Request):
    """获取单个知识库集合的详细信息"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求获取知识库 {collection_name} 详情")

        stats = await knowledge_service.get_collection_stats(collection_name)

        return JSONResponse({
            "status": "success",
            "data": {
                "name": collection_name,
                "stats": stats,
                "display_name": collection_name.replace("_", " ").title()
            }
        })

    except Exception as e:
        logger.error(f"获取知识库集合详情失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取知识库详情失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/collections/{collection_name}/metadata")
async def get_collection_metadata(collection_name: str, request: Request):
    """获取单个知识库集合的 metadata（用于增量上传时读取模板等信息）"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求获取知识库 {collection_name} 的 metadata")

        stats = await knowledge_service.get_collection_stats(collection_name)

        if "error" in stats:
            return JSONResponse({
                "status": "error",
                "message": f"知识库 '{collection_name}' 不存在或无法访问"
            }, status_code=404)

        metadata = stats.get("metadata", {})

        # 解析 source_files 为列表
        source_files_str = metadata.get("source_files", "")
        source_files_list = [f.strip() for f in source_files_str.split(",") if f.strip()]

        return JSONResponse({
            "status": "success",
            "data": {
                "name": collection_name,
                "metadata": {
                    "embedding_template": metadata.get("embedding_template", ""),
                    "document_template": metadata.get("document_template", ""),
                    "source_files": source_files_list,
                    "embedding_dimension": metadata.get("embedding_dimension"),
                    "description": metadata.get("description", ""),
                    "user_description": metadata.get("user_description", collection_name.replace("_", " ").title())
                }
            }
        })

    except Exception as e:
        logger.error(f"获取知识库 metadata 失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取 metadata 失败: {str(e)}"
        }, status_code=500)


@router.patch("/api/knowledge/collections/{collection_name}/metadata")
async def update_collection_metadata(collection_name: str, request: Request):
    """更新知识库集合的 metadata（如描述等）"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求更新知识库 {collection_name} 的 metadata")

        # 解析请求体
        body = await request.json()

        # 只允许更新特定字段（user_description）
        allowed_fields = {"user_description"}
        update_data = {k: v for k, v in body.items() if k in allowed_fields}

        if not update_data:
            return JSONResponse({
                "status": "error",
                "message": "没有可更新的字段，允许的字段: user_description"
            }, status_code=400)

        # 调用 vector_client 更新 metadata
        success = knowledge_service.vector_client.update_collection_metadata(collection_name, update_data)

        if success:
            return JSONResponse({
                "status": "success",
                "message": "metadata 更新成功",
                "data": {"updated_fields": list(update_data.keys())}
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "更新 metadata 失败"
            }, status_code=500)

    except Exception as e:
        logger.error(f"更新知识库 metadata 失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"更新 metadata 失败: {str(e)}"
        }, status_code=500)


@router.delete("/api/knowledge/collections/{collection_name}")
async def delete_collection(collection_name: str, request: Request):
    """删除知识库集合"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求删除知识库 {collection_name}")


        # 临时实现 - 直接调用向量数据库客户端
        success = knowledge_service.vector_client.delete_collection(collection_name)

        if success:
            logger.info(f"用户 {user.get('username', 'unknown')} 成功删除知识库 {collection_name}")
            return JSONResponse({
                "status": "success",
                "message": f"知识库 '{collection_name}' 删除成功"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"知识库 '{collection_name}' 删除失败"
            }, status_code=500)

    except Exception as e:
        logger.error(f"删除知识库集合失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"删除失败: {str(e)}"
        }, status_code=500)


@router.post("/api/knowledge/collections/{collection_name}/documents")
async def add_documents_to_collection(collection_name: str, request: Request):
    """向知识库集合添加文档"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求向知识库 {collection_name} 添加文档")

        # 获取请求体数据
        data = await request.json()
        documents = data.get("documents", [])

        if not documents:
            return JSONResponse({
                "status": "error",
                "message": "没有提供文档数据"
            }, status_code=400)

        # 调用知识库服务添加文档
        result = await knowledge_service.add_documents_to_collection(collection_name, documents)

        return JSONResponse({
            "status": "success",
            "message": f"成功添加 {len(documents)} 个文档到知识库 '{collection_name}'",
            "data": result
        })

    except Exception as e:
        logger.error(f"添加文档到知识库失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"添加文档失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/collections/{collection_name}/search")
async def search_in_collection(collection_name: str, request: Request):
    """在知识库集合中搜索"""
    try:
        user = get_user_from_request(request)

        # 获取查询参数
        query = request.query_params.get("q", "")
        limit = int(request.query_params.get("limit", 10))

        if not query:
            return JSONResponse({
                "status": "error",
                "message": "搜索查询不能为空"
            }, status_code=400)

        logger.info(f"用户 {user.get('username', 'unknown')} 在知识库 {collection_name} 中搜索: {query}")

        # 调用知识库服务进行搜索
        results = await knowledge_service.search_in_collection(collection_name, query, limit)

        return JSONResponse({
            "status": "success",
            "data": {
                "query": query,
                "collection": collection_name,
                "results": results,
                "total": len(results)
            }
        })

    except Exception as e:
        logger.error(f"在知识库中搜索失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"搜索失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/collections/{collection_name}/documents")
async def get_collection_documents(collection_name: str, request: Request):
    """获取知识库集合中的文档列表"""
    try:
        user = get_user_from_request(request)

        # 获取分页参数
        page = int(request.query_params.get("page", 1))
        page_size = int(request.query_params.get("page_size", 20))

        logger.info(f"用户 {user.get('username', 'unknown')} 请求获取知识库 {collection_name} 的文档列表")

        # 调用知识库服务获取文档列表
        documents = await knowledge_service.get_collection_documents(collection_name, page, page_size)

        return JSONResponse({
            "status": "success",
            "data": {
                "collection": collection_name,
                "documents": documents.get("documents", []),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": documents.get("total", 0),
                    "pages": documents.get("pages", 1)
                }
            }
        })

    except Exception as e:
        logger.error(f"获取知识库文档列表失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取文档列表失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/collections/{collection_name}/export")
async def export_collection_to_excel(collection_name: str, request: Request):
    """导出知识库为 xlsx 文件"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求导出知识库 {collection_name}")

        # 获取所有文档（不分页）
        all_data = knowledge_service.vector_client.get_all_data(collection_name, limit=None)

        # 排除字段
        exclude_fields = {'id', 'upload_time', 'embedding', 'dense_vector'}

        # 创建工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "知识库数据"

        if all_data:
            # 获取字段（排除系统字段）
            fields = [k for k in all_data[0].keys() if k not in exclude_fields]

            # 写表头
            ws.append(fields)

            # 写数据
            for doc in all_data:
                row = [str(doc.get(f, '')) if doc.get(f) is not None else '' for f in fields]
                ws.append(row)

        # 保存到内存
        output = BytesIO()
        wb.save(output)
        output.seek(0)

        # 对文件名进行编码处理
        safe_filename = collection_name.encode('utf-8').decode('utf-8')

        logger.info(f"用户 {user.get('username', 'unknown')} 成功导出知识库 {collection_name}，共 {len(all_data) if all_data else 0} 条记录")

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}.xlsx"}
        )
    except Exception as e:
        logger.error(f"导出知识库失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"导出失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/collections/{collection_name}/documents/{document_id}")
async def get_document_detail(collection_name: str, document_id: str, request: Request):
    """获取单个文档的详细信息"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求获取文档 {document_id}")

        document = await knowledge_service.get_document_by_id(collection_name, document_id)

        if document:
            return JSONResponse({
                "status": "success",
                "data": document
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "文档不存在"
            }, status_code=404)

    except Exception as e:
        logger.error(f"获取文档详情失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取文档失败: {str(e)}"
        }, status_code=500)


@router.put("/api/knowledge/collections/{collection_name}/documents/{document_id}")
async def update_document(collection_name: str, document_id: str, request: Request):
    """更新文档内容和元数据"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求更新文档 {document_id}")

        # 获取请求体
        data = await request.json()

        document_content = data.get("document", "")
        metadata = data.get("metadata", {})
        re_vectorize = data.get("re_vectorize", False)
        embedding_template = data.get("embedding_template")
        embedding_model = data.get("embedding_model")  # "provider,model" 格式

        # 解析 embedding 模型
        embedding_provider = None
        embedding_model_name = None
        if embedding_model and re_vectorize:
            try:
                embedding_provider, embedding_model_name = \
                    knowledge_service.embedding_service.parse_model_identifier(embedding_model)
            except Exception as e:
                return JSONResponse({
                    "status": "error",
                    "message": f"无效的embedding模型: {str(e)}"
                }, status_code=400)

        result = await knowledge_service.update_document_in_collection(
            collection_name=collection_name,
            document_id=document_id,
            document_content=document_content,
            metadata=metadata,
            re_vectorize=re_vectorize,
            embedding_template=embedding_template,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model_name
        )

        if result["success"]:
            return JSONResponse({
                "status": "success",
                "message": result["message"]
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": result["message"]
            }, status_code=500)

    except Exception as e:
        logger.error(f"更新文档失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"更新失败: {str(e)}"
        }, status_code=500)


@router.delete("/api/knowledge/collections/{collection_name}/documents/{document_id}")
async def delete_document_from_collection(collection_name: str, document_id: str, request: Request):
    """从知识库集合中删除文档"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 请求从知识库 {collection_name} 删除文档 {document_id}")

        # 调用知识库服务删除文档
        success = await knowledge_service.delete_document_from_collection(collection_name, document_id)

        if success:
            return JSONResponse({
                "status": "success",
                "message": f"成功从知识库 '{collection_name}' 删除文档 '{document_id}'"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": f"删除文档失败"
            }, status_code=500)

    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"删除文档失败: {str(e)}"
        }, status_code=500)

#####################

from fastapi import UploadFile, File, Form
import time

@router.post("/api/knowledge/upload/preview")
async def preview_upload_file(request: Request, file: UploadFile = File(...)):
    """预览上传的文件"""
    try:
        user = get_user_from_request(request)
        logger.info(f"用户 {user.get('username', 'unknown')} 预览文件: {file.filename}")

        # 读取文件内容
        file_content = await file.read()

        # 验证文件大小
        max_size = settings.KNOWLEDGE_UPLOAD_MAX_FILE_SIZE
        max_size_mb = max_size / 1024 / 1024
        if len(file_content) > max_size:
            return JSONResponse({
                "status": "error",
                "message": f"文件太大，限制为{max_size_mb:.0f}MB，当前文件: {len(file_content) / 1024 / 1024:.1f}MB"
            }, status_code=400)

        # 验证文件类型
        allowed_extensions = ['.xlsx', '.xls', '.csv', '.json']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return JSONResponse({
                "status": "error",
                "message": f"不支持的文件类型: {file_ext}，支持的类型: {', '.join(allowed_extensions)}"
            }, status_code=400)

        # 获取文件预览
        preview_result = await knowledge_service.get_file_preview(file_content, file.filename)

        if preview_result["success"]:
            return JSONResponse({
                "status": "success",
                "data": preview_result["data"]
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": preview_result["message"]
            }, status_code=400)

    except Exception as e:
        logger.error(f"文件预览失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"预览失败: {str(e)}"
        }, status_code=500)


progress_storage = {}


@router.post("/api/knowledge/upload/process")
async def process_upload_file(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        embedding_template: str = Form(...),
        document_template: str = Form(...),
        collection_name: str = Form(...),
        batch_size: int = Form(5),
        num_workers: int = Form(1),
        embedding_model: str = Form(None),  # "provider,model" 格式
        enable_incremental: bool = Form(True),  # 是否启用增量上传
        dedup_field: str = Form(None),  # 去重字段名（可选）
        insert_batch_multiplier: int = Form(10),  # 数据库插入倍率
        enable_column_update: bool = Form(False)  # 是否启用列更新
):
    """处理文件上传和向量化（异步处理，支持增量上传）"""
    try:
        user = get_user_from_request(request)

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 初始化进度
        progress_storage[task_id] = {
            "status": "starting",
            "stage": "初始化",
            "stage_number": 0,
            "total_stages": 5,  # 阶段5合并了向量化和存储
            "current_batch": 0,
            "total_batches": 0,
            "progress_percent": 0,
            "message": "任务已创建，准备开始...",
            "result": None,
            "error": None,
            "skipped_duplicates": 0  # 跳过的重复记录数
        }

        logger.info(f"用户 {user.get('username', 'unknown')} 创建向量化任务: {task_id}")

        # 验证参数
        if not embedding_template.strip():
            return JSONResponse({
                "status": "error",
                "message": "embedding模板不能为空"
            }, status_code=400)

        if not document_template.strip():
            return JSONResponse({
                "status": "error",
                "message": "document模板不能为空"
            }, status_code=400)

        if not collection_name.strip():
            return JSONResponse({
                "status": "error",
                "message": "collection名称不能为空"
            }, status_code=400)

        if batch_size < 1 or batch_size > 1000:
            return JSONResponse({
                "status": "error",
                "message": "批处理大小必须在1-1000之间"
            }, status_code=400)

        if num_workers < 1 or num_workers > 16:
            return JSONResponse({
                "status": "error",
                "message": "并发Worker数必须在1-16之间"
            }, status_code=400)

        if insert_batch_multiplier < 1 or insert_batch_multiplier > 100:
            return JSONResponse({
                "status": "error",
                "message": "数据库插入倍率必须在1-100之间"
            }, status_code=400)

        # 解析embedding模型
        embedding_provider = None
        embedding_model_name = None
        if embedding_model:
            try:
                embedding_provider, embedding_model_name = knowledge_service.embedding_service.parse_model_identifier(embedding_model)
                logger.info(f"任务 {task_id} 使用embedding模型: {embedding_provider},{embedding_model_name}")
            except Exception as e:
                return JSONResponse({
                    "status": "error",
                    "message": f"无效的embedding模型标识符: {str(e)}"
                }, status_code=400)

        # 读取文件内容
        file_content = await file.read()

        # 处理 dedup_field（空字符串转为 None）
        dedup_field_value = dedup_field.strip() if dedup_field and dedup_field.strip() else None

        logger.info(f"任务 {task_id} 增量上传配置: enable_incremental={enable_incremental}, dedup_field={dedup_field_value}, enable_column_update={enable_column_update}")

        # 后台异步处理
        background_tasks.add_task(
            knowledge_service.process_and_vectorize_file_async,
            task_id,
            file_content,
            file.filename,
            embedding_template,
            document_template,
            collection_name,
            batch_size,
            num_workers,
            progress_storage,
            embedding_provider,
            embedding_model_name,
            enable_incremental,
            dedup_field_value,
            insert_batch_multiplier,
            enable_column_update
        )

        return JSONResponse({
            "status": "success",
            "message": "任务已创建，正在后台处理",
            "data": {
                "task_id": task_id
            }
        })

    except Exception as e:
        logger.error(f"创建向量化任务失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"创建任务失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/upload/progress/{task_id}")
async def get_upload_progress(request: Request, task_id: str):
    """获取上传处理进度"""
    try:
        user = get_user_from_request(request)

        if task_id not in progress_storage:
            return JSONResponse({
                "status": "error",
                "message": "任务不存在"
            }, status_code=404)

        progress_info = progress_storage[task_id]

        return JSONResponse({
            "status": "success",
            "data": progress_info
        })

    except Exception as e:
        logger.error(f"获取进度失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取进度失败: {str(e)}"
        }, status_code=500)


@router.delete("/api/knowledge/upload/progress/{task_id}")
async def clear_upload_progress(request: Request, task_id: str):
    """清理任务进度信息"""
    try:
        user = get_user_from_request(request)

        if task_id in progress_storage:
            del progress_storage[task_id]

        return JSONResponse({
            "status": "success",
            "message": "进度信息已清理"
        })

    except Exception as e:
        logger.error(f"清理进度失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"清理进度失败: {str(e)}"
        }, status_code=500)



@router.get("/api/knowledge/templates")
async def get_embedding_templates(request: Request):
    """获取预设的embedding模板"""
    try:
        user = get_user_from_request(request)

        # 预设模板
        templates = [
            {
                "name": "常规模板",
                "description": "适用于多数数据集",
                "template": "{Text}",
                "required_fields": ["Text"]
            },
            {
                "name": "问题检索模板",
                "description": "适用于问题检索场景",
                "template": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{question}",
                "required_fields": ["question"]
            },
            {
                "name": "自定义模板",
                "description": "用户自定义模板",
                "template": "",
                "required_fields": []
            }
        ]

        return JSONResponse({
            "status": "success",
            "data": {
                "templates": templates
            }
        })

    except Exception as e:
        logger.error(f"获取模板失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取模板失败: {str(e)}"
        }, status_code=500)


@router.get("/api/knowledge/embedding/models")
async def get_embedding_models(request: Request):
    """获取所有可用的embedding模型"""
    try:
        user = get_user_from_request(request)

        # 获取模型列表
        models = knowledge_service.embedding_service.get_available_models()
        default_model = knowledge_service.embedding_service.get_default_model()

        return JSONResponse({
            "status": "success",
            "data": {
                "models": models,
                "default": default_model
            }
        })

    except Exception as e:
        logger.error(f"获取embedding模型列表失败: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"获取模型列表失败: {str(e)}"
        }, status_code=500)


@router.post("/api/knowledge/embedding/test")
async def test_embedding_connection(request: Request):
    """测试embedding API连接"""
    try:
        user = get_user_from_request(request)

        # 获取请求参数
        data = await request.json()
        model_identifier = data.get("model")  # "provider,model" 格式

        if not model_identifier:
            logger.warning("测试embedding连接请求缺少模型标识符")
            return JSONResponse({
                "status": "error",
                "message": "请提供模型标识符"
            }, status_code=400)

        # 解析模型标识符
        try:
            provider, model = knowledge_service.embedding_service.parse_model_identifier(model_identifier)
        except Exception as e:
            logger.warning(f"无效的模型标识符 '{model_identifier}': {e}")
            return JSONResponse({
                "status": "error",
                "message": f"无效的模型标识符: {str(e)}"
            }, status_code=400)

        # 测试连接
        logger.info(f"用户 {user.get('username', 'unknown')} 测试embedding连接: {provider},{model}")
        result = knowledge_service.embedding_service.test_connection(provider, model)

        if result["success"]:
            logger.info(f"用户 {user.get('username', 'unknown')} 测试embedding连接成功: {provider},{model}, 维度: {result['dimension']}")
            return JSONResponse({
                "status": "success",
                "message": result["message"],
                "data": {
                    "dimension": result["dimension"],
                    "provider": provider,
                    "model": model
                }
            })
        else:
            logger.warning(f"用户 {user.get('username', 'unknown')} 测试embedding连接失败: {provider},{model}, 原因: {result['message']}")
            return JSONResponse({
                "status": "error",
                "message": result["message"]
            }, status_code=500)

    except Exception as e:
        logger.error(f"测试embedding连接时发生异常: {e}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "message": f"测试连接失败: {str(e)}"
        }, status_code=500)
