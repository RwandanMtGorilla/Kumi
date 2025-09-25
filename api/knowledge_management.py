from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from services.knowledge_service import KnowledgeService
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import BackgroundTasks
import uuid

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
                collections_info.append({
                    "name": collection_name,
                    "stats": stats,
                    "display_name": collection_name.replace("_", " ").title(),
                    "document_count": stats.get("count", 0),
                    "created_time": stats.get("created_time", "未知"),
                    "last_updated": stats.get("last_updated", "未知")
                })
            except Exception as e:
                logger.warning(f"获取集合 {collection_name} 统计信息失败: {e}")
                collections_info.append({
                    "name": collection_name,
                    "stats": {},
                    "display_name": collection_name.replace("_", " ").title(),
                    "document_count": 0,
                    "created_time": "未知",
                    "last_updated": "未知",
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

        # 验证文件大小 (限制为50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(file_content) > max_size:
            return JSONResponse({
                "status": "error",
                "message": f"文件太大，限制为50MB，当前文件: {len(file_content) / 1024 / 1024:.1f}MB"
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
        batch_size: int = Form(5)
):
    """处理文件上传和向量化（异步处理）"""
    try:
        user = get_user_from_request(request)

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 初始化进度
        progress_storage[task_id] = {
            "status": "starting",
            "stage": "初始化",
            "stage_number": 0,
            "total_stages": 5,
            "current_batch": 0,
            "total_batches": 0,
            "progress_percent": 0,
            "message": "任务已创建，准备开始...",
            "result": None,
            "error": None
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

        # 读取文件内容
        file_content = await file.read()

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
            progress_storage
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
                "template": "Instruct: Given a customer query, retrieve the most relevant chunks needed to answer the query. \nQuery:{question}",
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
