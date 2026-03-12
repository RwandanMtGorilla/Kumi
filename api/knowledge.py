from fastapi import APIRouter, Depends, HTTPException
from .models import RetrievalRequest, RetrievalResponse, ErrorResponse
from .auth import verify_api_key, validate_knowledge_id
from services.knowledge_service import KnowledgeService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Knowledge"])
knowledge_service = KnowledgeService()


@router.post(
    "/retrieval",
    response_model=RetrievalResponse,
    responses={
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        404: {"model": ErrorResponse}
    },
    summary="知识库检索",
    description="根据查询文本检索相关的知识库内容"
)
async def retrieval(
        request: RetrievalRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    知识库检索API - 兼容Dify外部知识库接口
    """
    try:
        logger.info(f"收到检索请求: knowledge_id={request.knowledge_id}, query='{request.query[:50]}...'")

        # 验证知识库ID
        if not validate_knowledge_id(request.knowledge_id):
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": 2001,
                    "error_msg": "知识库不存在"
                }
            )

        # 调用服务层处理
        result = await knowledge_service.search(request)

        logger.info(f"检索完成: 返回 {len(result.records)} 条记录")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索过程中发生错误: {e}")
        error_msg = str(e)
        # 检查是否是知识库不存在的错误
        if "不存在" in error_msg and "知识库" in error_msg:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": 2001,
                    "error_msg": error_msg
                }
            )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": 500,
                "error_msg": f"内部服务器错误: {error_msg}"
            }
        )


@router.get("/collections", summary="列出所有知识库")
async def list_collections(api_key: str = Depends(verify_api_key)):
    """
    列出所有可用的知识库集合
    """
    try:
        collections = await knowledge_service.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"获取集合列表失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": 500,
                "error_msg": f"获取集合列表失败: {str(e)}"
            }
        )


@router.get("/collections/{collection_name}/stats", summary="获取知识库统计信息")
async def get_collection_stats(
        collection_name: str,
        api_key: str = Depends(verify_api_key)
):
    """
    获取指定知识库的统计信息
    """
    try:
        stats = await knowledge_service.get_collection_stats(collection_name)
        return stats
    except Exception as e:
        logger.error(f"获取集合统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": 500,
                "error_msg": f"获取统计信息失败: {str(e)}"
            }
        )


from services.similarity_service import SimilarityCalculator
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from config.settings import settings
import time

# 全局相似度计算器实例
similarity_calculator = None

# Pydantic模型定义
class SimilarityRequest(BaseModel):
    x_collection: str = Field(..., description="X轴collection名称")
    y_collection: str = Field(..., description="Y轴collection名称")
    x_max_items: Optional[int] = Field(100, description="X轴最大项目数")
    y_max_items: Optional[int] = Field(100, description="Y轴最大项目数")
    max_items: Optional[int] = Field(100, description="最大项目数（向后兼容）")

def init_similarity_calculator():
    """初始化相似度计算器"""
    global similarity_calculator
    try:
        similarity_calculator = SimilarityCalculator()
        logger.info("✅ 相似度计算器初始化成功")
    except Exception as e:
        logger.error(f"❌ 相似度计算器初始化失败: {e}")
        similarity_calculator = None

@router.get("/similarity/health")
async def similarity_health_check():
    """相似度服务健康检查"""
    try:
        calculator_ready = similarity_calculator is not None
        connection_info = {}

        if similarity_calculator:
            connection_info = similarity_calculator.test_connection()

        return {
            "status": "healthy",
            "calculator_ready": calculator_ready,
            "timestamp": time.time(),
            "connections": connection_info,
            "vector_db_type": settings.vector_db_type
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "calculator_ready": False,
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.get("/similarity/collections")
async def get_similarity_collections():
    """获取所有collections"""
    try:
        if not similarity_calculator:
            init_similarity_calculator()
            if not similarity_calculator:
                raise HTTPException(status_code=500, detail="相似度计算器未初始化")

        collections = similarity_calculator.get_collections()
        return {
            "success": True,
            "collections": collections,
            "count": len(collections),
            "vector_db_type": settings.vector_db_type
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )

@router.post("/similarity/calculate")
async def calculate_similarity_matrix(request: SimilarityRequest):
    """计算相似度矩阵"""
    try:
        if not similarity_calculator:
            init_similarity_calculator()
            if not similarity_calculator:
                raise HTTPException(status_code=500, detail="相似度计算器未初始化")

        # 解析参数
        x_collection = request.x_collection
        y_collection = request.y_collection
        x_max_items = int(request.x_max_items or request.max_items or 100)
        y_max_items = int(request.y_max_items or request.max_items or 100)

        # 限制最大值
        x_max_items = min(x_max_items, 3000)
        y_max_items = min(y_max_items, 3000)

        logger.info(f"🎯 收到相似度计算请求:")
        logger.info(f"   X: {x_collection} (最大{x_max_items}项)")
        logger.info(f"   Y: {y_collection} (最大{y_max_items}项)")

        # 计算相似度矩阵
        result = similarity_calculator.calculate_similarity_matrix(
            x_collection=x_collection,
            y_collection=y_collection,
            x_max_items=x_max_items,
            y_max_items=y_max_items
        )

        return {
            "success": True,
            "result": result,
            "message": f"成功计算 {len(result['y_data'])} x {len(result['x_data'])} 相似度矩阵",
            "vector_db_type": settings.vector_db_type
        }

    except Exception as e:
        logger.error(f"❌ 相似度计算API失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "error": str(e)}
        )
