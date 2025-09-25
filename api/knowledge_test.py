from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging

from services.similarity_service import SimilarityCalculator
from config.settings import settings

# 创建路由器
router = APIRouter(prefix="/knowledge/similarity", tags=["knowledge-test"])

# 设置日志
logger = logging.getLogger(__name__)

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

@router.get("/health")
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

@router.get("/collections")
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

@router.post("/calculate")
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
