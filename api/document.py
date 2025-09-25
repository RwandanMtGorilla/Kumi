from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from api.auth import verify_api_key
from config.settings import settings
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import os
import logging
import httpx
from markitdown import MarkItDown
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Conversion"])


# 请求模型
class DocumentUrlRequest(BaseModel):
    name: str
    url: str
    enable_plugins: Optional[bool] = False


# 响应模型
class DocumentConversionResponse(BaseModel):
    success: bool
    markdown_content: str
    metadata: Optional[Dict[str, Any]] = None
    original_filename: str
    file_size: int


@router.post("/convert/url", response_model=DocumentConversionResponse)
async def convert_document_from_url(
        request: DocumentUrlRequest,
        api_key: str = Depends(verify_api_key)
):
    """
    从URL下载文件并转换为Markdown格式
    """
    try:
        # 验证文件扩展名
        file_ext = os.path.splitext(request.name)[1].lower()
        if file_ext not in settings.MARKITDOWN_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": 2002,
                    "error_msg": f"不支持的文件格式: {file_ext}"
                }
            )

        # 下载文件
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(request.url)
                response.raise_for_status()
                file_content = response.content
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": 2005,
                        "error_msg": f"下载文件失败: {str(e)}"
                    }
                )
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_code": 2006,
                        "error_msg": f"HTTP错误: {e.response.status_code}"
                    }
                )

        # 验证文件大小
        file_size = len(file_content)
        if file_size > settings.MARKITDOWN_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error_code": 2001,
                    "error_msg": f"文件大小超过限制 ({settings.MARKITDOWN_MAX_FILE_SIZE} bytes)"
                }
            )

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            try:
                # 写入下载的内容到临时文件
                temp_file.write(file_content)
                temp_file.flush()

                # 使用MarkItDown转换
                md = MarkItDown(enable_plugins=request.enable_plugins or settings.MARKITDOWN_ENABLE_PLUGINS)
                result = md.convert(temp_file.name)

                logger.info(f"Successfully converted file from URL: {request.name}")

                return DocumentConversionResponse(
                    success=True,
                    markdown_content=result.text_content,
                    metadata=getattr(result, 'metadata', None),
                    original_filename=request.name,
                    file_size=file_size
                )

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file.name)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document conversion error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": 2003,
                "error_msg": f"文档转换失败: {str(e)}"
            }
        )


@router.get("/convert/supported-formats")
async def get_supported_formats(api_key: str = Depends(verify_api_key)):
    """
    获取支持的文件格式列表
    """
    return {
        "supported_extensions": settings.MARKITDOWN_ALLOWED_EXTENSIONS,
        "max_file_size": settings.MARKITDOWN_MAX_FILE_SIZE,
        "plugins_enabled": settings.MARKITDOWN_ENABLE_PLUGINS
    }
