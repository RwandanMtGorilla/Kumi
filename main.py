from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config.settings import settings
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KUMI Knowledge API",
    description="外部知识库API，兼容Dify",
    version="1.0.0"
)

# 获取项目根目录
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "web" / "static"

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加Session中间件
try:
    from api.middleware import SessionMiddleware
    from api.web import active_sessions, SESSION_TIMEOUT

    app.add_middleware(SessionMiddleware, active_sessions=active_sessions, session_timeout=SESSION_TIMEOUT)
    logger.info("Session middleware added")
except ImportError as e:
    logger.warning(f"Failed to import SessionMiddleware: {e}")


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": 500,
            "error_msg": "Internal Server Error"
        }
    )


class CachedStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response: FileResponse = await super().get_response(path, scope)

        if response.status_code == 200:
            # 设置缓存 30 天
            response.headers["Cache-Control"] = "public, max-age=2592000, immutable"
            logger.info(f"Setting Cache-Control for {path}: {response.headers['Cache-Control']}")

        return response

# 挂载静态文件
if STATIC_DIR.exists():
    app.mount("/static", CachedStaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"Static files mounted from: {STATIC_DIR}")
else:
    logger.warning(f"Static directory {STATIC_DIR} does not exist")

# 条件注册路由
enabled_services = []

if settings.ENABLE_KNOWLEDGE_API:
    try:
        from api.knowledge import router as knowledge_router

        app.include_router(knowledge_router, prefix="/api/v1/knowledge")
        enabled_services.append("Knowledge API")
        logger.info("Knowledge API enabled")
    except ImportError as e:
        logger.warning(f"Failed to import knowledge router: {e}")

# 注册知识库测试API
try:
    from api import knowledge_test

    app.include_router(knowledge_test.router, prefix="/api")
    enabled_services.append("Knowledge Test API")
    logger.info("Knowledge Test API enabled")
except ImportError as e:
    logger.warning(f"Failed to import knowledge_test router: {e}")

try:
    from api.knowledge_management import router as knowledge_management_router

    app.include_router(knowledge_management_router, prefix="/web")
    enabled_services.append("Knowledge Management")
    logger.info("Knowledge Management enabled")
except ImportError as e:
    logger.warning(f"Failed to import knowledge_management router: {e}")

if settings.ENABLE_DOCUMENT_CONVERSION_API:
    try:
        from api.document import router as document_router

        app.include_router(document_router, prefix="/api/v1/document")
        enabled_services.append("Document Conversion API")
        logger.info("Document Conversion API enabled")
    except ImportError as e:
        logger.warning(f"Failed to import document router: {e}")

# 注册Web界面路由
try:
    from api.web import router as web_router

    app.include_router(web_router, prefix="/web")
    enabled_services.append("Web Interface")
    logger.info("Web Interface enabled")
except ImportError as e:
    logger.warning(f"Failed to import web router: {e}")

# 注册llm路由
try:
    from api.llm_evaluation import router as llm_api_router

    app.include_router(llm_api_router, prefix="/web/api/llm", tags=["LLM Evaluation API"])
    enabled_services.append("llm Interface")
    logger.info("llm Interface enabled")
except ImportError as e:
    logger.warning(f"Failed to import llm router: {e}")

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("应用启动")

    # 初始化相似度计算器
    try:
        from api import knowledge_test
        knowledge_test.init_similarity_calculator()
        logger.info("相似度计算器初始化完成")
    except Exception as e:
        logger.error(f"相似度计算器初始化失败: {e}")


# 调试端点
@app.get("/debug/static")
async def debug_static():
    """调试静态文件配置"""
    debug_info = {
        "base_dir": str(BASE_DIR),
        "static_dir": str(STATIC_DIR),
        "static_dir_exists": STATIC_DIR.exists(),
        "current_working_dir": str(Path.cwd()),
        "files_in_static": []
    }

    if STATIC_DIR.exists():
        for file in STATIC_DIR.rglob("*"):
            if file.is_file():
                debug_info["files_in_static"].append(str(file.relative_to(STATIC_DIR)))

    return debug_info


# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "KUMI Knowledge API is running",
        "enabled_services": enabled_services
    }


# 服务信息
@app.get("/")
async def root():
    return {
        "name": "KUMI Knowledge API",
        "version": "1.0.0",
        "enabled_services": enabled_services,
        "endpoints": {
            "health": "/health",
            "knowledge": "/api/v1/knowledge" if settings.ENABLE_KNOWLEDGE_API else None,
            "knowledge_test": "/api/knowledge/similarity" if "Knowledge Test API" in enabled_services else None,
            "document": "/api/v1/document" if settings.ENABLE_DOCUMENT_CONVERSION_API else None,
            "web": "/web" if "Web Interface" in enabled_services else None
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
