from fastapi import APIRouter, Request, Form, HTTPException, status, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 获取项目根目录的绝对路径
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "web" / "templates"

# 模板配置
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 简单的session管理（生产环境建议使用Redis等）
active_sessions: Dict[str, Dict[str, Any]] = {}

# 配置
SESSION_TIMEOUT = 24 * 60 * 60  # 24小时

# 从设置中获取默认用户名密码
from config.settings import settings

DEFAULT_USERNAME = settings.ADMIN_USER_NAME
DEFAULT_PASSWORD = settings.ADMIN_PASSWORD
VERIFY_TOKEN = settings.VERIFY_TOKEN


def create_session_id(username: str, password: str) -> str:
    """创建session ID"""
    timestamp = str(int(time.time()))
    return hashlib.md5(f"{username}_{password}_{timestamp}".encode()).hexdigest()


def is_ajax_request(request: Request) -> bool:
    """检查是否为AJAX请求"""
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


def get_user_from_request(request: Request) -> dict:
    """从request中获取用户信息（由中间件设置）"""
    return getattr(request.state, 'user', {})


def get_template_context(request: Request, active_page: str = "") -> dict:
    """获取模板上下文"""
    user = get_user_from_request(request)
    return {
        "request": request,
        "user": user,
        "active_page": active_page,
        "is_ajax": is_ajax_request(request)
    }


# ==================== 认证相关路由 ====================

@router.get("/", response_class=HTMLResponse)
async def web_root(request: Request):
    """重定向到登录页或主页"""
    # 检查是否已登录（通过检查request.state.user）
    user = get_user_from_request(request)
    if user:
        return RedirectResponse(url="/web/dashboard", status_code=302)
    else:
        return RedirectResponse(url="/web/login", status_code=302)


@router.get("/login")
async def login_with_token(
    request: Request,
    token: str | None = Query(None)
):
    """
    处理 GET /login 的 token 直通登录
    - token 正确：直接创建 session 并跳转
    - token 缺失或错误：仅渲染登录页，不显示任何“密码错误”类提示
    """
    try:
        if token and token == VERIFY_TOKEN:
            session_id = create_session_id("token_login", token)
            active_sessions[session_id] = {
                "username": DEFAULT_USERNAME,
                "created_at": time.time(),
                "last_access": time.time(),
                "user_agent": request.headers.get("user-agent", ""),
                "ip_address": request.client.host if request.client else "unknown",
            }

            logger.info(f"使用 token 登录成功，IP: {request.client.host if request.client else 'unknown'}")

            response = RedirectResponse(url="/web/dashboard", status_code=302)
            response.set_cookie(
                key="session_id",
                value=session_id,
                max_age=SESSION_TIMEOUT,
                httponly=True,
                secure=False,  # 生产环境改为 True
            )
            return response

        # token 未提供或不匹配：直接展示登录页（无错误信息）
        return templates.TemplateResponse("login.html", {
            "request": request,
            "username": ""
        })

    except Exception as e:
        logger.error(f"token 登录处理异常: {str(e)}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "登录处理失败，请稍后重试",
            "username": ""
        })


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    """
    处理 POST /login 的表单登录（用户名/密码）
    """
    try:
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session_id = create_session_id(username, password)
            active_sessions[session_id] = {
                "username": username,
                "created_at": time.time(),
                "last_access": time.time(),
                "user_agent": request.headers.get("user-agent", ""),
                "ip_address": request.client.host if request.client else "unknown",
            }

            logger.info(f"用户 {username} 登录成功，IP: {request.client.host if request.client else 'unknown'}")

            response = RedirectResponse(url="/web/dashboard", status_code=302)
            response.set_cookie(
                key="session_id",
                value=session_id,
                max_age=SESSION_TIMEOUT,
                httponly=True,
                secure=False,  # 生产环境改为 True
            )
            return response

        # 仅在表单登录失败时提示用户名或密码错误
        logger.warning(f"登录失败，用户名: {username}, IP: {request.client.host if request.client else 'unknown'}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "用户名或密码错误",
            "username": username or ""
        })

    except Exception as e:
        logger.error(f"登录处理异常: {str(e)}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "登录处理失败，请稍后重试",
            "username": username or ""
        })


@router.get("/logout")
async def logout(request: Request):
    """登出"""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in active_sessions:
        username = active_sessions[session_id].get("username", "unknown")
        del active_sessions[session_id]
        logger.info(f"用户 {username} 登出")

    response = RedirectResponse(url="/web/login", status_code=302)
    response.delete_cookie(key="session_id")
    return response


# ==================== 主页面路由 ====================
# 注意：以下所有路由都不需要再检查session，因为中间件已经处理了

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """主页面/仪表板"""
    if is_ajax_request(request):
        user = get_user_from_request(request)
        return JSONResponse({
            "status": "success",
            "page": "dashboard",
            "user": user
        })
    else:
        context = get_template_context(request, "dashboard")
        return templates.TemplateResponse("index.html", context)


# ==================== 知识库相关路由 ====================


@router.get("/knowledge/upload", response_class=HTMLResponse)
async def knowledge_upload(request: Request):
    """知识文件上传页面"""
    context = get_template_context(request, "knowledge_upload")
    return templates.TemplateResponse("pages/knowledge/upload.html", context)


@router.get("/knowledge/test", response_class=HTMLResponse)
async def knowledge_test(request: Request):
    """知识库测试页面"""
    context = get_template_context(request, "knowledge_test")
    return templates.TemplateResponse("pages/knowledge/test.html", context)


# ==================== 大模型相关路由 ====================

@router.get("/llm/config", response_class=HTMLResponse)
async def llm_config(request: Request):
    """大模型配置页面"""
    context = get_template_context(request, "llm_config")
    return templates.TemplateResponse("pages/llm/config.html", context)


@router.get("/llm/upload-dataset", response_class=HTMLResponse)
async def llm_upload_dataset(request: Request):
    """测评数据集上传页面"""
    context = get_template_context(request, "llm_upload_dataset")
    return templates.TemplateResponse("pages/llm/upload_dataset.html", context)


@router.get("/llm/upload-rules", response_class=HTMLResponse)
async def llm_upload_rules(request: Request):
    """测评规则上传页面"""
    context = get_template_context(request, "llm_upload_rules")
    return templates.TemplateResponse("pages/llm/upload_rules.html", context)


@router.get("/llm/evaluation", response_class=HTMLResponse)
async def llm_evaluation(request: Request):
    """测评任务页面"""
    context = get_template_context(request, "llm_evaluation")
    return templates.TemplateResponse("pages/llm/evaluation.html", context)


# ==================== API接口路由 ====================

@router.get("/api/session/info")
async def get_session_info(request: Request):
    """获取当前session信息"""
    user = get_user_from_request(request)
    session_id = request.cookies.get("session_id")
    session_data = active_sessions.get(session_id, {})

    return JSONResponse({
        "status": "success",
        "data": {
            "username": user.get("username"),
            "login_time": session_data.get("created_at"),
            "last_access": session_data.get("last_access"),
            "ip_address": session_data.get("ip_address"),
            "user_agent": session_data.get("user_agent")
        }
    })


@router.get("/api/sessions/active")
async def get_active_sessions(request: Request):
    """获取活跃session列表（管理员功能）"""
    user = get_user_from_request(request)

    # 检查管理员权限
    if user.get("username") != DEFAULT_USERNAME:
        raise HTTPException(status_code=403, detail="权限不足")

    sessions = []
    current_time = time.time()

    for session_id, session_data in active_sessions.items():
        sessions.append({
            "session_id": session_id[:8] + "...",
            "username": session_data.get("username"),
            "ip_address": session_data.get("ip_address"),
            "login_time": session_data.get("created_at"),
            "last_access": session_data.get("last_access"),
            "duration": current_time - session_data.get("created_at", current_time)
        })

    return JSONResponse({
        "status": "success",
        "data": {
            "total": len(sessions),
            "sessions": sessions
        }
    })


@router.post("/api/session/refresh")
async def refresh_session(request: Request):
    """刷新session"""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in active_sessions:
        active_sessions[session_id]["last_access"] = time.time()
        return JSONResponse({
            "status": "success",
            "message": "Session已刷新"
        })

    raise HTTPException(status_code=401, detail="Session无效")


# ==================== 工具函数 ====================

def cleanup_expired_sessions():
    """清理过期的session"""
    current_time = time.time()
    expired_sessions = []

    for session_id, session_data in active_sessions.items():
        if current_time - session_data.get("created_at", 0) > SESSION_TIMEOUT:
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        username = active_sessions[session_id].get("username", "unknown")
        del active_sessions[session_id]
        logger.info(f"清理过期session: {username}")

    return len(expired_sessions)


@router.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    logger.info("Web模块启动")


@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("Web模块关闭")
    active_sessions.clear()
