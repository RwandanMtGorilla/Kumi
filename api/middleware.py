from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionMiddleware(BaseHTTPMiddleware):
    """Session验证中间件"""

    def __init__(self, app, active_sessions: Dict[str, Dict[str, Any]], session_timeout: int = 24 * 60 * 60):
        super().__init__(app)
        self.active_sessions = active_sessions
        self.session_timeout = session_timeout

        # 不需要验证session的路径
        self.public_paths = {
            "/web/login",
            "/web/logout",
            "/health",
            "/",
            "/docs",
            "/openapi.json",
            "/redoc",
        }

        # 不需要验证session的路径前缀
        self.public_prefixes = [
            "/static/",
            "/api/v1/",  # API接口可能有自己的认证机制
            "/debug/",
        ]

    def is_public_path(self, path: str) -> bool:
        """检查是否为公共路径"""
        # 检查完全匹配的路径
        if path in self.public_paths:
            return True

        # 检查路径前缀
        for prefix in self.public_prefixes:
            if path.startswith(prefix):
                return True

        return False

    def is_ajax_request(self, request: Request) -> bool:
        """检查是否为AJAX请求"""
        return request.headers.get("X-Requested-With") == "XMLHttpRequest"

    def verify_session(self, request: Request) -> Dict[str, Any]:
        """验证session"""
        session_id = request.cookies.get("session_id")

        if not session_id or session_id not in self.active_sessions:
            raise HTTPException(status_code=401, detail="未登录或session已过期")

        session_data = self.active_sessions[session_id]

        # 检查session是否过期
        if time.time() - session_data.get("created_at", 0) > self.session_timeout:
            del self.active_sessions[session_id]
            logger.info(f"Session过期，用户: {session_data.get('username', 'unknown')}")
            raise HTTPException(status_code=401, detail="Session已过期，请重新登录")

        # 更新最后访问时间
        session_data["last_access"] = time.time()

        return session_data

    async def dispatch(self, request: Request, call_next):
        """中间件处理逻辑"""
        path = request.url.path

        # 如果是公共路径，直接通过
        if self.is_public_path(path):
            return await call_next(request)

        # 对于需要验证的web页面路径
        if path.startswith("/web/"):
            try:
                # 验证session
                session_data = self.verify_session(request)

                # 将用户信息添加到request state中，供后续使用
                request.state.user = session_data

                return await call_next(request)

            except HTTPException:
                # Session验证失败
                if self.is_ajax_request(request):
                    # AJAX请求返回JSON错误
                    return JSONResponse(
                        status_code=401,
                        content={"error": "未登录或session已过期", "redirect": "/web/login"}
                    )
                else:
                    # 普通请求重定向到登录页
                    return RedirectResponse(url="/web/login", status_code=302)

        # 其他路径直接通过
        return await call_next(request)
