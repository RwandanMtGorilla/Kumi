from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Bearer Token 安全方案
security = HTTPBearer()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    验证API Key
    """
    try:
        # 检查Authorization头格式
        if not credentials:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": 1001,
                    "error_msg": "无效的 Authorization 头格式。预期格式为 'Bearer <api-key>'。"
                }
            )

        token = credentials.credentials

        # 这里可以实现更复杂的验证逻辑
        # 目前简单验证：检查token是否为预设值或者从数据库验证
        valid_tokens = [
            settings.API_API_KEY,  # 测试token
            settings.OPENAI_API_KEY,  # 使用现有配置中的key作为有效token
        ]

        # 过滤掉None值
        valid_tokens = [t for t in valid_tokens if t is not None]

        if token not in valid_tokens:
            logger.warning(f"Invalid API key attempted: {token[:10]}...")
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": 1002,
                    "error_msg": "授权失败"
                }
            )

        logger.info(f"Valid API key authenticated: {token[:10]}...")
        return token

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": 1002,
                "error_msg": "授权失败"
            }
        )


def validate_knowledge_id(knowledge_id: str) -> bool:
    """
    验证知识库ID是否存在
    """
    # 这里可以实现知识库ID的验证逻辑
    # 目前简单实现：检查ID格式和是否在允许的列表中

    if not knowledge_id or len(knowledge_id) < 3:
        return False

    # 可以从数据库或配置文件中获取有效的知识库ID列表
    # 目前允许所有合理格式的ID
    return True
