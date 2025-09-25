# llm/exceptions.py

class LLMException(Exception):
    """LLM基础异常类"""
    pass


class LLMAPIException(LLMException):
    """LLM API调用异常"""
    pass


class LLMConfigException(LLMException):
    """LLM配置异常"""
    pass


class LLMRateLimitException(LLMException):
    """LLM限流异常"""
    pass


class LLMAuthenticationException(LLMException):
    """LLM认证异常"""
    pass
