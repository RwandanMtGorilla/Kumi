# llm/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Union
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """聊天消息类"""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None


@dataclass
class ChatResponse:
    """聊天响应类"""
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None


class LLMClientInterface(ABC):
    """LLM客户端接口"""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """单次聊天"""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式聊天"""
        pass

    def simple_chat(self, prompt: str, **kwargs) -> str:
        """简单聊天接口"""
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.chat(messages, **kwargs)
        return response.content

    def _format_messages(self, messages: List[Union[ChatMessage, Dict[str, str]]]) -> List[Dict[str, str]]:
        """格式化消息为OpenAI格式"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_msg = {"role": msg.role, "content": msg.content}
                if msg.name:
                    formatted_msg["name"] = msg.name
            elif isinstance(msg, dict):
                formatted_msg = msg
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")
            formatted_messages.append(formatted_msg)
        return formatted_messages

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""
        pass
