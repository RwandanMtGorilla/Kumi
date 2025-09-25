# llm/openai_client.py

import time
from typing import List, Dict, Any, Optional, Generator, Union
from openai import OpenAI
from .base import LLMClientInterface, ChatMessage, ChatResponse
from .exceptions import LLMAPIException, LLMConfigException, LLMAuthenticationException
from config.settings import settings


class OpenAIClient(LLMClientInterface):
    """OpenAI客户端实现"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.default_model = kwargs.get('model', settings.DEFAULT_MODEL)
        self.default_temperature = kwargs.get('temperature', settings.DEFAULT_TEMPERATURE)
        self.default_max_tokens = kwargs.get('max_tokens', settings.DEFAULT_MAX_TOKENS)

        if not self.api_key:
            raise LLMConfigException("OpenAI API key is required")

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except Exception as e:
            raise LLMConfigException(f"Failed to initialize OpenAI client: {e}")

        print(f"✅ OpenAI客户端初始化成功")
        print(f"   Base URL: {self.base_url or 'https://api.openai.com'}")
        print(f"   默认模型: {self.default_model}")

    def _get_token_param_name(self, model: str) -> str:
        """
        根据模型确定使用哪个token参数
        新模型使用 max_completion_tokens，旧模型使用 max_tokens
        """
        # GPT-4o 系列和较新的模型使用 max_completion_tokens
        newer_models = [
            'gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024', 'chatgpt-4o-latest',
            'gpt-4-turbo', 'gpt-4-turbo-2024-04-09'
        ]

        # 检查是否是新模型
        for newer_model in newer_models:
            if newer_model in model.lower():
                return 'max_completion_tokens'

        # 默认使用旧参数
        return 'max_tokens'

    def _prepare_chat_params(
            self,
            messages: List[Dict[str, str]],
            model: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """准备聊天参数，自动选择正确的token参数"""
        chat_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            **kwargs
        }

        # 设置token限制参数
        if max_tokens is not None or self.default_max_tokens:
            token_param = self._get_token_param_name(model)
            chat_params[token_param] = max_tokens or self.default_max_tokens

        return chat_params

    def chat(
            self,
            messages: List[Union[ChatMessage, Dict[str, str]]],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> ChatResponse:
        """单次聊天"""
        try:
            start_time = time.time()

            # 格式化消息
            formatted_messages = self._format_messages(messages)

            # 确定使用的模型
            used_model = model or self.default_model

            # 准备聊天参数
            chat_params = self._prepare_chat_params(
                formatted_messages,
                used_model,
                temperature,
                max_tokens,
                **kwargs
            )

            # 调用OpenAI API
            response = self.client.chat.completions.create(**chat_params)

            end_time = time.time()

            # 构造响应
            return ChatResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.model_dump() if response.usage else {},
                finish_reason=response.choices[0].finish_reason,
                response_time=end_time - start_time
            )

        except Exception as e:
            error_str = str(e).lower()
            if "authentication" in error_str:
                raise LLMAuthenticationException(f"OpenAI authentication failed: {e}")
            elif "rate limit" in error_str:
                raise LLMAPIException(f"OpenAI rate limit exceeded: {e}")
            elif "unsupported parameter" in error_str and "max_tokens" in error_str:
                # 如果遇到max_tokens参数错误，尝试重新调用
                print("⚠️ 检测到max_tokens参数不支持，尝试使用max_completion_tokens...")
                try:
                    # 重新格式化消息
                    formatted_messages = self._format_messages(messages)
                    used_model = model or self.default_model

                    # 强制使用max_completion_tokens
                    chat_params = {
                        "model": used_model,
                        "messages": formatted_messages,
                        "temperature": temperature if temperature is not None else self.default_temperature,
                        **kwargs
                    }

                    if max_tokens is not None or self.default_max_tokens:
                        chat_params["max_completion_tokens"] = max_tokens or self.default_max_tokens

                    response = self.client.chat.completions.create(**chat_params)
                    end_time = time.time()

                    return ChatResponse(
                        content=response.choices[0].message.content,
                        model=response.model,
                        usage=response.usage.model_dump() if response.usage else {},
                        finish_reason=response.choices[0].finish_reason,
                        response_time=end_time - start_time
                    )
                except Exception as retry_e:
                    raise LLMAPIException(f"OpenAI API error (retry failed): {retry_e}")
            else:
                raise LLMAPIException(f"OpenAI API error: {e}")

    def chat_stream(
            self,
            messages: List[Union[ChatMessage, Dict[str, str]]],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            **kwargs
    ) -> Generator[str, None, None]:
        """流式聊天"""
        try:
            # 格式化消息
            formatted_messages = self._format_messages(messages)

            # 确定使用的模型
            used_model = model or self.default_model

            # 准备聊天参数
            chat_params = self._prepare_chat_params(
                formatted_messages,
                used_model,
                temperature,
                max_tokens,
                stream=True,
                **kwargs
            )

            # 调用OpenAI API
            stream = self.client.chat.completions.create(**chat_params)

            for chunk in stream:
                # 安全检查：确保chunk有choices且不为空
                if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    # 检查delta是否存在且有content
                    if hasattr(choice, 'delta') and hasattr(choice.delta,
                                                            'content') and choice.delta.content is not None:
                        yield choice.delta.content

        except Exception as e:
            error_str = str(e).lower()
            if "authentication" in error_str:
                raise LLMAuthenticationException(f"OpenAI authentication failed: {e}")
            elif "rate limit" in error_str:
                raise LLMAPIException(f"OpenAI rate limit exceeded: {e}")
            elif "unsupported parameter" in error_str and "max_tokens" in error_str:
                # 如果遇到max_tokens参数错误，尝试重新调用
                print("⚠️ 检测到max_tokens参数不支持，尝试使用max_completion_tokens...")
                try:
                    formatted_messages = self._format_messages(messages)
                    used_model = model or self.default_model

                    chat_params = {
                        "model": used_model,
                        "messages": formatted_messages,
                        "temperature": temperature if temperature is not None else self.default_temperature,
                        "stream": True,
                        **kwargs
                    }

                    if max_tokens is not None or self.default_max_tokens:
                        chat_params["max_completion_tokens"] = max_tokens or self.default_max_tokens

                    stream = self.client.chat.completions.create(**chat_params)

                    for chunk in stream:
                        # 安全检查：确保chunk有choices且不为空
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            choice = chunk.choices[0]
                            # 检查delta是否存在且有content
                            if hasattr(choice, 'delta') and hasattr(choice.delta,
                                                                    'content') and choice.delta.content is not None:
                                yield choice.delta.content

                except Exception as retry_e:
                    raise LLMAPIException(f"OpenAI API error (retry failed): {retry_e}")
            else:
                raise LLMAPIException(f"OpenAI API error: {e}")

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise LLMAPIException(f"Failed to get available models: {e}")

    def validate_config(self) -> bool:
        """验证配置"""
        try:
            # 尝试发送一个简单的测试消息来验证配置
            test_messages = [{"role": "user", "content": "test"}]
            chat_params = self._prepare_chat_params(
                test_messages,
                self.default_model,
                max_tokens=1
            )
            self.client.chat.completions.create(**chat_params)
            return True
        except Exception:
            return False
