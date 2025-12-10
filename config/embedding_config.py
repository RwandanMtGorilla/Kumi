"""
Embedding配置数据模型
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import yaml
import re


class EmbeddingProvider:
    """Embedding提供商配置"""
    def __init__(self, name: str, api_base_url: str, api_key: str, models: List[str]):
        self.name = name
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.models = models

    def __repr__(self):
        return f"EmbeddingProvider(name={self.name}, models={len(self.models)})"


class EmbeddingConfig:
    """Embedding配置管理器"""

    def __init__(self, config_path: str = None):
        """
        初始化配置管理器

        Args:
            config_path: YAML配置文件路径,默认为 config/embedding_providers.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "embedding_providers.yaml"

        self.config_path = Path(config_path)
        self.providers: Dict[str, EmbeddingProvider] = {}
        self.default_provider: str = ""
        self.default_model: str = ""

        self._load_config()

    def _load_config(self):
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Embedding配置文件不存在: {self.config_path}\n"
                f"请从 {self.config_path}.example 复制并修改"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 解析提供商配置
        for provider_config in config.get('Providers', []):
            provider = EmbeddingProvider(
                name=provider_config['name'],
                api_base_url=provider_config['api_base_url'],
                api_key=provider_config['api_key'],
                models=provider_config['models']
            )
            self.providers[provider.name] = provider

        # 解析默认路由
        router_config = config.get('Router', {})
        default_str = router_config.get('default', '')

        if default_str:
            parts = default_str.split(',')
            if len(parts) == 2:
                self.default_provider = parts[0].strip()
                self.default_model = parts[1].strip()
            else:
                raise ValueError(f"Router.default 格式错误,应为 'provider,model': {default_str}")

        if not self.providers:
            raise ValueError("配置文件中没有定义任何提供商")

    def get_provider(self, provider_name: str) -> Optional[EmbeddingProvider]:
        """获取指定提供商配置"""
        return self.providers.get(provider_name)

    def get_all_providers(self) -> List[str]:
        """获取所有提供商名称"""
        return list(self.providers.keys())

    def get_all_models(self) -> List[Dict[str, str]]:
        """
        获取所有可用模型列表

        Returns:
            [
                {"provider": "openrouter", "model": "google/gemini-embedding-001", "display_name": "openrouter,google/gemini-embedding-001"},
                ...
            ]
        """
        models = []
        for provider_name, provider in self.providers.items():
            for model in provider.models:
                models.append({
                    "provider": provider_name,
                    "model": model,
                    "display_name": f"{provider_name},{model}"
                })
        return models

    def get_default_model(self) -> Tuple[str, str]:
        """
        获取默认模型

        Returns:
            (provider_name, model_name)
        """
        return (self.default_provider, self.default_model)

    def get_model_info(self, provider_name: str, model_name: str) -> Optional[Dict[str, str]]:
        """
        获取指定模型的完整信息

        Returns:
            {
                "provider": "openrouter",
                "model": "google/gemini-embedding-001",
                "api_base_url": "https://openrouter.ai/api",
                "api_key": "sk-xxx"
            }
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return None

        if model_name not in provider.models:
            return None

        return {
            "provider": provider_name,
            "model": model_name,
            "api_base_url": provider.api_base_url,
            "api_key": provider.api_key
        }

    def parse_model_identifier(self, model_identifier: str) -> Tuple[str, str]:
        """
        解析模型标识符

        Args:
            model_identifier: "provider,model" 或 "provider/model" 格式

        Returns:
            (provider_name, model_name)
        """
        if ',' in model_identifier:
            parts = model_identifier.split(',', 1)
        elif '/' in model_identifier:
            # 兼容旧格式
            parts = model_identifier.split('/', 1)
        else:
            raise ValueError(f"无效的模型标识符格式: {model_identifier}")

        if len(parts) != 2:
            raise ValueError(f"无效的模型标识符格式: {model_identifier}")

        return parts[0].strip(), parts[1].strip()

    @staticmethod
    def generate_model_abbreviation(model_name: str) -> str:
        """
        生成模型缩写用于 collection 名称

        规则:
        - 字母: 如果模型名中存在`/`则从`/`开始往右取三个字母，否则直接从最左往右取三个字母
        - 数字: 从模型名中最靠右的一个数字开始往左取连续的数字(包含小数点`.`)，最多三位
        - 格式: f"{字母}-{数字}"

        Examples:
            "google/gemini-embedding-001" -> "gem-001"
            "Qwen3-Embedding-0.6B" -> "Qwe-0.6"
            "baai/bge-m3" -> "bge-3"
            "thenlper/gte-base" -> "gte"
            "text-embedding-3-large" -> "tex-3"
        """
        # 提取字母部分
        if '/' in model_name:
            # 从`/`后开始取字母
            after_slash = model_name.split('/')[-1]
            letter_match = re.search(r'[a-zA-Z]{1,3}', after_slash)
            left_part = letter_match.group(0) if letter_match else ""
        else:
            # 从最左边开始取字母
            letter_match = re.search(r'[a-zA-Z]{1,3}', model_name)
            left_part = letter_match.group(0) if letter_match else ""

        # 提取数字部分：从最右边的数字开始往左取连续的数字(包含小数点)，最多三位
        # 先找到最右边的数字位置
        rightmost_digit_pos = -1
        for i in range(len(model_name) - 1, -1, -1):
            if model_name[i].isdigit():
                rightmost_digit_pos = i
                break

        right_part = ""
        if rightmost_digit_pos >= 0:
            # 从这个位置往左取连续的数字和小数点
            start_pos = rightmost_digit_pos
            for i in range(rightmost_digit_pos - 1, -1, -1):
                if model_name[i].isdigit() or model_name[i] == '.':
                    start_pos = i
                else:
                    break

            # 提取数字部分
            number_str = model_name[start_pos:rightmost_digit_pos + 1]

            # 限制最多三位（包括小数点）
            if len(number_str) > 3:
                # 优先保留最右边的部分
                number_str = number_str[-3:]

            right_part = number_str

        # 组合结果
        if left_part and right_part:
            return f"{left_part}-{right_part}"
        elif left_part:
            return left_part
        elif right_part:
            return right_part
        else:
            return "emb"  # 默认值
