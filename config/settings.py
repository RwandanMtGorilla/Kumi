import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from typing import List, Optional
from pathlib import Path

# 加载环境变量
load_dotenv()

# 导入embedding配置
try:
    from config.embedding_config import EmbeddingConfig
    _embedding_config = None  # 延迟加载
except ImportError:
    _embedding_config = None


class Settings:
    """应用配置类"""

    # 应用基础配置
    APP_NAME: str = os.getenv("APP_NAME", "KUMI Service")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    ADMIN_USER_NAME: str = os.getenv("ADMIN_USER_NAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "KUMI-admin-123456")

    test_host: str = os.getenv("TEST_HOST", "127.0.0.1")
    test_port: int = int(os.getenv("TEST_PORT", "8001"))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # 资源目录配置
    RESOURCE_BASE_DIR: str = os.getenv("RESOURCE_BASE_DIR", "resources")
    DDL_EXPORT_DIR: str = os.getenv("DDL_EXPORT_DIR", "ddl_exports")

    # 服务控制 - 控制哪些API服务启用
    ENABLE_KNOWLEDGE_API: bool = os.getenv("ENABLE_KNOWLEDGE_API", "true").lower() == "true"
    ENABLE_DOCUMENT_CONVERSION_API: bool = os.getenv("ENABLE_DOCUMENT_CONVERSION_API", "true").lower() == "true"

    # MarkItDown配置
    MARKITDOWN_ENABLE_PLUGINS: bool = os.getenv("MARKITDOWN_ENABLE_PLUGINS", "false").lower() == "true"
    MARKITDOWN_MAX_FILE_SIZE: int = int(os.getenv("MARKITDOWN_MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB

    # 支持的文件扩展名（从环境变量读取，用逗号分隔）
    _allowed_extensions = os.getenv("MARKITDOWN_ALLOWED_EXTENSIONS",
                                    ".pdf,.docx,.pptx,.xlsx,.xls,.txt,.md,.html,.csv,.wav,.mp3")
    MARKITDOWN_ALLOWED_EXTENSIONS: List[str] = [ext.strip() for ext in _allowed_extensions.split(",")]

    # Chroma向量数据库配置
    vector_db_type: str = os.getenv('VECTOR_DB_TYPE', 'chroma')

    # Milvus Configuration
    milvus_host: str = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port: int = int(os.getenv('MILVUS_PORT', '19530'))
    milvus_user: str = os.getenv('MILVUS_USER', 'root')
    milvus_password: str = os.getenv('MILVUS_PASSWORD', 'Milvus')

    # ChromaDB Configuration
    chroma_host: str = os.getenv('CHROMA_HOST', 'localhost')
    chroma_port: int = int(os.getenv('CHROMA_PORT', '8000'))

    # Embedding配置文件路径
    _EMBEDDING_CONFIG_PATH: str = os.getenv('EMBEDDING_CONFIG_PATH', 'config/embedding_providers.yaml')

    VERIFY_TOKEN: str = os.getenv('VERIFY_TOKEN', 'NOTPOSSIBLEADMINp0ss')

    # PATH
    PROJECT_ROOT: str = str(Path(__file__).parent.parent.absolute())

    # 如果需要自定义项目根目录，可以通过环境变量覆盖
    _custom_root = os.getenv('PROJECT_ROOT')
    if _custom_root:
        PROJECT_ROOT = str(Path(_custom_root).absolute())

    # 相对于项目根目录的路径配置
    _CSV_TEST_RELATIVE_PATH: str = os.getenv('CSV_TEST_RELATIVE_PATH', 'dataset/dataset_csv_test')
    _YAML_EVALUATE_RELATIVE_PATH: str = os.getenv('YAML_EVALUATE_RELATIVE_PATH', 'dataset/yaml_eval')
    _YAML_EVALUATE_TEMPLATES_RELATIVE_PATH: str = os.getenv('YAML_EVALUATE_TEMPLATES_RELATIVE_PATH',
                                                            'dataset/yaml_eval_templates')

    _WORKFLOWS_CONFIG_RELATIVE_PATH: str = os.getenv('WORKFLOWS_CONFIG_RELATIVE_PATH', 'dataset/workflows.json')
    _TABLE_FOLDER_RELATIVE_PATH: str = os.getenv('TABLE_FOLDER_RELATIVE_PATH', 'dataset/dataset_csv_test')
    _TEMP_RESULTS_RELATIVE_PATH: str = os.getenv('TEMP_RESULTS_RELATIVE_PATH', 'output/temp_results')
    _EVALUATION_RESULTS_RELATIVE_PATH: str = os.getenv('EVALUATION_RESULTS_RELATIVE_PATH', 'output/evaluation_results')
    _YAML_RULES_RELATIVE_PATH: str = os.getenv('YAML_RULES_RELATIVE_PATH', 'dataset/yaml_eval')

    # LLM配置
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    CLAUDE_API_KEY: Optional[str] = os.getenv("CLAUDE_API_KEY")
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.4"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))

    API_API_KEY: Optional[str] = os.getenv("API_API_KEY")
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    # 生成绝对路径
    @property
    def CSV_TEST_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._CSV_TEST_RELATIVE_PATH)

    @property
    def YAML_EVALUATE_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._YAML_EVALUATE_RELATIVE_PATH)

    @property
    def YAML_EVALUATE_TEMPLATES_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._YAML_EVALUATE_TEMPLATES_RELATIVE_PATH)

    @property
    def WORKFLOWS_CONFIG_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._WORKFLOWS_CONFIG_RELATIVE_PATH)

    @property
    def TABLE_FOLDER_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._TABLE_FOLDER_RELATIVE_PATH)

    @property
    def TEMP_RESULTS_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._TEMP_RESULTS_RELATIVE_PATH)

    @property
    def EVALUATION_RESULTS_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._EVALUATION_RESULTS_RELATIVE_PATH)

    @property
    def YAML_RULES_PATH(self) -> str:
        return str(Path(self.PROJECT_ROOT) / self._YAML_RULES_RELATIVE_PATH)

    def get_absolute_path(self, relative_path: str) -> str:
        """
        根据相对路径获取绝对路径

        Args:
            relative_path: 相对于项目根目录的路径

        Returns:
            str: 绝对路径
        """
        return str(Path(self.PROJECT_ROOT) / relative_path)

    def create_directories(self):
        """创建所有配置的目录"""
        directories = [
            self.CSV_TEST_PATH,
            self.YAML_EVALUATE_PATH,
            self.YAML_EVALUATE_TEMPLATES_PATH,
            self.TABLE_FOLDER_PATH,
            self.TEMP_RESULTS_PATH,
            self.EVALUATION_RESULTS_PATH,
            self.YAML_RULES_PATH,
            # 配置文件的目录
            str(Path(self.WORKFLOWS_CONFIG_PATH).parent)
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def print_config(self):
        """打印当前配置信息"""
        print("⚙️  当前路径配置:")
        print(f"   项目根目录: {self.PROJECT_ROOT}")
        print(f"   工作流配置: {self.WORKFLOWS_CONFIG_PATH}")
        print(f"   表格文件夹: {self.TABLE_FOLDER_PATH}")
        print(f"   临时结果: {self.TEMP_RESULTS_PATH}")
        print(f"   评测结果: {self.EVALUATION_RESULTS_PATH}")
        print(f"   YAML规则: {self.YAML_RULES_PATH}")


    @property
    def chroma_url(self) -> str:
        """构建Chroma连接URL"""
        return f"http://{self.CHROMA_HOST}:{self.CHROMA_PORT}"

    @property
    def resource_dir(self) -> str:
        """获取资源目录完整路径"""
        return os.path.abspath(self.RESOURCE_BASE_DIR)

    @property
    def ddl_export_path(self) -> str:
        """获取DDL导出目录完整路径"""
        return os.path.join(self.resource_dir, self.DDL_EXPORT_DIR)

    @property
    def EMBEDDING_CONFIG_PATH(self) -> str:
        """获取embedding配置文件完整路径"""
        return str(Path(self.PROJECT_ROOT) / self._EMBEDDING_CONFIG_PATH)

    def get_embedding_config(self) -> 'EmbeddingConfig':
        """
        获取embedding配置实例(单例模式)

        Returns:
            EmbeddingConfig实例
        """
        global _embedding_config
        if _embedding_config is None:
            from config.embedding_config import EmbeddingConfig
            _embedding_config = EmbeddingConfig(self.EMBEDDING_CONFIG_PATH)
        return _embedding_config


# 创建全局配置实例
settings = Settings()