import logging
import sys
import os
from pathlib import Path
from typing import Optional
from .settings import settings


def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """配置日志"""
    level = log_level or settings.LOG_LEVEL
    file_path = log_file or settings.LOG_FILE

    # 配置日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定了日志文件）
    if file_path:
        try:
            # 确保日志文件的目录存在
            log_dir = os.path.dirname(file_path)
            if log_dir:  # 如果有目录部分
                Path(log_dir).mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # 如果文件日志设置失败，只使用控制台日志
            print(f"警告: 无法设置文件日志 ({file_path}): {e}")
            print("将仅使用控制台日志")

    # 设置第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
