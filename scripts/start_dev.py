import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 改变工作目录到项目根目录
os.chdir(project_root)

# 现在导入和运行
if __name__ == "__main__":
    import uvicorn
    from config.settings import settings

    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
