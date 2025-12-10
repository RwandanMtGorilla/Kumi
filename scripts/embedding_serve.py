#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用的 Sentence Transformers 模型启动脚本
支持通过命令行参数配置模型路径、模型名和API密钥
"""

import os
import sys
import argparse
import numpy as np
import logging
import uvicorn
import datetime
import torch
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import Optional, List, Union
import base64
from pathlib import Path

# ================ 配置解析 ================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='通用的 Sentence Transformers Embedding API 服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用相对路径
  python embedding_serve.py --model_path ./models/my-model --model_name my-model-v1

  # 使用绝对路径和自定义API密钥
  python embedding_serve.py --model_path F:/models/qwen --model_name qwen3 --apikey mykey123

  # 指定端口
  python embedding_serve.py --model_path ./model --model_name test --port 8000
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径(相对路径或绝对路径),必填'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称,必填'
    )

    parser.add_argument(
        '--apikey',
        type=str,
        default='123',
        help='API访问密钥,选填(默认: 123)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8504,
        help='服务端口,选填(默认: 8504)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务监听地址,选填(默认: 0.0.0.0)'
    )

    return parser.parse_args()


# ================ 全局配置 ================

# 解析命令行参数
args = parse_arguments()

# 处理模型路径 - 支持相对路径和绝对路径
model_path_input = args.model_path
if os.path.isabs(model_path_input):
    # 绝对路径
    EMBEDDING_MODEL_PATH = model_path_input
else:
    # 相对路径 - 相对于脚本所在目录或当前工作目录
    script_dir = Path(__file__).parent
    model_path_relative_to_script = script_dir / model_path_input
    model_path_relative_to_cwd = Path(model_path_input)

    # 优先检查相对于脚本目录的路径
    if model_path_relative_to_script.exists():
        EMBEDDING_MODEL_PATH = str(model_path_relative_to_script.resolve())
    elif model_path_relative_to_cwd.exists():
        EMBEDDING_MODEL_PATH = str(model_path_relative_to_cwd.resolve())
    else:
        # 都不存在时,尝试使用相对于当前工作目录的路径(可能是下载路径)
        EMBEDDING_MODEL_PATH = str(model_path_relative_to_cwd.resolve())

MODEL_NAME = args.model_name
API_KEY = args.apikey
SERVER_PORT = args.port
SERVER_HOST = args.host

print(f"=== 配置信息 ===")
print(f"模型路径: {EMBEDDING_MODEL_PATH}")
print(f"模型名称: {MODEL_NAME}")
print(f"API密钥: {'*' * len(API_KEY)}")
print(f"服务地址: {SERVER_HOST}:{SERVER_PORT}")
print(f"===============\n")

# ================ FastAPI 应用 ================

app = FastAPI(title=f"{MODEL_NAME} Embedding API")
security = HTTPBearer()

# ================ 数据模型 ================

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]  # 支持单个文本或文本列表
    model: Optional[str] = MODEL_NAME  # 模型名称(可选)
    encoding_format: Optional[str] = "float"  # 编码格式(float或base64)


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


# ================ 单例模式 ================

class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


# ================ Embedding 模型封装 ================

class SentenceTransformerEmbedding(metaclass=Singleton):
    def __init__(self, model_path: str, model_name: str):
        print(f"正在初始化 Sentence Transformers 模型: {model_path}")
        print(f"初始化时间: {datetime.datetime.now()}")

        self.model_name = model_name

        # 检查GPU可用性
        self.device_count = torch.cuda.device_count()
        print(f"检测到 {self.device_count} 张GPU卡")

        # 设置设备
        if self.device_count > 0:
            self.device = 'cuda'
            print(f"使用GPU设备")
        else:
            self.device = 'cpu'
            print("使用CPU设备")

        # 初始化SentenceTransformer模型
        try:
            # 配置模型参数
            model_kwargs = {}
            tokenizer_kwargs = {}

            if self.device == 'cuda':
                # 如果有GPU,尝试使用flash_attention_2和device_map
                try:
                    model_kwargs.update({
                        "attn_implementation": "flash_attention_2",
                        "device_map": "auto",
                        "torch_dtype": torch.float16,  # 使用半精度节省显存
                    })
                    tokenizer_kwargs["padding_side"] = "left"
                    print("启用flash_attention_2和自动设备映射")
                except Exception as e:
                    print(f"无法启用flash_attention_2: {e}")
                    model_kwargs = {"torch_dtype": torch.float16}

            # 初始化模型
            self.model = SentenceTransformer(
                model_path,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                device=self.device
            )
            print("SentenceTransformer模型初始化成功")

        except Exception as e:
            print(f"模型初始化失败: {e}")
            # 尝试更保守的配置
            try:
                print("尝试使用保守配置重新初始化...")
                self.model = SentenceTransformer(
                    model_path,
                    device=self.device
                )
                print("使用保守配置初始化成功")

            except Exception as e2:
                print(f"保守配置也失败: {e2}")
                raise RuntimeError(f"无法初始化SentenceTransformer模型: {e2}")

        # 获取模型信息
        try:
            # 测试一个简单的embedding来获取维度信息
            test_embedding = self.model.encode(["test"], convert_to_tensor=True)
            self.embedding_dim = test_embedding.shape[1]
            print(f"模型embedding维度: {self.embedding_dim}")
        except Exception as e:
            print(f"无法获取embedding维度: {e}")
            self.embedding_dim = 768  # 默认维度(BERT-base大小)

        print(f"模型初始化完成: {datetime.datetime.now()}")

        # 打印GPU内存使用情况
        if self.device == 'cuda':
            self._print_gpu_memory_usage()

    def _print_gpu_memory_usage(self):
        """打印所有GPU的内存使用情况"""
        print("\n=== GPU内存使用情况 ===")
        for i in range(self.device_count):
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
                cached_memory = torch.cuda.memory_reserved(i) / 1024**3

                print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
                print(f"  总显存: {total_memory:.2f} GB")
                print(f"  已分配: {allocated_memory:.2f} GB")
                print(f"  已缓存: {cached_memory:.2f} GB")
                print(f"  使用率: {(allocated_memory/total_memory)*100:.1f}%")
        print("========================\n")

    def encode_texts(self, texts: List[str], use_query_prompt: bool = False) -> np.ndarray:
        """
        将文本列表编码为向量

        Args:
            texts: 要编码的文本列表
            use_query_prompt: 是否使用查询提示符(用于查询类文本)

        Returns:
            numpy数组,形状为 (len(texts), embedding_dim)
        """
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"正在处理 {len(texts)} 个文本...")

            # 使用SentenceTransformer进行embedding
            if use_query_prompt:
                # 使用查询提示
                if hasattr(self.model, 'prompts') and 'query' in self.model.prompts:
                    embeddings = self.model.encode(
                        texts,
                        prompt_name="query",
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                else:
                    # 如果没有内置的query prompt,使用默认编码
                    embeddings = self.model.encode(
                        texts,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
            else:
                # 文档编码,不使用特殊prompt
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

            # 转换为numpy数组
            if isinstance(embeddings, torch.Tensor):
                embeddings_array = embeddings.cpu().numpy().astype(np.float32)
            else:
                embeddings_array = np.array(embeddings, dtype=np.float32)

            print(f"成功生成embeddings,形状: {embeddings_array.shape}")
            return embeddings_array

        except Exception as e:
            print(f"文本编码时出错: {e}")

            # 如果出现内存错误,尝试分批处理
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print("检测到GPU内存不足,尝试分批处理...")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return self._encode_texts_in_batches(texts, use_query_prompt, batch_size=16)
                except Exception as e2:
                    print(f"分批处理也失败: {e2}")

            # 返回零向量作为fallback
            print(f"返回零向量作为fallback, 维度: {self.embedding_dim}")
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

    def _encode_texts_in_batches(self, texts: List[str], use_query_prompt: bool = False, batch_size: int = 16) -> np.ndarray:
        """分批处理文本编码,用于处理内存不足的情况"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}, 文本数量: {len(batch_texts)}")

            # 处理当前批次
            if use_query_prompt:
                if hasattr(self.model, 'prompts') and 'query' in self.model.prompts:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        prompt_name="query",
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
                else:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
            else:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

            # 转换为numpy并添加到列表
            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu().numpy()

            all_embeddings.append(batch_embeddings)

            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 合并所有批次的结果
        return np.vstack(all_embeddings).astype(np.float32)

    def get_embedding_dimension(self) -> int:
        """获取embedding维度"""
        return self.embedding_dim


# ================ Embedding 服务 ================

class EmbeddingService:
    def __init__(self, model_path: str, model_name: str):
        self.embedder = SentenceTransformerEmbedding(model_path, model_name)
        self.model_name = model_name

    def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """创建文本embeddings"""
        # 处理输入文本
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        if not texts:
            raise ValueError("输入文本不能为空")

        # 检测是否为查询类文本(启发式方法)
        use_query_prompt = any(
            text.strip().endswith('?') or
            any(keyword in text.lower() for keyword in ['what', 'how', 'where', 'when', 'why', 'who', 'find', 'search', 'query'])
            for text in texts
        )

        # 获取embeddings
        embeddings = self.embedder.encode_texts(texts, use_query_prompt=use_query_prompt)

        # 构建响应数据
        data = []
        for i, embedding in enumerate(embeddings):
            if request.encoding_format == "base64":
                # 转换为base64编码
                embedding_bytes = embedding.astype(np.float32).tobytes()
                embedding_data = base64.b64encode(embedding_bytes).decode('utf-8')
            else:
                # 默认返回float列表
                embedding_data = embedding.astype(float).tolist()

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding_data
            })

        # 构建使用统计
        usage = {
            "prompt_tokens": sum(len(text.split()) for text in texts),  # 简单估算token数
            "total_tokens": sum(len(text.split()) for text in texts)
        }

        return EmbeddingResponse(
            data=data,
            model=self.model_name,
            usage=usage
        )


# 全局服务实例
embedding_service = None


def get_embedding_service():
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService(EMBEDDING_MODEL_PATH, MODEL_NAME)
    return embedding_service


# ================ API 端点 ================

@app.post('/v1/embeddings')
async def create_embeddings(
    request: EmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """创建文本embeddings的API端点"""
    # 验证token
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        service = get_embedding_service()
        response = service.create_embeddings(request)
        return response.dict()
    except Exception as e:
        print(f"创建embeddings时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理embeddings请求时出错: {str(e)}")


@app.post('/v1/embed')  # 简化版端点
async def embed_text(
    request: EmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """简化版的文本embedding端点"""
    return await create_embeddings(request, credentials)


@app.get('/v1/models')
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(datetime.datetime.now().timestamp()),
                "owned_by": "",
                "root": MODEL_NAME,
                "parent": None,
                "permission": []
            }
        ]
    }


@app.get('/health')
async def health_check():
    """健康检查端点"""
    try:
        service = get_embedding_service()
        embedding_dim = service.embedder.get_embedding_dimension()

        # 获取GPU状态信息
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "gpu_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB",
                    "memory_total": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                })

        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "embedding_dimension": embedding_dim,
            "gpu_count": torch.cuda.device_count(),
            "gpu_info": gpu_info,
            "engine": "SentenceTransformers",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }


@app.get('/gpu_status')
async def gpu_status():
    """获取GPU状态信息"""
    if not torch.cuda.is_available():
        return {"error": "CUDA不可用"}

    gpu_status_list = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        cached_memory = torch.cuda.memory_reserved(i) / 1024**3

        gpu_status_list.append({
            "gpu_id": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": round(total_memory, 2),
            "allocated_memory_gb": round(allocated_memory, 2),
            "cached_memory_gb": round(cached_memory, 2),
            "utilization_percent": round((allocated_memory/total_memory)*100, 1)
        })

    return {
        "gpu_count": torch.cuda.device_count(),
        "gpu_status": gpu_status_list,
        "engine": "SentenceTransformers",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get('/')
async def root():
    """根路径信息"""
    return {
        "message": f"{MODEL_NAME} Embedding API - SentenceTransformers",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "engine": "SentenceTransformers",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "embed": "/v1/embed",
            "models": "/v1/models",
            "health": "/health",
            "gpu_status": "/gpu_status"
        }
    }


# ================ 主程序入口 ================

if __name__ == "__main__":
    # 检查环境变量中的token(可选)
    env_token = os.getenv("ACCESS_TOKEN")
    if env_token:
        API_KEY = env_token
        print(f"从环境变量读取 ACCESS_TOKEN")

    # 启动前检查设备可用性
    device_info = "CPU"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info = f"GPU (数量: {device_count})"
        print(f"\nCUDA可用,GPU数量: {device_count}")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\nCUDA不可用,将使用CPU")

    # 检查必要的包
    try:
        import sentence_transformers
        print(f"SentenceTransformers版本: {sentence_transformers.__version__}")
    except ImportError:
        print("错误: 未安装sentence-transformers包")
        print("请安装: pip install sentence-transformers")
        sys.exit(1)

    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        print("错误: 未安装transformers包")
        print("请安装: pip install transformers")
        sys.exit(1)

    # 检查模型路径是否存在
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"\n警告: 模型路径不存在: {EMBEDDING_MODEL_PATH}")
        print("模型可能会从 Hugging Face 下载")

    try:
        print(f"\n正在启动 {MODEL_NAME} Embedding API 服务...")
        print(f"使用设备: {device_info}")
        print(f"监听地址: {SERVER_HOST}:{SERVER_PORT}\n")
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
    except Exception as e:
        print(f"API启动失败！\n报错：\n{e}")
        sys.exit(1)
