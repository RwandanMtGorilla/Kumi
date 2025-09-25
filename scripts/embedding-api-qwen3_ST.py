#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

app = FastAPI()
security = HTTPBearer()
env_bearer_token = '123'

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]  # 支持单个文本或文本列表
    model: Optional[str] = "Qwen3-Embedding-0.6B"  # 模型名称（可选）
    encoding_format: Optional[str] = "float"  # 编码格式（float或base64）
    dimensions: Optional[int] = None  # 输出维度（可选，用于降维）

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

EMBEDDING_MODEL_PATH = "F:\\embedding\\Qwen3-Embedding-0.6B"

class QwenEmbedding(metaclass=Singleton):
    def __init__(self, model_path):
        print(f"正在初始化Qwen3 Embedding模型 (SentenceTransformers版本): {model_path}")
        print(f"初始化时间: {datetime.datetime.now()}")
        
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
                # 如果有GPU，尝试使用flash_attention_2和device_map
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
            self.embedding_dim = 4096  # 默认维度
        
        print(f"模型初始化完成: {datetime.datetime.now()}")
        
        # 打印GPU内存使用情况
        if self.device == 'cuda':
            self._print_gpu_memory_usage()

    def _print_gpu_memory_usage(self):
        """
        打印所有GPU的内存使用情况
        """
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
            use_query_prompt: 是否使用查询提示符（用于查询类文本）
        
        Returns:
            numpy数组，形状为 (len(texts), embedding_dim)
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
                    # 如果没有内置的query prompt，使用默认编码
                    embeddings = self.model.encode(
                        texts,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )
            else:
                # 文档编码，不使用特殊prompt
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
            
            print(f"成功生成embeddings，形状: {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            print(f"文本编码时出错: {e}")
            
            # 如果出现内存错误，尝试分批处理
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print("检测到GPU内存不足，尝试分批处理...")
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
        """
        分批处理文本编码，用于处理内存不足的情况
        """
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

class EmbeddingService:
    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH):
        self.embedder = QwenEmbedding(model_path)
        self.model_name = "Qwen3-Embedding-4B"

    def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        创建文本embeddings
        """
        # 处理输入文本
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        if not texts:
            raise ValueError("输入文本不能为空")
        
        # 检测是否为查询类文本（启发式方法）
        use_query_prompt = any(
            text.strip().endswith('?') or 
            any(keyword in text.lower() for keyword in ['what', 'how', 'where', 'when', 'why', 'who', 'find', 'search', 'query'])
            for text in texts
        )
        
        # 获取embeddings
        embeddings = self.embedder.encode_texts(texts, use_query_prompt=use_query_prompt)
        
        # 处理维度调整（如果指定了dimensions参数）
        if request.dimensions and request.dimensions < embeddings.shape[1]:
            # 简单截断到指定维度
            embeddings = embeddings[:, :request.dimensions]
        
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
        embedding_service = EmbeddingService()
    return embedding_service

@app.post('/v1/embeddings')
async def create_embeddings(
    request: EmbeddingRequest, 
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    创建文本embeddings的API端点
    """
    # 验证token
    token = credentials.credentials
    if env_bearer_token is not None and token != env_bearer_token:
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
    """
    简化版的文本embedding端点
    """
    return await create_embeddings(request, credentials)

@app.get('/v1/models')
async def list_models():
    """
    列出可用模型
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen3-Embedding-4B",
                "object": "model",
                "created": 1677610602,
                "owned_by": "qwen",
                "root": "Qwen3-Embedding-4B",
                "parent": None,
                "permission": []
            }
        ]
    }

@app.get('/health')
async def health_check():
    """
    健康检查端点
    """
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
            "model": "Qwen3-Embedding-4B",
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
    """
    获取GPU状态信息
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA不可用"}
    
    gpu_status = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        cached_memory = torch.cuda.memory_reserved(i) / 1024**3
        
        gpu_status.append({
            "gpu_id": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": round(total_memory, 2),
            "allocated_memory_gb": round(allocated_memory, 2),
            "cached_memory_gb": round(cached_memory, 2),
            "utilization_percent": round((allocated_memory/total_memory)*100, 1)
        })
    
    return {
        "gpu_count": torch.cuda.device_count(),
        "gpu_status": gpu_status,
        "engine": "SentenceTransformers",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get('/')
async def root():
    """
    根路径信息
    """
    return {
        "message": "Qwen3 Embedding API - SentenceTransformers Version",
        "version": "1.0.0",
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

if __name__ == "__main__":
    # 从环境变量获取token
    token = os.getenv("ACCESS_TOKEN")
    if token is not None:
        env_bearer_token = token
    
    # 启动前检查设备可用性
    device_info = "CPU"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info = f"GPU (数量: {device_count})"
        print(f"CUDA可用，GPU数量: {device_count}")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("CUDA不可用，将使用CPU")
    
    # 检查必要的包
    try:
        import sentence_transformers
        print(f"SentenceTransformers版本: {sentence_transformers.__version__}")
    except ImportError:
        print("错误: 未安装sentence-transformers包")
        print("请安装: pip install sentence-transformers")
        exit(1)
    
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        print("错误: 未安装transformers包")
        print("请安装: pip install transformers")
        exit(1)
    
    try:
        print(f"正在启动Qwen3 Embedding API服务（SentenceTransformers版本）...")
        print(f"使用设备: {device_info}")
        uvicorn.run(app, host='0.0.0.0', port=8504)
    except Exception as e:
        print(f"API启动失败！\n报错：\n{e}")
