import time
import requests
from typing import List, Callable, Optional
from config.settings import settings
import threading


class _QwenEmbeddingAPI:
    """Qwen3 Embedding API 客户端 - 内部单例实现"""

    def __init__(self, base_url: str = None, token: str = None, model: str = None, max_batch_size: int = 100):
        self.base_url = (base_url or settings.embedding_api_url).rstrip('/')
        self.token = token or settings.embedding_api_token
        self.model = model or settings.embedding_model
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.max_batch_size = max_batch_size
        self.request_timeout = 120
        self.embedding_dim = None
        self._initialized = False
        self._lock = threading.Lock()

        # 延迟初始化
        self._lazy_init()

    def _lazy_init(self):
        """延迟初始化，确保只初始化一次"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # 测试连接并获取维度
            self._test_connection()
            self._initialized = True
            print("🔄 QwenEmbeddingAPI 单例已初始化")

    def set_batch_size(self, batch_size: int):
        """动态设置批处理大小"""
        self._lazy_init()
        self.max_batch_size = batch_size
        print(f"📦 Embedding批处理大小已设置为: {self.max_batch_size}")

    def _test_connection(self):
        """测试API连接并获取实际的embedding维度"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=60)
            if response.status_code == 200:
                health_status = response.text.strip('"')
                print(f"✅ API连接成功: {health_status}")
                self.embedding_dim = self._get_actual_embedding_dimension()
                print(f"📐 实际Embedding维度: {self.embedding_dim}")
            else:
                raise Exception(f"API健康检查失败: {response.status_code}")
        except Exception as e:
            print(f"❌ API连接失败: {e}")
            raise

    def _get_actual_embedding_dimension(self) -> int:
        """通过实际调用API获取embedding维度"""
        try:
            print("🔍 正在检测embedding维度...")
            test_response = self._encode_single_batch(["测试文本"], get_dimension=True)
            if test_response and len(test_response) > 0:
                actual_dim = len(test_response[0])
                print(f"✅ 检测到embedding维度: {actual_dim}")
                return actual_dim
            else:
                print("⚠️ 无法检测embedding维度，使用默认值1024")
                return 1024
        except Exception as e:
            print(f"⚠️ 检测embedding维度时出错: {e}，使用默认值1024")
            return 1024

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为向量"""
        self._lazy_init()
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.max_batch_size - 1) // self.max_batch_size

        print(f"📦 开始批量向量化: {len(texts)} 个文本，分为 {total_batches} 个批次")

        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i + self.max_batch_size]
            batch_num = i // self.max_batch_size + 1

            print(f"   处理批次 {batch_num}/{total_batches} ({len(batch_texts)} 个文本)")
            batch_embeddings = self._encode_single_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            if i + self.max_batch_size < len(texts):
                time.sleep(0.2)

        print(f"✅ 批量向量化完成: 共处理 {len(all_embeddings)} 个向量")
        return all_embeddings

    def encode_texts_with_progress(self, texts: List[str],
                                   progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[
        List[float]]:
        """批量编码文本为向量（带进度回调的新方法）"""
        self._lazy_init()
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.max_batch_size - 1) // self.max_batch_size

        # 初始进度回调
        if progress_callback:
            progress_callback(0, total_batches, f"开始向量化 {len(texts)} 个文本，分为 {total_batches} 个批次")

        print(f"📦 开始批量向量化: {len(texts)} 个文本，分为 {total_batches} 个批次")

        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i + self.max_batch_size]
            batch_num = i // self.max_batch_size + 1

            # 批次开始回调
            if progress_callback:
                progress_callback(batch_num - 1, total_batches,
                                  f"正在处理第 {batch_num}/{total_batches} 批次 ({len(batch_texts)} 个文本)")

            print(f"   处理批次 {batch_num}/{total_batches} ({len(batch_texts)} 个文本)")

            start_time = time.time()
            batch_embeddings = self._encode_single_batch(batch_texts)
            end_time = time.time()

            all_embeddings.extend(batch_embeddings)

            # 批次完成回调
            if progress_callback:
                progress_callback(batch_num, total_batches,
                                  f"第 {batch_num}/{total_batches} 批次完成 ({end_time - start_time:.2f}秒)")

            if i + self.max_batch_size < len(texts):
                time.sleep(0.2)

        print(f"✅ 批量向量化完成: 共处理 {len(all_embeddings)} 个向量")

        # 最终完成回调
        if progress_callback:
            progress_callback(total_batches, total_batches, f"向量化完成，共处理 {len(all_embeddings)} 个向量")

        return all_embeddings

    def _encode_single_batch(self, texts: List[str], get_dimension: bool = False) -> List[List[float]]:
        """编码单个批次的文本"""
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                end_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    if not get_dimension:
                        print(f"      ✅ 批次完成 ({end_time - start_time:.2f}秒)")
                    return embeddings
                else:
                    error_msg = f"API请求失败: {response.status_code} - {response.text}"
                    if attempt == max_retries - 1:
                        raise Exception(error_msg)
                    else:
                        print(f"      ⚠️ {error_msg}, 重试中... ({attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception("API请求超时")
                else:
                    print(f"      ⚠️ API请求超时，重试中... ({attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API请求异常: {e}")
                else:
                    print(f"      ⚠️ API请求异常: {e}, 重试中... ({attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)

        print(f"      ❌ 批次编码失败，返回零向量")
        fallback_dim = getattr(self, 'embedding_dim', 1024)
        return [[0.0] * fallback_dim for _ in texts]


# 模块级单例实例
_embedding_instance = None
_embedding_lock = threading.Lock()


class QwenEmbeddingAPI:
    """QwenEmbeddingAPI 的代理类，确保始终返回同一个实例"""

    def __new__(cls, *args, **kwargs):
        global _embedding_instance
        if _embedding_instance is None:
            with _embedding_lock:
                if _embedding_instance is None:
                    _embedding_instance = _QwenEmbeddingAPI(*args, **kwargs)
        return _embedding_instance
