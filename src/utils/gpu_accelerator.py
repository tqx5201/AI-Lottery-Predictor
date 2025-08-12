"""
GPU加速计算模块
支持CUDA、OpenCL和CPU fallback的高性能计算
"""

import numpy as np
import logging
from typing import Optional, Union, List, Dict, Any
import time
from contextlib import contextmanager

# 尝试导入GPU加速库
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

try:
    from numba import cuda, jit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    cuda = None
    jit = None
    prange = None

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """GPU加速计算器"""
    
    def __init__(self, prefer_gpu: bool = True, device_id: int = 0):
        """
        初始化GPU加速器
        
        Args:
            prefer_gpu: 是否优先使用GPU
            device_id: GPU设备ID
        """
        self.prefer_gpu = prefer_gpu
        self.device_id = device_id
        self.compute_backend = None
        self.context = None
        self.queue = None
        
        # 检测并初始化最佳计算后端
        self._initialize_backend()
        
        logger.info(f"GPU加速器初始化完成，使用后端: {self.compute_backend}")
    
    def _initialize_backend(self):
        """初始化计算后端"""
        if not self.prefer_gpu:
            self.compute_backend = "numpy"
            return
        
        # 优先级: CUDA > OpenCL > Numba > NumPy
        if CUPY_AVAILABLE:
            try:
                # 检查CUDA设备
                if cp.cuda.runtime.getDeviceCount() > self.device_id:
                    cp.cuda.Device(self.device_id).use()
                    # 测试CUDA功能
                    test_array = cp.array([1, 2, 3])
                    _ = cp.sum(test_array)
                    self.compute_backend = "cupy"
                    logger.info(f"CUDA GPU加速已启用，设备: {cp.cuda.Device().id}")
                    return
            except Exception as e:
                logger.warning(f"CUDA初始化失败: {e}")
        
        if OPENCL_AVAILABLE:
            try:
                # 初始化OpenCL
                platforms = cl.get_platforms()
                if platforms:
                    devices = platforms[0].get_devices()
                    if devices and len(devices) > self.device_id:
                        self.context = cl.Context([devices[self.device_id]])
                        self.queue = cl.CommandQueue(self.context)
                        self.compute_backend = "opencl"
                        logger.info(f"OpenCL GPU加速已启用，设备: {devices[self.device_id].name}")
                        return
            except Exception as e:
                logger.warning(f"OpenCL初始化失败: {e}")
        
        if NUMBA_AVAILABLE:
            try:
                # 检查CUDA支持
                if cuda.is_available():
                    self.compute_backend = "numba_cuda"
                    logger.info("Numba CUDA加速已启用")
                    return
                else:
                    self.compute_backend = "numba_cpu"
                    logger.info("Numba CPU加速已启用")
                    return
            except Exception as e:
                logger.warning(f"Numba初始化失败: {e}")
        
        # 回退到NumPy
        self.compute_backend = "numpy"
        logger.info("使用NumPy CPU计算")
    
    @contextmanager
    def performance_timer(self, operation_name: str):
        """性能计时器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.debug(f"{operation_name} 耗时: {elapsed:.4f}秒 (后端: {self.compute_backend})")
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """将数组转移到GPU"""
        if self.compute_backend == "cupy":
            return cp.asarray(array)
        elif self.compute_backend == "opencl":
            return cl_array.to_device(self.queue, array)
        else:
            return array
    
    def to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """将数组转移到CPU"""
        if self.compute_backend == "cupy" and hasattr(array, 'get'):
            return array.get()
        elif self.compute_backend == "opencl" and hasattr(array, 'get'):
            return array.get()
        else:
            return np.asarray(array)
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU加速矩阵乘法"""
        with self.performance_timer("矩阵乘法"):
            if self.compute_backend == "cupy":
                a_gpu = cp.asarray(a)
                b_gpu = cp.asarray(b)
                result_gpu = cp.dot(a_gpu, b_gpu)
                return result_gpu.get()
            
            elif self.compute_backend == "opencl":
                # OpenCL矩阵乘法实现
                return self._opencl_matrix_multiply(a, b)
            
            elif self.compute_backend == "numba_cuda":
                return self._numba_cuda_matrix_multiply(a, b)
            
            elif self.compute_backend == "numba_cpu":
                return self._numba_cpu_matrix_multiply(a, b)
            
            else:
                return np.dot(a, b)
    
    def correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """GPU加速相关性矩阵计算"""
        with self.performance_timer("相关性矩阵"):
            if self.compute_backend == "cupy":
                data_gpu = cp.asarray(data)
                result_gpu = cp.corrcoef(data_gpu.T)
                return result_gpu.get()
            
            elif self.compute_backend == "numba_cpu":
                return self._numba_correlation_matrix(data)
            
            else:
                return np.corrcoef(data.T)
    
    def pca_transform(self, data: np.ndarray, n_components: int) -> tuple:
        """GPU加速PCA变换"""
        with self.performance_timer("PCA变换"):
            if self.compute_backend == "cupy":
                data_gpu = cp.asarray(data)
                
                # 中心化数据
                mean_gpu = cp.mean(data_gpu, axis=0)
                centered_gpu = data_gpu - mean_gpu
                
                # 计算协方差矩阵
                cov_gpu = cp.cov(centered_gpu.T)
                
                # 特征值分解
                eigenvals_gpu, eigenvecs_gpu = cp.linalg.eigh(cov_gpu)
                
                # 按特征值排序
                idx = cp.argsort(eigenvals_gpu)[::-1]
                eigenvals_gpu = eigenvals_gpu[idx]
                eigenvecs_gpu = eigenvecs_gpu[:, idx]
                
                # 选择主成分
                components_gpu = eigenvecs_gpu[:, :n_components]
                
                # 变换数据
                transformed_gpu = cp.dot(centered_gpu, components_gpu)
                
                return (transformed_gpu.get(), 
                       components_gpu.get(), 
                       eigenvals_gpu[:n_components].get())
            
            else:
                return self._cpu_pca_transform(data, n_components)
    
    def kmeans_clustering(self, data: np.ndarray, n_clusters: int, 
                         max_iters: int = 100) -> tuple:
        """GPU加速K-means聚类"""
        with self.performance_timer("K-means聚类"):
            if self.compute_backend == "cupy":
                return self._cupy_kmeans(data, n_clusters, max_iters)
            elif self.compute_backend == "numba_cpu":
                return self._numba_kmeans(data, n_clusters, max_iters)
            else:
                return self._cpu_kmeans(data, n_clusters, max_iters)
    
    def statistical_features(self, data: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray]:
        """GPU加速统计特征计算"""
        with self.performance_timer("统计特征计算"):
            if self.compute_backend == "cupy":
                data_gpu = cp.asarray(data)
                
                features = {
                    'mean': cp.mean(data_gpu, axis=axis).get(),
                    'std': cp.std(data_gpu, axis=axis).get(),
                    'var': cp.var(data_gpu, axis=axis).get(),
                    'min': cp.min(data_gpu, axis=axis).get(),
                    'max': cp.max(data_gpu, axis=axis).get(),
                    'median': cp.median(data_gpu, axis=axis).get(),
                    'sum': cp.sum(data_gpu, axis=axis).get()
                }
                
                return features
            
            else:
                return {
                    'mean': np.mean(data, axis=axis),
                    'std': np.std(data, axis=axis),
                    'var': np.var(data, axis=axis),
                    'min': np.min(data, axis=axis),
                    'max': np.max(data, axis=axis),
                    'median': np.median(data, axis=axis),
                    'sum': np.sum(data, axis=axis)
                }
    
    def _opencl_matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """OpenCL矩阵乘法实现"""
        try:
            a_gpu = cl_array.to_device(self.queue, a.astype(np.float32))
            b_gpu = cl_array.to_device(self.queue, b.astype(np.float32))
            
            result_shape = (a.shape[0], b.shape[1])
            result_gpu = cl_array.empty(self.queue, result_shape, np.float32)
            
            # OpenCL内核代码
            kernel_code = """
            __kernel void matrix_multiply(__global const float* A,
                                        __global const float* B,
                                        __global float* C,
                                        const int M, const int N, const int K) {
                int i = get_global_id(0);
                int j = get_global_id(1);
                
                if (i < M && j < N) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
            """
            
            program = cl.Program(self.context, kernel_code).build()
            kernel = program.matrix_multiply
            
            kernel(self.queue, result_shape, None,
                  a_gpu.data, b_gpu.data, result_gpu.data,
                  np.int32(a.shape[0]), np.int32(b.shape[1]), np.int32(a.shape[1]))
            
            return result_gpu.get()
            
        except Exception as e:
            logger.warning(f"OpenCL矩阵乘法失败，回退到CPU: {e}")
            return np.dot(a, b)
    
    def _numba_cuda_matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Numba CUDA矩阵乘法"""
        try:
            @cuda.jit
            def cuda_matrix_multiply(A, B, C):
                i, j = cuda.grid(2)
                if i < C.shape[0] and j < C.shape[1]:
                    tmp = 0.0
                    for k in range(A.shape[1]):
                        tmp += A[i, k] * B[k, j]
                    C[i, j] = tmp
            
            # 分配GPU内存
            a_gpu = cuda.to_device(a.astype(np.float32))
            b_gpu = cuda.to_device(b.astype(np.float32))
            c_gpu = cuda.device_array((a.shape[0], b.shape[1]), dtype=np.float32)
            
            # 配置线程块和网格
            threads_per_block = (16, 16)
            blocks_per_grid_x = (a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (b.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # 执行内核
            cuda_matrix_multiply[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
            
            return c_gpu.copy_to_host()
            
        except Exception as e:
            logger.warning(f"Numba CUDA矩阵乘法失败，回退到CPU: {e}")
            return np.dot(a, b)
    
    def _numba_cpu_matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Numba CPU优化矩阵乘法"""
        @jit(nopython=True, parallel=True)
        def numba_dot(A, B):
            return np.dot(A, B)
        
        return numba_dot(a, b)
    
    def _numba_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Numba优化相关性矩阵"""
        @jit(nopython=True, parallel=True)
        def correlation_matrix(X):
            n, p = X.shape
            C = np.zeros((p, p))
            
            # 计算均值
            means = np.zeros(p)
            for j in prange(p):
                for i in range(n):
                    means[j] += X[i, j]
                means[j] /= n
            
            # 计算协方差
            for i in prange(p):
                for j in range(i, p):
                    cov = 0.0
                    for k in range(n):
                        cov += (X[k, i] - means[i]) * (X[k, j] - means[j])
                    cov /= (n - 1)
                    C[i, j] = cov
                    C[j, i] = cov
            
            # 转换为相关系数
            for i in prange(p):
                for j in range(p):
                    if i != j:
                        C[i, j] /= np.sqrt(C[i, i] * C[j, j])
            
            # 对角线设为1
            for i in range(p):
                C[i, i] = 1.0
            
            return C
        
        return correlation_matrix(data)
    
    def _cupy_kmeans(self, data: np.ndarray, n_clusters: int, max_iters: int) -> tuple:
        """CuPy K-means实现"""
        data_gpu = cp.asarray(data)
        n_samples, n_features = data.shape
        
        # 随机初始化聚类中心
        centroids_gpu = cp.random.rand(n_clusters, n_features)
        labels_gpu = cp.zeros(n_samples, dtype=cp.int32)
        
        for iteration in range(max_iters):
            # 计算距离并分配标签
            distances = cp.linalg.norm(
                data_gpu[:, cp.newaxis, :] - centroids_gpu[cp.newaxis, :, :],
                axis=2
            )
            new_labels = cp.argmin(distances, axis=1)
            
            # 检查收敛
            if cp.array_equal(labels_gpu, new_labels):
                break
            
            labels_gpu = new_labels
            
            # 更新聚类中心
            for k in range(n_clusters):
                mask = labels_gpu == k
                if cp.sum(mask) > 0:
                    centroids_gpu[k] = cp.mean(data_gpu[mask], axis=0)
        
        return labels_gpu.get(), centroids_gpu.get()
    
    def _numba_kmeans(self, data: np.ndarray, n_clusters: int, max_iters: int) -> tuple:
        """Numba K-means实现"""
        @jit(nopython=True)
        def kmeans_numba(X, k, max_iter):
            n, d = X.shape
            centroids = np.random.rand(k, d)
            labels = np.zeros(n, dtype=np.int32)
            
            for iteration in range(max_iter):
                # 分配标签
                for i in range(n):
                    min_dist = np.inf
                    for j in range(k):
                        dist = 0.0
                        for l in range(d):
                            dist += (X[i, l] - centroids[j, l]) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            labels[i] = j
                
                # 更新聚类中心
                for j in range(k):
                    count = 0
                    for l in range(d):
                        centroids[j, l] = 0.0
                    
                    for i in range(n):
                        if labels[i] == j:
                            count += 1
                            for l in range(d):
                                centroids[j, l] += X[i, l]
                    
                    if count > 0:
                        for l in range(d):
                            centroids[j, l] /= count
            
            return labels, centroids
        
        return kmeans_numba(data, n_clusters, max_iters)
    
    def _cpu_pca_transform(self, data: np.ndarray, n_components: int) -> tuple:
        """CPU PCA实现"""
        # 中心化数据
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered.T)
        
        # 特征值分解
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 按特征值排序
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # 选择主成分
        components = eigenvecs[:, :n_components]
        
        # 变换数据
        transformed = np.dot(centered, components)
        
        return transformed, components, eigenvals[:n_components]
    
    def _cpu_kmeans(self, data: np.ndarray, n_clusters: int, max_iters: int) -> tuple:
        """CPU K-means实现"""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iters, random_state=42)
        labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_
        
        return labels, centroids
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'backend': self.compute_backend,
            'prefer_gpu': self.prefer_gpu,
            'device_id': self.device_id
        }
        
        if self.compute_backend == "cupy":
            try:
                device = cp.cuda.Device()
                info.update({
                    'device_name': device.attributes['Name'],
                    'compute_capability': device.compute_capability,
                    'total_memory': device.mem_info[1],
                    'free_memory': device.mem_info[0]
                })
            except:
                pass
        
        elif self.compute_backend == "opencl" and self.context:
            try:
                device = self.context.devices[0]
                info.update({
                    'device_name': device.name,
                    'device_type': device.type,
                    'max_work_group_size': device.max_work_group_size,
                    'global_mem_size': device.global_mem_size
                })
            except:
                pass
        
        return info
    
    def benchmark_performance(self, data_size: int = 1000) -> Dict[str, float]:
        """性能基准测试"""
        logger.info(f"开始性能基准测试，数据大小: {data_size}x{data_size}")
        
        # 生成测试数据
        data = np.random.rand(data_size, data_size).astype(np.float32)
        
        results = {}
        
        # 矩阵乘法测试
        start_time = time.time()
        _ = self.matrix_multiply(data, data)
        results['matrix_multiply'] = time.time() - start_time
        
        # 统计特征测试
        start_time = time.time()
        _ = self.statistical_features(data)
        results['statistical_features'] = time.time() - start_time
        
        # 相关性矩阵测试（小数据集）
        small_data = data[:min(500, data_size), :min(100, data_size)]
        start_time = time.time()
        _ = self.correlation_matrix(small_data)
        results['correlation_matrix'] = time.time() - start_time
        
        logger.info(f"性能基准测试完成: {results}")
        return results


# 全局GPU加速器实例
_gpu_accelerator = None

def get_gpu_accelerator(prefer_gpu: bool = True, device_id: int = 0) -> GPUAccelerator:
    """获取GPU加速器实例（单例模式）"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(prefer_gpu, device_id)
    return _gpu_accelerator

def is_gpu_available() -> bool:
    """检查GPU是否可用"""
    return CUPY_AVAILABLE or OPENCL_AVAILABLE or (NUMBA_AVAILABLE and cuda and cuda.is_available())

def get_available_backends() -> List[str]:
    """获取可用的计算后端"""
    backends = ["numpy"]
    
    if CUPY_AVAILABLE:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                backends.append("cupy")
        except:
            pass
    
    if OPENCL_AVAILABLE:
        try:
            if cl.get_platforms():
                backends.append("opencl")
        except:
            pass
    
    if NUMBA_AVAILABLE:
        backends.append("numba_cpu")
        if cuda and cuda.is_available():
            backends.append("numba_cuda")
    
    return backends
