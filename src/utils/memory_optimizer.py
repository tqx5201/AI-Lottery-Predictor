"""
内存使用优化模块
包含内存监控、对象池、懒加载、数据压缩等优化技术
"""

import gc
import sys
import psutil
import numpy as np
import pandas as pd
import pickle
import gzip
import lzma
import threading
import weakref
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from functools import wraps, lru_cache
from collections import defaultdict, OrderedDict
import logging
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.memory_history = []
        self.monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        try:
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            info = {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': self.process.memory_percent(),
                'available': virtual_memory.available / 1024 / 1024,  # MB
                'total': virtual_memory.total / 1024 / 1024,  # MB
                'system_percent': virtual_memory.percent
            }
            
            # 更新峰值
            if info['rss'] > self.peak_memory:
                self.peak_memory = info['rss']
            
            return info
            
        except Exception as e:
            logger.warning(f"获取内存信息失败: {e}")
            return {}
    
    def start_monitoring(self, interval: float = 1.0):
        """开始内存监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    info = self.get_memory_info()
                    if info:
                        with self._lock:
                            self.memory_history.append({
                                'timestamp': time.time(),
                                'memory': info['rss']
                            })
                            
                            # 保持历史记录在合理范围内
                            if len(self.memory_history) > 1000:
                                self.memory_history = self.memory_history[-500:]
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"内存监控错误: {e}")
                    break
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("内存监控已停止")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        current_info = self.get_memory_info()
        
        with self._lock:
            if self.memory_history:
                memory_values = [h['memory'] for h in self.memory_history]
                stats = {
                    'current_memory': current_info.get('rss', 0),
                    'peak_memory': self.peak_memory,
                    'average_memory': np.mean(memory_values),
                    'min_memory': np.min(memory_values),
                    'max_memory': np.max(memory_values),
                    'memory_trend': memory_values[-10:] if len(memory_values) >= 10 else memory_values,
                    'system_info': {
                        'available': current_info.get('available', 0),
                        'total': current_info.get('total', 0),
                        'system_percent': current_info.get('system_percent', 0)
                    }
                }
            else:
                stats = {
                    'current_memory': current_info.get('rss', 0),
                    'peak_memory': self.peak_memory,
                    'system_info': {
                        'available': current_info.get('available', 0),
                        'total': current_info.get('total', 0),
                        'system_percent': current_info.get('system_percent', 0)
                    }
                }
        
        return stats


class ObjectPool(Generic[T]):
    """对象池"""
    
    def __init__(self, factory: Callable[[], T], max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool = []
        self.lock = threading.Lock()
    
    def get(self) -> T:
        """获取对象"""
        with self.lock:
            if self.pool:
                return self.pool.pop()
            else:
                return self.factory()
    
    def put(self, obj: T):
        """归还对象"""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(obj)
    
    @contextmanager
    def get_object(self):
        """上下文管理器方式获取对象"""
        obj = self.get()
        try:
            yield obj
        finally:
            self.put(obj)


class LazyLoader:
    """懒加载器"""
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._data = None
        self._loaded = False
        self._lock = threading.Lock()
    
    def load(self):
        """加载数据"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._data = self.loader_func(*self.args, **self.kwargs)
                    self._loaded = True
        return self._data
    
    def unload(self):
        """卸载数据"""
        with self._lock:
            self._data = None
            self._loaded = False
            gc.collect()
    
    def is_loaded(self) -> bool:
        """检查是否已加载"""
        return self._loaded
    
    @property
    def data(self):
        """获取数据（自动加载）"""
        return self.load()


class DataCompressor:
    """数据压缩器"""
    
    @staticmethod
    def compress_array(array: np.ndarray, method: str = 'gzip') -> bytes:
        """压缩numpy数组"""
        try:
            # 序列化数组
            array_bytes = pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL)
            
            if method == 'gzip':
                return gzip.compress(array_bytes)
            elif method == 'lzma':
                return lzma.compress(array_bytes)
            else:
                return array_bytes
                
        except Exception as e:
            logger.error(f"数组压缩失败: {e}")
            return pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def decompress_array(compressed_data: bytes, method: str = 'gzip') -> np.ndarray:
        """解压numpy数组"""
        try:
            if method == 'gzip':
                array_bytes = gzip.decompress(compressed_data)
            elif method == 'lzma':
                array_bytes = lzma.decompress(compressed_data)
            else:
                array_bytes = compressed_data
            
            return pickle.loads(array_bytes)
            
        except Exception as e:
            logger.error(f"数组解压失败: {e}")
            return pickle.loads(compressed_data)
    
    @staticmethod
    def compress_dataframe(df: pd.DataFrame, method: str = 'gzip') -> bytes:
        """压缩DataFrame"""
        try:
            # 优化DataFrame存储
            df_optimized = df.copy()
            
            # 优化数值列
            for col in df_optimized.select_dtypes(include=[np.number]).columns:
                if df_optimized[col].dtype == np.float64:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                elif df_optimized[col].dtype == np.int64:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            
            # 优化字符串列
            for col in df_optimized.select_dtypes(include=['object']).columns:
                if df_optimized[col].dtype == 'object':
                    try:
                        df_optimized[col] = df_optimized[col].astype('category')
                    except:
                        pass
            
            # 序列化
            df_bytes = pickle.dumps(df_optimized, protocol=pickle.HIGHEST_PROTOCOL)
            
            if method == 'gzip':
                return gzip.compress(df_bytes)
            elif method == 'lzma':
                return lzma.compress(df_bytes)
            else:
                return df_bytes
                
        except Exception as e:
            logger.error(f"DataFrame压缩失败: {e}")
            return pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def decompress_dataframe(compressed_data: bytes, method: str = 'gzip') -> pd.DataFrame:
        """解压DataFrame"""
        try:
            if method == 'gzip':
                df_bytes = gzip.decompress(compressed_data)
            elif method == 'lzma':
                df_bytes = lzma.decompress(compressed_data)
            else:
                df_bytes = compressed_data
            
            return pickle.loads(df_bytes)
            
        except Exception as e:
            logger.error(f"DataFrame解压失败: {e}")
            return pickle.loads(compressed_data)


class MemoryEfficientCache:
    """内存高效缓存"""
    
    def __init__(self, max_memory_mb: float = 100, compression_threshold: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.compression_threshold = compression_threshold  # bytes
        self.cache = OrderedDict()
        self.compressed_cache = {}
        self.sizes = {}
        self.current_memory = 0
        self.lock = threading.Lock()
        self.compressor = DataCompressor()
    
    def _estimate_size(self, obj: Any) -> int:
        """估算对象大小"""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            else:
                return sys.getsizeof(obj)
        except:
            return sys.getsizeof(obj)
    
    def _should_compress(self, size: int) -> bool:
        """判断是否应该压缩"""
        return size > self.compression_threshold
    
    def _evict_if_needed(self):
        """如果需要则驱逐数据"""
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        
        while self.current_memory > max_memory_bytes and (self.cache or self.compressed_cache):
            # 优先驱逐未压缩的旧数据
            if self.cache:
                key = next(iter(self.cache))
                self._remove_from_cache(key)
            elif self.compressed_cache:
                key = next(iter(self.compressed_cache))
                self._remove_from_compressed_cache(key)
    
    def _remove_from_cache(self, key: str):
        """从普通缓存中移除"""
        if key in self.cache:
            del self.cache[key]
            if key in self.sizes:
                self.current_memory -= self.sizes[key]
                del self.sizes[key]
    
    def _remove_from_compressed_cache(self, key: str):
        """从压缩缓存中移除"""
        if key in self.compressed_cache:
            del self.compressed_cache[key]
            if key in self.sizes:
                self.current_memory -= self.sizes[key]
                del self.sizes[key]
    
    def put(self, key: str, value: Any):
        """存储数据"""
        with self.lock:
            # 移除旧数据
            self._remove_from_cache(key)
            self._remove_from_compressed_cache(key)
            
            # 估算大小
            size = self._estimate_size(value)
            
            if self._should_compress(size):
                # 压缩存储
                try:
                    if isinstance(value, np.ndarray):
                        compressed = self.compressor.compress_array(value)
                    elif isinstance(value, pd.DataFrame):
                        compressed = self.compressor.compress_dataframe(value)
                    else:
                        compressed = gzip.compress(pickle.dumps(value))
                    
                    compressed_size = len(compressed)
                    self.compressed_cache[key] = {
                        'data': compressed,
                        'type': type(value).__name__,
                        'original_size': size
                    }
                    self.sizes[key] = compressed_size
                    self.current_memory += compressed_size
                    
                except Exception as e:
                    logger.warning(f"压缩失败，使用普通存储: {e}")
                    self.cache[key] = value
                    self.sizes[key] = size
                    self.current_memory += size
            else:
                # 普通存储
                self.cache[key] = value
                self.sizes[key] = size
                self.current_memory += size
            
            # 驱逐旧数据
            self._evict_if_needed()
    
    def get(self, key: str) -> Optional[Any]:
        """获取数据"""
        with self.lock:
            # 检查普通缓存
            if key in self.cache:
                # 移到末尾（LRU）
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            
            # 检查压缩缓存
            if key in self.compressed_cache:
                try:
                    compressed_data = self.compressed_cache[key]
                    data_type = compressed_data['type']
                    compressed = compressed_data['data']
                    
                    # 解压数据
                    if data_type == 'ndarray':
                        value = self.compressor.decompress_array(compressed)
                    elif data_type == 'DataFrame':
                        value = self.compressor.decompress_dataframe(compressed)
                    else:
                        value = pickle.loads(gzip.decompress(compressed))
                    
                    # 移动到普通缓存（热数据）
                    del self.compressed_cache[key]
                    compressed_size = self.sizes[key]
                    
                    new_size = self._estimate_size(value)
                    self.cache[key] = value
                    self.sizes[key] = new_size
                    self.current_memory = self.current_memory - compressed_size + new_size
                    
                    return value
                    
                except Exception as e:
                    logger.error(f"解压数据失败: {e}")
                    # 清理损坏的数据
                    self._remove_from_compressed_cache(key)
        
        return None
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.compressed_cache.clear()
            self.sizes.clear()
            self.current_memory = 0
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_items = len(self.cache) + len(self.compressed_cache)
            compressed_items = len(self.compressed_cache)
            
            return {
                'total_items': total_items,
                'uncompressed_items': len(self.cache),
                'compressed_items': compressed_items,
                'compression_ratio': compressed_items / max(total_items, 1),
                'current_memory_mb': self.current_memory / 1024 / 1024,
                'max_memory_mb': self.max_memory_mb,
                'memory_usage_ratio': (self.current_memory / 1024 / 1024) / self.max_memory_mb
            }


class MemoryOptimizer:
    """内存优化器主类"""
    
    def __init__(self):
        self.monitor = MemoryMonitor()
        self.cache = MemoryEfficientCache()
        self.object_pools = {}
        self.lazy_loaders = weakref.WeakValueDictionary()
        
        # 启动内存监控
        self.monitor.start_monitoring()
        
        logger.info("内存优化器初始化完成")
    
    def create_object_pool(self, name: str, factory: Callable, max_size: int = 100):
        """创建对象池"""
        self.object_pools[name] = ObjectPool(factory, max_size)
        return self.object_pools[name]
    
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """获取对象池"""
        return self.object_pools.get(name)
    
    def create_lazy_loader(self, name: str, loader_func: Callable, *args, **kwargs) -> LazyLoader:
        """创建懒加载器"""
        loader = LazyLoader(loader_func, *args, **kwargs)
        self.lazy_loaders[name] = loader
        return loader
    
    def get_lazy_loader(self, name: str) -> Optional[LazyLoader]:
        """获取懒加载器"""
        return self.lazy_loaders.get(name)
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        try:
            original_memory = df.memory_usage(deep=True).sum()
            optimized_df = df.copy()
            
            # 优化数值列
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_min = optimized_df[col].min()
                col_max = optimized_df[col].max()
                
                if optimized_df[col].dtype == np.float64:
                    if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                
                elif optimized_df[col].dtype == np.int64:
                    if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
            
            # 优化字符串列
            for col in optimized_df.select_dtypes(include=['object']).columns:
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])
                
                if num_unique_values / num_total_values < 0.5:  # 如果重复值较多
                    optimized_df[col] = optimized_df[col].astype('category')
            
            # 优化日期时间列
            for col in optimized_df.select_dtypes(include=['datetime64']).columns:
                optimized_df[col] = pd.to_datetime(optimized_df[col])
            
            new_memory = optimized_df.memory_usage(deep=True).sum()
            reduction = (original_memory - new_memory) / original_memory * 100
            
            logger.info(f"DataFrame内存优化完成，减少 {reduction:.1f}% ({original_memory/1024/1024:.1f}MB -> {new_memory/1024/1024:.1f}MB)")
            
            return optimized_df
            
        except Exception as e:
            logger.error(f"DataFrame内存优化失败: {e}")
            return df
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        before_memory = self.monitor.get_memory_info().get('rss', 0)
        
        # 执行垃圾回收
        collected = gc.collect()
        
        after_memory = self.monitor.get_memory_info().get('rss', 0)
        freed_memory = before_memory - after_memory
        
        logger.info(f"垃圾回收完成，回收对象: {collected}, 释放内存: {freed_memory:.1f}MB")
        
        return {
            'collected_objects': collected,
            'freed_memory_mb': freed_memory,
            'before_memory_mb': before_memory,
            'after_memory_mb': after_memory
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = {
            'memory_monitor': self.monitor.get_memory_stats(),
            'cache_stats': self.cache.get_stats(),
            'object_pools': {
                name: {
                    'pool_size': len(pool.pool),
                    'max_size': pool.max_size
                }
                for name, pool in self.object_pools.items()
            },
            'lazy_loaders': {
                name: {
                    'loaded': loader.is_loaded()
                }
                for name, loader in self.lazy_loaders.items()
            },
            'gc_stats': {
                'generation_0': gc.get_count()[0],
                'generation_1': gc.get_count()[1],
                'generation_2': gc.get_count()[2],
                'gc_stats': gc.get_stats()
            }
        }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        self.monitor.stop_monitoring()
        self.cache.clear()
        self.object_pools.clear()
        self.lazy_loaders.clear()
        self.force_garbage_collection()
        
        logger.info("内存优化器清理完成")


# 装饰器
def memory_efficient(cache_result: bool = True, max_memory_mb: float = 50):
    """内存高效装饰器"""
    def decorator(func):
        if cache_result:
            cache = MemoryEfficientCache(max_memory_mb=max_memory_mb)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cache_result:
                # 生成缓存键
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # 检查缓存
                result = cache.get(cache_key)
                if result is not None:
                    return result
            
            # 执行函数
            with memory_context():
                result = func(*args, **kwargs)
            
            # 缓存结果
            if cache_result:
                cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def memory_context():
    """内存上下文管理器"""
    before_memory = psutil.Process().memory_info().rss
    try:
        yield
    finally:
        after_memory = psutil.Process().memory_info().rss
        memory_diff = (after_memory - before_memory) / 1024 / 1024
        if memory_diff > 10:  # 如果内存增长超过10MB
            logger.warning(f"内存使用增长: {memory_diff:.1f}MB")
            gc.collect()


# 全局内存优化器实例
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """获取内存优化器实例（单例模式）"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer
