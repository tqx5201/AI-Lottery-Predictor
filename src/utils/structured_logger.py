"""
结构化日志系统
提供高性能、可配置的结构化日志记录功能
"""

import logging
import json
import time
import os
import sys
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import atexit
from contextlib import contextmanager
import traceback
import psutil


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """日志格式"""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    extra_fields: Optional[Dict[str, Any]] = None
    exception_info: Optional[Dict[str, Any]] = None
    performance_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['timestamp_iso'] = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.active_operations = {}
        self.lock = threading.Lock()
    
    def start_operation(self, operation_id: str, operation_name: str, 
                       context: Optional[Dict[str, Any]] = None):
        """开始操作跟踪"""
        with self.lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': time.time(),
                'context': context or {}
            }
    
    def end_operation(self, operation_id: str, 
                     result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """结束操作跟踪"""
        with self.lock:
            operation = self.active_operations.pop(operation_id, None)
            if operation:
                duration = time.time() - operation['start_time']
                return {
                    'operation_name': operation['name'],
                    'duration_seconds': duration,
                    'start_time': operation['start_time'],
                    'context': operation['context'],
                    'result': result or {}
                }
        return None


class LogFilter:
    """日志过滤器"""
    
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """添加过滤函数"""
        self.filters.append(filter_func)
    
    def should_log(self, log_entry: LogEntry) -> bool:
        """检查是否应该记录日志"""
        for filter_func in self.filters:
            try:
                if not filter_func(log_entry):
                    return False
            except Exception:
                # 过滤器异常时默认允许记录
                pass
        return True


class AsyncLogWriter:
    """异步日志写入器"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self.writers = []
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
    
    def add_writer(self, writer: Callable[[LogEntry], None]):
        """添加写入器"""
        with self.lock:
            self.writers.append(writer)
    
    def start(self):
        """启动异步写入"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """停止异步写入"""
        self.running = False
        if self.worker_thread:
            # 发送停止信号
            try:
                self.log_queue.put(None, timeout=1)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=5)
    
    def write_log(self, log_entry: LogEntry):
        """写入日志"""
        try:
            self.log_queue.put(log_entry, block=False)
        except queue.Full:
            # 队列满时丢弃最旧的日志
            try:
                self.log_queue.get(block=False)
                self.log_queue.put(log_entry, block=False)
            except (queue.Empty, queue.Full):
                pass
    
    def _worker_loop(self):
        """工作循环"""
        while self.running:
            try:
                log_entry = self.log_queue.get(timeout=1)
                if log_entry is None:  # 停止信号
                    break
                
                # 写入到所有写入器
                with self.lock:
                    for writer in self.writers:
                        try:
                            writer(log_entry)
                        except Exception as e:
                            # 避免日志写入错误导致程序崩溃
                            print(f"日志写入器错误: {e}", file=sys.stderr)
                
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"日志工作线程错误: {e}", file=sys.stderr)


class FileWriter:
    """文件写入器"""
    
    def __init__(self, file_path: str, log_format: LogFormat = LogFormat.JSON,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 5):
        self.file_path = Path(file_path)
        self.log_format = log_format
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # 确保目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        self.handler = RotatingFileHandler(
            self.file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        self.lock = threading.Lock()
    
    def write(self, log_entry: LogEntry):
        """写入日志条目"""
        with self.lock:
            try:
                if self.log_format == LogFormat.JSON:
                    message = log_entry.to_json()
                elif self.log_format == LogFormat.STRUCTURED:
                    message = self._format_structured(log_entry)
                else:
                    message = self._format_text(log_entry)
                
                # 创建日志记录
                record = logging.LogRecord(
                    name=log_entry.logger_name,
                    level=getattr(logging, log_entry.level),
                    pathname="",
                    lineno=log_entry.line_number or 0,
                    msg=message,
                    args=(),
                    exc_info=None
                )
                
                self.handler.emit(record)
                
            except Exception as e:
                print(f"文件写入错误: {e}", file=sys.stderr)
    
    def _format_structured(self, log_entry: LogEntry) -> str:
        """格式化结构化文本"""
        timestamp = datetime.fromtimestamp(log_entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        parts = [
            f"[{timestamp}]",
            f"[{log_entry.level}]",
            f"[{log_entry.logger_name}]",
            log_entry.message
        ]
        
        if log_entry.extra_fields:
            extra_str = json.dumps(log_entry.extra_fields, ensure_ascii=False)
            parts.append(f"EXTRA: {extra_str}")
        
        if log_entry.performance_info:
            perf_str = json.dumps(log_entry.performance_info, ensure_ascii=False)
            parts.append(f"PERF: {perf_str}")
        
        if log_entry.exception_info:
            exc_str = json.dumps(log_entry.exception_info, ensure_ascii=False)
            parts.append(f"EXCEPTION: {exc_str}")
        
        return " ".join(parts)
    
    def _format_text(self, log_entry: LogEntry) -> str:
        """格式化纯文本"""
        timestamp = datetime.fromtimestamp(log_entry.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return f"{timestamp} [{log_entry.level}] {log_entry.logger_name}: {log_entry.message}"


class ConsoleWriter:
    """控制台写入器"""
    
    def __init__(self, log_format: LogFormat = LogFormat.TEXT, 
                 colored: bool = True):
        self.log_format = log_format
        self.colored = colored
        self.lock = threading.Lock()
        
        # 颜色映射
        self.colors = {
            'DEBUG': '\033[36m',    # 青色
            'INFO': '\033[32m',     # 绿色
            'WARNING': '\033[33m',  # 黄色
            'ERROR': '\033[31m',    # 红色
            'CRITICAL': '\033[35m', # 紫色
            'RESET': '\033[0m'      # 重置
        } if colored else {}
    
    def write(self, log_entry: LogEntry):
        """写入日志条目"""
        with self.lock:
            try:
                if self.log_format == LogFormat.JSON:
                    message = log_entry.to_json()
                else:
                    message = self._format_colored_text(log_entry)
                
                print(message, file=sys.stdout if log_entry.level in ['DEBUG', 'INFO'] else sys.stderr)
                
            except Exception as e:
                print(f"控制台写入错误: {e}", file=sys.stderr)
    
    def _format_colored_text(self, log_entry: LogEntry) -> str:
        """格式化彩色文本"""
        timestamp = datetime.fromtimestamp(log_entry.timestamp).strftime('%H:%M:%S.%f')[:-3]
        
        if self.colored and log_entry.level in self.colors:
            color = self.colors[log_entry.level]
            reset = self.colors['RESET']
            level_str = f"{color}[{log_entry.level}]{reset}"
        else:
            level_str = f"[{log_entry.level}]"
        
        return f"{timestamp} {level_str} {log_entry.logger_name}: {log_entry.message}"


class StructuredLogger:
    """结构化日志器"""
    
    def __init__(self, name: str):
        self.name = name
        self.extra_fields = {}
        self.performance_tracker = PerformanceTracker()
        self.log_filter = LogFilter()
        
        # 获取全局配置
        self.config = get_logger_config()
        
        # 获取异步写入器
        self.async_writer = get_async_writer()
    
    def set_extra_fields(self, **fields):
        """设置额外字段"""
        self.extra_fields.update(fields)
    
    def clear_extra_fields(self):
        """清除额外字段"""
        self.extra_fields.clear()
    
    def _create_log_entry(self, level: str, message: str, 
                         extra: Optional[Dict[str, Any]] = None,
                         exc_info: Optional[Exception] = None,
                         performance_info: Optional[Dict[str, Any]] = None) -> LogEntry:
        """创建日志条目"""
        # 获取调用信息
        frame = sys._getframe(3)  # 跳过内部调用层级
        
        # 合并额外字段
        merged_extra = self.extra_fields.copy()
        if extra:
            merged_extra.update(extra)
        
        # 处理异常信息
        exception_info = None
        if exc_info:
            exception_info = {
                'type': type(exc_info).__name__,
                'message': str(exc_info),
                'traceback': traceback.format_exc()
            }
        
        return LogEntry(
            timestamp=time.time(),
            level=level,
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            extra_fields=merged_extra if merged_extra else None,
            exception_info=exception_info,
            performance_info=performance_info
        )
    
    def _log(self, level: str, message: str, **kwargs):
        """内部日志方法"""
        if not self._should_log(level):
            return
        
        log_entry = self._create_log_entry(level, message, **kwargs)
        
        # 应用过滤器
        if not self.log_filter.should_log(log_entry):
            return
        
        # 异步写入
        self.async_writer.write_log(log_entry)
    
    def _should_log(self, level: str) -> bool:
        """检查是否应该记录日志"""
        level_values = {
            'DEBUG': 10,
            'INFO': 20,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50
        }
        
        current_level = level_values.get(level, 20)
        if hasattr(self.config.min_level, 'value'):
            min_level = level_values.get(self.config.min_level.value, 20)
        else:
            min_level = level_values.get(self.config.min_level, 20)
        
        return current_level >= min_level
    
    def debug(self, message: str, **kwargs):
        """记录DEBUG日志"""
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录INFO日志"""
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录WARNING日志"""
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """记录ERROR日志"""
        self._log('ERROR', message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """记录CRITICAL日志"""
        self._log('CRITICAL', message, exc_info=exc_info, **kwargs)
    
    @contextmanager
    def performance_context(self, operation_name: str, 
                          context: Optional[Dict[str, Any]] = None):
        """性能跟踪上下文"""
        operation_id = f"{self.name}_{operation_name}_{time.time()}"
        
        self.performance_tracker.start_operation(operation_id, operation_name, context)
        
        try:
            yield
        except Exception as e:
            # 记录异常但继续跟踪性能
            self.error(f"操作 {operation_name} 发生异常", exc_info=e)
            raise
        finally:
            perf_info = self.performance_tracker.end_operation(operation_id)
            if perf_info:
                self.info(f"操作完成: {operation_name}", performance_info=perf_info)
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """添加日志过滤器"""
        self.log_filter.add_filter(filter_func)


@dataclass
class LoggerConfig:
    """日志器配置"""
    min_level: LogLevel = LogLevel.INFO
    console_enabled: bool = True
    console_format: LogFormat = LogFormat.TEXT
    console_colored: bool = True
    file_enabled: bool = True
    file_path: str = "logs/app.log"
    file_format: LogFormat = LogFormat.JSON
    file_max_size: int = 100 * 1024 * 1024  # 100MB
    file_backup_count: int = 5
    async_enabled: bool = True
    async_queue_size: int = 10000


class LoggerManager:
    """日志器管理器"""
    
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.loggers = {}
        self.async_writer = None
        self.lock = threading.Lock()
        
        self._initialize()
    
    def _initialize(self):
        """初始化日志系统"""
        if self.config.async_enabled:
            self.async_writer = AsyncLogWriter(self.config.async_queue_size)
            
            # 添加写入器
            if self.config.console_enabled:
                console_writer = ConsoleWriter(
                    self.config.console_format,
                    self.config.console_colored
                )
                self.async_writer.add_writer(console_writer.write)
            
            if self.config.file_enabled:
                file_writer = FileWriter(
                    self.config.file_path,
                    self.config.file_format,
                    self.config.file_max_size,
                    self.config.file_backup_count
                )
                self.async_writer.add_writer(file_writer.write)
            
            self.async_writer.start()
            
            # 注册清理函数
            atexit.register(self.cleanup)
    
    def get_logger(self, name: str) -> StructuredLogger:
        """获取日志器"""
        with self.lock:
            if name not in self.loggers:
                self.loggers[name] = StructuredLogger(name)
            return self.loggers[name]
    
    def cleanup(self):
        """清理资源"""
        if self.async_writer:
            self.async_writer.stop()


# 全局配置和管理器
_logger_config = LoggerConfig()
_logger_manager = None

def configure_logging(config: LoggerConfig):
    """配置日志系统"""
    global _logger_config, _logger_manager
    _logger_config = config
    _logger_manager = LoggerManager(config)

def get_logger_config() -> LoggerConfig:
    """获取日志配置"""
    return _logger_config

def get_async_writer() -> AsyncLogWriter:
    """获取异步写入器"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager(_logger_config)
    return _logger_manager.async_writer

def get_structured_logger(name: str) -> StructuredLogger:
    """获取结构化日志器"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager(_logger_config)
    return _logger_manager.get_logger(name)


# 便捷函数
def setup_default_logging(log_dir: str = "logs", log_level: LogLevel = LogLevel.INFO):
    """设置默认日志配置"""
    config = LoggerConfig(
        min_level=log_level,
        file_path=os.path.join(log_dir, "app.log")
    )
    configure_logging(config)


# 性能监控装饰器
def log_performance(logger_name: str = None, operation_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        actual_logger_name = logger_name or func.__module__
        actual_operation_name = operation_name or func.__name__
        
        def wrapper(*args, **kwargs):
            logger = get_structured_logger(actual_logger_name)
            with logger.performance_context(actual_operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
