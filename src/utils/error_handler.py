"""
增强的错误处理和用户提示系统
提供智能错误处理、用户友好的提示信息、错误恢复机制
"""

import sys
import traceback
import logging
import json
import time
from typing import Dict, List, Optional, Callable, Any, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorLevel(Enum):
    """错误级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"          # 系统错误
    NETWORK = "network"        # 网络错误
    DATABASE = "database"      # 数据库错误
    COMPUTATION = "computation" # 计算错误
    UI = "ui"                 # 界面错误
    DATA = "data"             # 数据错误
    PERMISSION = "permission"  # 权限错误
    VALIDATION = "validation"  # 验证错误


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    level: ErrorLevel
    category: ErrorCategory
    title: str
    message: str
    details: Optional[str] = None
    suggestions: Optional[List[str]] = None
    timestamp: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    traceback_info: Optional[str] = None
    user_action: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['level'] = self.level.value
        result['category'] = self.category.value
        return result


class ErrorRecoveryStrategy:
    """错误恢复策略"""
    
    def __init__(self, strategy_name: str, recovery_func: Callable, 
                 max_attempts: int = 3, delay: float = 1.0):
        self.strategy_name = strategy_name
        self.recovery_func = recovery_func
        self.max_attempts = max_attempts
        self.delay = delay
        self.attempt_count = 0
    
    def can_recover(self) -> bool:
        """检查是否可以恢复"""
        return self.attempt_count < self.max_attempts
    
    def recover(self, error_info: ErrorInfo) -> bool:
        """执行恢复"""
        if not self.can_recover():
            return False
        
        self.attempt_count += 1
        try:
            time.sleep(self.delay * self.attempt_count)  # 指数退避
            return self.recovery_func(error_info)
        except Exception as e:
            logger.error(f"恢复策略 {self.strategy_name} 失败: {e}")
            return False
    
    def reset(self):
        """重置尝试次数"""
        self.attempt_count = 0


class UserNotification:
    """用户通知"""
    
    def __init__(self):
        self.notification_handlers = []
    
    def add_handler(self, handler: Callable[[ErrorInfo], None]):
        """添加通知处理器"""
        self.notification_handlers.append(handler)
    
    def notify(self, error_info: ErrorInfo):
        """发送通知"""
        for handler in self.notification_handlers:
            try:
                handler(error_info)
            except Exception as e:
                logger.error(f"通知处理器执行失败: {e}")


class ErrorStatistics:
    """错误统计"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_history = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.category_counts = defaultdict(int)
        self.level_counts = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_error(self, error_info: ErrorInfo):
        """记录错误"""
        with self.lock:
            self.error_history.append(error_info)
            self.error_counts[error_info.error_id] += 1
            self.category_counts[error_info.category.value] += 1
            self.level_counts[error_info.level.value] += 1
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        with self.lock:
            # 筛选时间范围内的错误
            recent_errors = [
                error for error in self.error_history 
                if error.timestamp >= cutoff_time
            ]
            
            # 统计
            recent_counts = defaultdict(int)
            recent_categories = defaultdict(int)
            recent_levels = defaultdict(int)
            
            for error in recent_errors:
                recent_counts[error.error_id] += 1
                recent_categories[error.category.value] += 1
                recent_levels[error.level.value] += 1
            
            return {
                'total_errors': len(recent_errors),
                'error_rate': len(recent_errors) / max(hours, 1),
                'top_errors': dict(sorted(recent_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]),
                'category_distribution': dict(recent_categories),
                'level_distribution': dict(recent_levels),
                'time_range_hours': hours
            }
    
    def get_error_trends(self) -> List[Dict[str, Any]]:
        """获取错误趋势"""
        with self.lock:
            # 按小时分组统计
            hourly_stats = defaultdict(int)
            current_time = time.time()
            
            for error in self.error_history:
                hour = int((current_time - error.timestamp) // 3600)
                if hour <= 24:  # 只统计最近24小时
                    hourly_stats[24 - hour] += 1
            
            # 转换为列表
            trends = []
            for hour in range(25):  # 0-24小时
                trends.append({
                    'hour': hour,
                    'count': hourly_stats.get(hour, 0)
                })
            
            return trends


class EnhancedErrorHandler:
    """增强的错误处理器"""
    
    def __init__(self):
        self.error_templates = {}
        self.recovery_strategies = {}
        self.user_notification = UserNotification()
        self.error_statistics = ErrorStatistics()
        self.context_providers = []
        
        # 初始化默认错误模板
        self._initialize_error_templates()
        
        # 初始化默认恢复策略
        self._initialize_recovery_strategies()
        
        logger.info("增强错误处理器初始化完成")
    
    def _initialize_error_templates(self):
        """初始化错误模板"""
        templates = {
            # 网络错误
            "network_connection_failed": ErrorInfo(
                error_id="network_connection_failed",
                level=ErrorLevel.ERROR,
                category=ErrorCategory.NETWORK,
                title="网络连接失败",
                message="无法连接到服务器，请检查网络连接",
                suggestions=[
                    "检查网络连接是否正常",
                    "确认服务器地址是否正确",
                    "尝试重新连接",
                    "检查防火墙设置"
                ],
                user_action="点击重试按钮或检查网络设置"
            ),
            
            "network_timeout": ErrorInfo(
                error_id="network_timeout",
                level=ErrorLevel.WARNING,
                category=ErrorCategory.NETWORK,
                title="网络请求超时",
                message="请求超时，服务器响应缓慢",
                suggestions=[
                    "稍后重试",
                    "检查网络速度",
                    "尝试更换网络环境"
                ],
                user_action="点击重试或稍后再试"
            ),
            
            # 数据库错误
            "database_connection_failed": ErrorInfo(
                error_id="database_connection_failed",
                level=ErrorLevel.CRITICAL,
                category=ErrorCategory.DATABASE,
                title="数据库连接失败",
                message="无法连接到数据库",
                suggestions=[
                    "检查数据库文件是否存在",
                    "确认数据库文件权限",
                    "重启应用程序",
                    "检查磁盘空间"
                ],
                user_action="重启应用或联系技术支持"
            ),
            
            "database_query_failed": ErrorInfo(
                error_id="database_query_failed",
                level=ErrorLevel.ERROR,
                category=ErrorCategory.DATABASE,
                title="数据查询失败",
                message="数据库查询执行失败",
                suggestions=[
                    "检查数据完整性",
                    "重试操作",
                    "重新加载数据"
                ],
                user_action="点击重试或刷新数据"
            ),
            
            # 计算错误
            "computation_memory_error": ErrorInfo(
                error_id="computation_memory_error",
                level=ErrorLevel.ERROR,
                category=ErrorCategory.COMPUTATION,
                title="内存不足",
                message="计算过程中内存不足",
                suggestions=[
                    "关闭其他应用程序释放内存",
                    "减少数据处理量",
                    "重启应用程序",
                    "升级系统内存"
                ],
                user_action="关闭其他程序或减少处理数据量"
            ),
            
            "computation_invalid_input": ErrorInfo(
                error_id="computation_invalid_input",
                level=ErrorLevel.WARNING,
                category=ErrorCategory.VALIDATION,
                title="输入数据无效",
                message="输入的数据格式不正确或不完整",
                suggestions=[
                    "检查输入数据格式",
                    "确保数据完整性",
                    "参考示例数据格式"
                ],
                user_action="检查并修正输入数据"
            ),
            
            # UI错误
            "ui_component_load_failed": ErrorInfo(
                error_id="ui_component_load_failed",
                level=ErrorLevel.ERROR,
                category=ErrorCategory.UI,
                title="界面组件加载失败",
                message="界面组件无法正常加载",
                suggestions=[
                    "刷新界面",
                    "重启应用程序",
                    "检查系统资源"
                ],
                user_action="刷新界面或重启应用"
            ),
            
            # 权限错误
            "permission_denied": ErrorInfo(
                error_id="permission_denied",
                level=ErrorLevel.ERROR,
                category=ErrorCategory.PERMISSION,
                title="权限不足",
                message="没有足够的权限执行此操作",
                suggestions=[
                    "以管理员身份运行程序",
                    "检查文件权限",
                    "联系系统管理员"
                ],
                user_action="以管理员身份运行或联系管理员"
            )
        }
        
        for error_id, template in templates.items():
            self.error_templates[error_id] = template
    
    def _initialize_recovery_strategies(self):
        """初始化恢复策略"""
        # 网络重连策略
        def network_retry_strategy(error_info: ErrorInfo) -> bool:
            logger.info(f"尝试网络重连: {error_info.error_id}")
            time.sleep(1)
            return True  # 简化实现，实际应该测试网络连接
        
        self.recovery_strategies["network_retry"] = ErrorRecoveryStrategy(
            "网络重连", network_retry_strategy, max_attempts=3, delay=2.0
        )
        
        # 数据库重连策略
        def database_retry_strategy(error_info: ErrorInfo) -> bool:
            logger.info(f"尝试数据库重连: {error_info.error_id}")
            # 实际实现应该重新初始化数据库连接
            return True
        
        self.recovery_strategies["database_retry"] = ErrorRecoveryStrategy(
            "数据库重连", database_retry_strategy, max_attempts=2, delay=3.0
        )
        
        # 内存清理策略
        def memory_cleanup_strategy(error_info: ErrorInfo) -> bool:
            logger.info("执行内存清理")
            import gc
            gc.collect()
            return True
        
        self.recovery_strategies["memory_cleanup"] = ErrorRecoveryStrategy(
            "内存清理", memory_cleanup_strategy, max_attempts=1, delay=0.5
        )
    
    def add_context_provider(self, provider: Callable[[], Dict[str, Any]]):
        """添加上下文提供者"""
        self.context_providers.append(provider)
    
    def _gather_context(self) -> Dict[str, Any]:
        """收集上下文信息"""
        context = {}
        
        for provider in self.context_providers:
            try:
                provider_context = provider()
                if isinstance(provider_context, dict):
                    context.update(provider_context)
            except Exception as e:
                logger.warning(f"上下文提供者执行失败: {e}")
        
        return context
    
    def create_error_info(self, error_id: str, exception: Optional[Exception] = None,
                         **kwargs) -> ErrorInfo:
        """创建错误信息"""
        # 获取模板
        template = self.error_templates.get(error_id)
        if not template:
            template = ErrorInfo(
                error_id=error_id,
                level=ErrorLevel.ERROR,
                category=ErrorCategory.SYSTEM,
                title="未知错误",
                message="发生了未知错误"
            )
        
        # 复制模板并更新
        error_info = ErrorInfo(
            error_id=template.error_id,
            level=template.level,
            category=template.category,
            title=template.title,
            message=template.message,
            details=template.details,
            suggestions=template.suggestions,
            user_action=template.user_action
        )
        
        # 更新自定义字段
        for key, value in kwargs.items():
            if hasattr(error_info, key):
                setattr(error_info, key, value)
        
        # 添加异常信息
        if exception:
            error_info.details = str(exception)
            error_info.traceback_info = traceback.format_exc()
        
        # 添加上下文
        error_info.context = self._gather_context()
        
        return error_info
    
    def handle_error(self, error_info: ErrorInfo, 
                    auto_recover: bool = True) -> Optional[ErrorInfo]:
        """处理错误"""
        # 记录错误
        self.error_statistics.record_error(error_info)
        
        # 记录日志
        log_level = {
            ErrorLevel.INFO: logging.INFO,
            ErrorLevel.WARNING: logging.WARNING,
            ErrorLevel.ERROR: logging.ERROR,
            ErrorLevel.CRITICAL: logging.CRITICAL
        }.get(error_info.level, logging.ERROR)
        
        logger.log(log_level, f"错误处理: {error_info.title} - {error_info.message}")
        
        # 尝试自动恢复
        if auto_recover:
            recovery_attempted = self._attempt_recovery(error_info)
            if recovery_attempted:
                logger.info(f"错误 {error_info.error_id} 已尝试自动恢复")
                return None  # 恢复成功，不需要用户干预
        
        # 发送用户通知
        self.user_notification.notify(error_info)
        
        return error_info
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """尝试错误恢复"""
        # 根据错误类别选择恢复策略
        strategy_map = {
            ErrorCategory.NETWORK: ["network_retry"],
            ErrorCategory.DATABASE: ["database_retry"],
            ErrorCategory.COMPUTATION: ["memory_cleanup"]
        }
        
        strategies = strategy_map.get(error_info.category, [])
        
        for strategy_name in strategies:
            strategy = self.recovery_strategies.get(strategy_name)
            if strategy and strategy.can_recover():
                logger.info(f"尝试恢复策略: {strategy_name}")
                if strategy.recover(error_info):
                    logger.info(f"恢复策略 {strategy_name} 成功")
                    return True
                else:
                    logger.warning(f"恢复策略 {strategy_name} 失败")
        
        return False
    
    def register_error_template(self, error_id: str, error_info: ErrorInfo):
        """注册错误模板"""
        self.error_templates[error_id] = error_info
    
    def register_recovery_strategy(self, name: str, strategy: ErrorRecoveryStrategy):
        """注册恢复策略"""
        self.recovery_strategies[name] = strategy
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误统计"""
        return self.error_statistics.get_statistics(hours)
    
    def get_error_trends(self) -> List[Dict[str, Any]]:
        """获取错误趋势"""
        return self.error_statistics.get_error_trends()


def error_handler(error_id: str = None, auto_recover: bool = True, 
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 level: ErrorLevel = ErrorLevel.ERROR):
    """错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                
                # 生成错误ID
                actual_error_id = error_id or f"{func.__name__}_error"
                
                # 创建错误信息
                error_info = handler.create_error_info(
                    actual_error_id,
                    exception=e,
                    level=level,
                    category=category,
                    title=f"函数 {func.__name__} 执行失败",
                    message=f"执行 {func.__name__} 时发生错误: {str(e)}"
                )
                
                # 处理错误
                result = handler.handle_error(error_info, auto_recover)
                
                if result:  # 如果没有恢复成功
                    raise e  # 重新抛出原始异常
                
                # 如果恢复成功，返回默认值或重试
                return None
        
        return wrapper
    return decorator


@contextmanager
def error_context(error_id: str, auto_recover: bool = True):
    """错误上下文管理器"""
    try:
        yield
    except Exception as e:
        handler = get_error_handler()
        
        error_info = handler.create_error_info(
            error_id,
            exception=e,
            title="操作执行失败",
            message=f"在执行 {error_id} 时发生错误: {str(e)}"
        )
        
        result = handler.handle_error(error_info, auto_recover)
        
        if result:  # 如果没有恢复成功
            raise e


def safe_execute(func: Callable, error_id: str = None, 
                default_return: Any = None, **kwargs) -> Any:
    """安全执行函数"""
    try:
        return func(**kwargs)
    except Exception as e:
        handler = get_error_handler()
        
        actual_error_id = error_id or f"{func.__name__}_safe_execute"
        
        error_info = handler.create_error_info(
            actual_error_id,
            exception=e,
            title="安全执行失败",
            message=f"安全执行 {func.__name__} 失败: {str(e)}"
        )
        
        handler.handle_error(error_info, auto_recover=True)
        
        return default_return


# 全局错误处理器实例
_error_handler = None

def get_error_handler() -> EnhancedErrorHandler:
    """获取错误处理器实例（单例模式）"""
    global _error_handler
    if _error_handler is None:
        _error_handler = EnhancedErrorHandler()
    return _error_handler


# 便捷函数
def handle_network_error(exception: Exception) -> ErrorInfo:
    """处理网络错误"""
    handler = get_error_handler()
    error_info = handler.create_error_info("network_connection_failed", exception)
    return handler.handle_error(error_info)

def handle_database_error(exception: Exception) -> ErrorInfo:
    """处理数据库错误"""
    handler = get_error_handler()
    error_info = handler.create_error_info("database_connection_failed", exception)
    return handler.handle_error(error_info)

def handle_computation_error(exception: Exception) -> ErrorInfo:
    """处理计算错误"""
    handler = get_error_handler()
    error_info = handler.create_error_info("computation_memory_error", exception)
    return handler.handle_error(error_info)

def handle_ui_error(exception: Exception) -> ErrorInfo:
    """处理UI错误"""
    handler = get_error_handler()
    error_info = handler.create_error_info("ui_component_load_failed", exception)
    return handler.handle_error(error_info)
