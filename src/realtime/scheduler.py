"""
任务调度器 - 自动执行定时任务
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
import logging
from queue import Queue, Empty
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """任务数据类"""
    id: str
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    schedule_time: datetime = None
    interval_seconds: Optional[int] = None
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskScheduler:
    """
    任务调度器
    支持定时任务、周期性任务和即时任务
    """
    
    def __init__(self, max_workers: int = 5):
        """
        初始化调度器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.tasks = {}  # 所有任务
        self.task_queue = Queue()  # 待执行任务队列
        self.workers = []  # 工作线程列表
        self.running = False  # 调度器运行状态
        self.lock = threading.RLock()  # 线程锁
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0
        }
        
        # 周期性任务跟踪
        self.recurring_tasks = {}
        
        logger.info(f"任务调度器初始化完成，最大工作线程数: {max_workers}")
    
    def start(self):
        """启动调度器"""
        if self.running:
            logger.warning("调度器已经在运行")
            return
        
        self.running = True
        
        # 启动工作线程
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # 启动调度线程
        scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="TaskScheduler",
            daemon=True
        )
        scheduler_thread.start()
        
        logger.info(f"调度器已启动，{len(self.workers)}个工作线程")
    
    def stop(self):
        """停止调度器"""
        if not self.running:
            return
        
        logger.info("正在停止调度器...")
        self.running = False
        
        # 等待所有任务完成（最多等待30秒）
        for _ in range(30):
            if self.task_queue.empty():
                break
            time.sleep(1)
        
        logger.info("调度器已停止")
    
    def add_task(self, task_id: str, name: str, func: Callable,
                args: tuple = (), kwargs: dict = None,
                schedule_time: Optional[datetime] = None,
                max_retries: int = 3) -> bool:
        """
        添加一次性任务
        
        Args:
            task_id: 任务ID
            name: 任务名称
            func: 执行函数
            args: 函数参数
            kwargs: 函数关键字参数
            schedule_time: 计划执行时间（None表示立即执行）
            max_retries: 最大重试次数
            
        Returns:
            是否添加成功
        """
        try:
            with self.lock:
                if task_id in self.tasks:
                    logger.warning(f"任务ID {task_id} 已存在")
                    return False
                
                task = Task(
                    id=task_id,
                    name=name,
                    func=func,
                    args=args,
                    kwargs=kwargs or {},
                    schedule_time=schedule_time,
                    max_retries=max_retries
                )
                
                self.tasks[task_id] = task
                self.stats['total_tasks'] += 1
                
                # 如果是立即执行的任务，加入队列
                if schedule_time is None or schedule_time <= datetime.now():
                    self.task_queue.put(task)
                
                logger.info(f"任务已添加: {task_id} - {name}")
                return True
                
        except Exception as e:
            logger.error(f"添加任务失败: {e}")
            return False
    
    def add_recurring_task(self, task_id: str, name: str, func: Callable,
                          interval_seconds: int, args: tuple = (),
                          kwargs: dict = None, max_retries: int = 3) -> bool:
        """
        添加周期性任务
        
        Args:
            task_id: 任务ID
            name: 任务名称
            func: 执行函数
            interval_seconds: 执行间隔（秒）
            args: 函数参数
            kwargs: 函数关键字参数
            max_retries: 最大重试次数
            
        Returns:
            是否添加成功
        """
        try:
            with self.lock:
                if task_id in self.recurring_tasks:
                    logger.warning(f"周期性任务ID {task_id} 已存在")
                    return False
                
                task_config = {
                    'name': name,
                    'func': func,
                    'interval_seconds': interval_seconds,
                    'args': args,
                    'kwargs': kwargs or {},
                    'max_retries': max_retries,
                    'next_run': datetime.now(),
                    'last_run': None,
                    'run_count': 0,
                    'active': True
                }
                
                self.recurring_tasks[task_id] = task_config
                
                logger.info(f"周期性任务已添加: {task_id} - {name} (间隔: {interval_seconds}秒)")
                return True
                
        except Exception as e:
            logger.error(f"添加周期性任务失败: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否取消成功
        """
        try:
            with self.lock:
                # 取消一次性任务
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                        task.status = TaskStatus.CANCELLED
                        self.stats['cancelled_tasks'] += 1
                        logger.info(f"任务已取消: {task_id}")
                        return True
                
                # 取消周期性任务
                if task_id in self.recurring_tasks:
                    self.recurring_tasks[task_id]['active'] = False
                    logger.info(f"周期性任务已取消: {task_id}")
                    return True
                
                logger.warning(f"任务不存在: {task_id}")
                return False
                
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        try:
            with self.lock:
                # 查找一次性任务
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    return {
                        'id': task.id,
                        'name': task.name,
                        'status': task.status.value,
                        'created_at': task.created_at.isoformat(),
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'retry_count': task.retry_count,
                        'max_retries': task.max_retries,
                        'error': task.error,
                        'type': 'one_time'
                    }
                
                # 查找周期性任务
                if task_id in self.recurring_tasks:
                    config = self.recurring_tasks[task_id]
                    return {
                        'id': task_id,
                        'name': config['name'],
                        'status': 'active' if config['active'] else 'inactive',
                        'interval_seconds': config['interval_seconds'],
                        'next_run': config['next_run'].isoformat(),
                        'last_run': config['last_run'].isoformat() if config['last_run'] else None,
                        'run_count': config['run_count'],
                        'type': 'recurring'
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return None
    
    def list_tasks(self, status_filter: Optional[str] = None) -> List[Dict]:
        """
        列出所有任务
        
        Args:
            status_filter: 状态过滤器
            
        Returns:
            任务列表
        """
        try:
            with self.lock:
                task_list = []
                
                # 一次性任务
                for task in self.tasks.values():
                    if status_filter is None or task.status.value == status_filter:
                        task_info = {
                            'id': task.id,
                            'name': task.name,
                            'status': task.status.value,
                            'created_at': task.created_at.isoformat(),
                            'type': 'one_time'
                        }
                        task_list.append(task_info)
                
                # 周期性任务
                for task_id, config in self.recurring_tasks.items():
                    task_status = 'active' if config['active'] else 'inactive'
                    if status_filter is None or task_status == status_filter:
                        task_info = {
                            'id': task_id,
                            'name': config['name'],
                            'status': task_status,
                            'interval_seconds': config['interval_seconds'],
                            'run_count': config['run_count'],
                            'type': 'recurring'
                        }
                        task_list.append(task_info)
                
                return task_list
                
        except Exception as e:
            logger.error(f"列出任务失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                'total_tasks': self.stats['total_tasks'],
                'completed_tasks': self.stats['completed_tasks'],
                'failed_tasks': self.stats['failed_tasks'],
                'cancelled_tasks': self.stats['cancelled_tasks'],
                'pending_tasks': sum(1 for task in self.tasks.values() 
                                   if task.status == TaskStatus.PENDING),
                'running_tasks': sum(1 for task in self.tasks.values() 
                                   if task.status == TaskStatus.RUNNING),
                'recurring_tasks': len(self.recurring_tasks),
                'active_recurring_tasks': sum(1 for config in self.recurring_tasks.values() 
                                            if config['active']),
                'queue_size': self.task_queue.qsize(),
                'workers': len(self.workers),
                'scheduler_running': self.running
            }
    
    def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("调度器主循环已启动")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                with self.lock:
                    # 检查定时任务
                    for task in list(self.tasks.values()):
                        if (task.status == TaskStatus.PENDING and 
                            task.schedule_time and 
                            task.schedule_time <= current_time):
                            self.task_queue.put(task)
                    
                    # 检查周期性任务
                    for task_id, config in self.recurring_tasks.items():
                        if (config['active'] and 
                            config['next_run'] <= current_time):
                            
                            # 创建新的任务实例
                            instance_id = f"{task_id}_{config['run_count']}"
                            task = Task(
                                id=instance_id,
                                name=f"{config['name']} (#{config['run_count']})",
                                func=config['func'],
                                args=config['args'],
                                kwargs=config['kwargs'],
                                max_retries=config['max_retries']
                            )
                            
                            self.tasks[instance_id] = task
                            self.task_queue.put(task)
                            
                            # 更新周期性任务配置
                            config['last_run'] = current_time
                            config['next_run'] = current_time + timedelta(seconds=config['interval_seconds'])
                            config['run_count'] += 1
                            
                            logger.debug(f"周期性任务已调度: {task_id}")
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                time.sleep(5)  # 异常时等待更久
        
        logger.info("调度器主循环已结束")
    
    def _worker_loop(self):
        """工作线程循环"""
        worker_name = threading.current_thread().name
        logger.debug(f"工作线程 {worker_name} 已启动")
        
        while self.running:
            try:
                # 从队列获取任务（超时1秒）
                task = self.task_queue.get(timeout=1)
                
                if task.status == TaskStatus.CANCELLED:
                    self.task_queue.task_done()
                    continue
                
                # 执行任务
                self._execute_task(task)
                self.task_queue.task_done()
                
            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"工作线程 {worker_name} 异常: {e}")
        
        logger.debug(f"工作线程 {worker_name} 已结束")
    
    def _execute_task(self, task: Task):
        """执行单个任务"""
        try:
            logger.info(f"开始执行任务: {task.id} - {task.name}")
            
            with self.lock:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
            
            # 执行任务函数
            result = task.func(*task.args, **task.kwargs)
            
            with self.lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                self.stats['completed_tasks'] += 1
            
            logger.info(f"任务执行完成: {task.id}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"任务执行失败: {task.id} - {error_msg}")
            
            with self.lock:
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    # 重试任务
                    task.status = TaskStatus.PENDING
                    task.error = f"重试 {task.retry_count}: {error_msg}"
                    
                    # 延迟重试（指数退避）
                    delay = min(60, 2 ** task.retry_count)
                    retry_time = datetime.now() + timedelta(seconds=delay)
                    task.schedule_time = retry_time
                    
                    logger.info(f"任务将在 {delay} 秒后重试: {task.id}")
                else:
                    # 达到最大重试次数，标记为失败
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    task.error = error_msg
                    self.stats['failed_tasks'] += 1
                    
                    logger.error(f"任务最终失败: {task.id}")
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """清理已完成的旧任务"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            with self.lock:
                tasks_to_remove = []
                
                for task_id, task in self.tasks.items():
                    if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                        task.completed_at and task.completed_at < cutoff_time):
                        tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"已清理 {cleaned_count} 个旧任务")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理任务失败: {e}")
            return 0
    
    def export_task_history(self, filepath: str) -> bool:
        """导出任务历史"""
        try:
            with self.lock:
                history_data = {
                    'export_time': datetime.now().isoformat(),
                    'statistics': self.get_statistics(),
                    'tasks': [],
                    'recurring_tasks': {}
                }
                
                # 导出一次性任务
                for task in self.tasks.values():
                    task_data = {
                        'id': task.id,
                        'name': task.name,
                        'status': task.status.value,
                        'created_at': task.created_at.isoformat(),
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'retry_count': task.retry_count,
                        'max_retries': task.max_retries,
                        'error': task.error
                    }
                    history_data['tasks'].append(task_data)
                
                # 导出周期性任务
                for task_id, config in self.recurring_tasks.items():
                    recurring_data = {
                        'name': config['name'],
                        'interval_seconds': config['interval_seconds'],
                        'run_count': config['run_count'],
                        'active': config['active'],
                        'last_run': config['last_run'].isoformat() if config['last_run'] else None,
                        'next_run': config['next_run'].isoformat()
                    }
                    history_data['recurring_tasks'][task_id] = recurring_data
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"任务历史已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出任务历史失败: {e}")
            return False


# 全局调度器实例
_global_scheduler = None
_scheduler_lock = threading.Lock()


def get_global_scheduler() -> TaskScheduler:
    """获取全局调度器实例"""
    global _global_scheduler
    
    with _scheduler_lock:
        if _global_scheduler is None:
            _global_scheduler = TaskScheduler()
            _global_scheduler.start()
        
        return _global_scheduler


def shutdown_global_scheduler():
    """关闭全局调度器"""
    global _global_scheduler
    
    with _scheduler_lock:
        if _global_scheduler is not None:
            _global_scheduler.stop()
            _global_scheduler = None
