"""
实时处理器 - 整合数据获取、分析和通知功能
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import threading
import json
import os

try:
    from realtime.data_fetcher import DataFetcher
    from realtime.scheduler import TaskScheduler, get_global_scheduler
    from core.database_manager import DatabaseManager
    from ml.model_manager import ModelManager
except ImportError:
    # 回退到相对导入
    from .data_fetcher import DataFetcher
    from .scheduler import TaskScheduler, get_global_scheduler
    from ..core.database_manager import DatabaseManager
    from ..ml.model_manager import ModelManager
try:
    from ml.recommendation_engine import RecommendationEngine
except ImportError:
    from ..ml.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)


class RealtimeProcessor:
    """
    实时处理器
    负责协调数据获取、分析处理和结果通知
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 model_manager: Optional[ModelManager] = None,
                 scheduler: Optional[TaskScheduler] = None):
        """
        初始化实时处理器
        
        Args:
            db_manager: 数据库管理器
            model_manager: 模型管理器
            scheduler: 任务调度器
        """
        self.db_manager = db_manager or DatabaseManager()
        self.model_manager = model_manager or ModelManager()
        self.scheduler = scheduler or get_global_scheduler()
        self.data_fetcher = DataFetcher()
        
        # 推荐引擎
        self.recommendation_engines = {
            '双色球': RecommendationEngine('双色球', self.model_manager),
            '大乐透': RecommendationEngine('大乐透', self.model_manager)
        }
        
        # 配置
        self.config = {
            'auto_fetch_enabled': True,
            'auto_analysis_enabled': True,
            'auto_prediction_enabled': True,
            'fetch_interval_hours': 2,
            'analysis_interval_hours': 6,
            'prediction_interval_hours': 12,
            'notification_enabled': True,
            'max_data_age_hours': 48
        }
        
        # 回调函数
        self.callbacks = {
            'new_data': [],
            'analysis_complete': [],
            'prediction_complete': [],
            'error_occurred': []
        }
        
        # 状态跟踪
        self.status = {
            'last_fetch': {},
            'last_analysis': {},
            'last_prediction': {},
            'running': False
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        logger.info("实时处理器初始化完成")
    
    def start(self):
        """启动实时处理器"""
        try:
            if self.status['running']:
                logger.warning("实时处理器已经在运行")
                return
            
            with self.lock:
                self.status['running'] = True
            
            # 注册定时任务
            self._register_scheduled_tasks()
            
            # 执行初始数据检查
            self._initial_data_check()
            
            logger.info("实时处理器已启动")
            
        except Exception as e:
            logger.error(f"启动实时处理器失败: {e}")
            self.status['running'] = False
    
    def stop(self):
        """停止实时处理器"""
        try:
            with self.lock:
                self.status['running'] = False
            
            # 取消所有相关任务
            self._cancel_scheduled_tasks()
            
            logger.info("实时处理器已停止")
            
        except Exception as e:
            logger.error(f"停止实时处理器失败: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        注册回调函数
        
        Args:
            event_type: 事件类型 ('new_data', 'analysis_complete', 'prediction_complete', 'error_occurred')
            callback: 回调函数
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"已注册 {event_type} 事件回调")
        else:
            logger.warning(f"未知事件类型: {event_type}")
    
    def update_config(self, new_config: Dict):
        """更新配置"""
        try:
            with self.lock:
                self.config.update(new_config)
            
            # 重新注册任务（如果间隔时间改变）
            if self.status['running']:
                self._cancel_scheduled_tasks()
                self._register_scheduled_tasks()
            
            logger.info("配置已更新")
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
    
    def get_status(self) -> Dict:
        """获取处理器状态"""
        with self.lock:
            return {
                'running': self.status['running'],
                'last_fetch': self.status['last_fetch'].copy(),
                'last_analysis': self.status['last_analysis'].copy(),
                'last_prediction': self.status['last_prediction'].copy(),
                'config': self.config.copy(),
                'data_fetcher_cache': self.data_fetcher.get_cache_status()
            }
    
    def force_fetch_data(self, lottery_type: str, periods: int = 10) -> bool:
        """
        强制获取数据
        
        Args:
            lottery_type: 彩票类型
            periods: 获取期数
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"强制获取{lottery_type}数据...")
            
            new_data = self.data_fetcher.get_latest_data(lottery_type, periods)
            
            if new_data:
                success = self._process_new_data(lottery_type, new_data)
                if success:
                    with self.lock:
                        self.status['last_fetch'][lottery_type] = datetime.now().isoformat()
                    
                    self._trigger_callbacks('new_data', {
                        'lottery_type': lottery_type,
                        'data_count': len(new_data),
                        'latest_period': new_data[0].get('period', '') if new_data else ''
                    })
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"强制获取数据失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'force_fetch_data'
            })
            return False
    
    def force_fetch_all_data(self, lottery_type: str, max_periods: int = 2000) -> bool:
        """
        强制获取全部历史数据
        
        Args:
            lottery_type: 彩票类型
            max_periods: 最大获取期数
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"开始强制获取{lottery_type}全部历史数据...")
            
            all_data = self.data_fetcher.get_all_historical_data(lottery_type, max_periods)
            
            if all_data:
                success = self._process_new_data(lottery_type, all_data)
                if success:
                    with self.lock:
                        self.status['last_fetch'][lottery_type] = datetime.now().isoformat()
                    
                    self._trigger_callbacks('new_data', {
                        'lottery_type': lottery_type,
                        'data_count': len(all_data),
                        'latest_period': all_data[0].get('period', '') if all_data else '',
                        'fetch_type': 'all_historical'
                    })
                    
                    logger.info(f"成功获取并处理{len(all_data)}期{lottery_type}历史数据")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"强制获取全部数据失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'force_fetch_all_data'
            })
            return False
    
    def auto_sync_all_data(self, lottery_types: List[str] = None) -> Dict[str, bool]:
        """
        自动同步所有彩票类型的全部数据
        
        Args:
            lottery_types: 彩票类型列表，默认为所有支持的类型
            
        Returns:
            各彩票类型的同步结果
        """
        if lottery_types is None:
            lottery_types = ['双色球', '大乐透']
        
        results = {}
        
        for lottery_type in lottery_types:
            try:
                logger.info(f"开始同步{lottery_type}全部数据...")
                
                # 使用自动更新功能
                success = self.data_fetcher.auto_update_data(lottery_type, 1)  # 1小时检查间隔
                
                if not success:
                    # 如果自动更新失败，尝试强制获取最新数据
                    success = self.force_fetch_data(lottery_type, 50)
                
                results[lottery_type] = success
                
                if success:
                    logger.info(f"{lottery_type}数据同步成功")
                else:
                    logger.warning(f"{lottery_type}数据同步失败")
                    
            except Exception as e:
                logger.error(f"同步{lottery_type}数据时发生异常: {e}")
                results[lottery_type] = False
        
        return results
    
    def force_analysis(self, lottery_type: str) -> bool:
        """
        强制执行分析
        
        Args:
            lottery_type: 彩票类型
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"强制分析{lottery_type}数据...")
            
            result = self._perform_analysis(lottery_type)
            
            if result:
                with self.lock:
                    self.status['last_analysis'][lottery_type] = datetime.now().isoformat()
                
                self._trigger_callbacks('analysis_complete', {
                    'lottery_type': lottery_type,
                    'analysis_result': result
                })
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"强制分析失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'force_analysis'
            })
            return False
    
    def force_prediction(self, lottery_type: str) -> Optional[Dict]:
        """
        强制生成预测
        
        Args:
            lottery_type: 彩票类型
            
        Returns:
            预测结果
        """
        try:
            logger.info(f"强制生成{lottery_type}预测...")
            
            result = self._generate_predictions(lottery_type)
            
            if result and result.get('success'):
                with self.lock:
                    self.status['last_prediction'][lottery_type] = datetime.now().isoformat()
                
                self._trigger_callbacks('prediction_complete', {
                    'lottery_type': lottery_type,
                    'prediction_result': result
                })
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"强制预测失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'force_prediction'
            })
            return None
    
    def get_latest_results(self, lottery_type: str) -> Dict:
        """获取最新的分析和预测结果"""
        try:
            conn = self.db_manager.get_connection()
            
            # 获取最新的分析结果
            cursor = conn.execute('''
                SELECT * FROM analysis_results 
                WHERE lottery_type = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (lottery_type,))
            
            analysis_row = cursor.fetchone()
            latest_analysis = dict(analysis_row) if analysis_row else None
            
            # 获取最新的预测结果
            cursor = conn.execute('''
                SELECT * FROM prediction_records 
                WHERE lottery_type = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (lottery_type,))
            
            prediction_row = cursor.fetchone()
            latest_prediction = dict(prediction_row) if prediction_row else None
            
            conn.close()
            
            return {
                'analysis': latest_analysis,
                'prediction': latest_prediction,
                'last_fetch': self.status['last_fetch'].get(lottery_type),
                'last_analysis': self.status['last_analysis'].get(lottery_type),
                'last_prediction': self.status['last_prediction'].get(lottery_type)
            }
            
        except Exception as e:
            logger.error(f"获取最新结果失败: {e}")
            return {}
    
    def _register_scheduled_tasks(self):
        """注册定时任务"""
        try:
            # 数据获取任务
            if self.config['auto_fetch_enabled']:
                for lottery_type in ['双色球', '大乐透']:
                    self.scheduler.add_recurring_task(
                        task_id=f"fetch_data_{lottery_type}",
                        name=f"自动获取{lottery_type}数据",
                        func=self._scheduled_fetch_data,
                        interval_seconds=self.config['fetch_interval_hours'] * 3600,
                        args=(lottery_type,)
                    )
            
            # 数据分析任务
            if self.config['auto_analysis_enabled']:
                for lottery_type in ['双色球', '大乐透']:
                    self.scheduler.add_recurring_task(
                        task_id=f"analysis_{lottery_type}",
                        name=f"自动分析{lottery_type}数据",
                        func=self._scheduled_analysis,
                        interval_seconds=self.config['analysis_interval_hours'] * 3600,
                        args=(lottery_type,)
                    )
            
            # 预测生成任务
            if self.config['auto_prediction_enabled']:
                for lottery_type in ['双色球', '大乐透']:
                    self.scheduler.add_recurring_task(
                        task_id=f"prediction_{lottery_type}",
                        name=f"自动生成{lottery_type}预测",
                        func=self._scheduled_prediction,
                        interval_seconds=self.config['prediction_interval_hours'] * 3600,
                        args=(lottery_type,)
                    )
            
            # 清理任务
            self.scheduler.add_recurring_task(
                task_id="cleanup_old_data",
                name="清理旧数据",
                func=self._cleanup_old_data,
                interval_seconds=24 * 3600  # 每天执行一次
            )
            
            logger.info("定时任务已注册")
            
        except Exception as e:
            logger.error(f"注册定时任务失败: {e}")
    
    def _cancel_scheduled_tasks(self):
        """取消定时任务"""
        try:
            task_ids = [
                "fetch_data_双色球", "fetch_data_大乐透",
                "analysis_双色球", "analysis_大乐透",
                "prediction_双色球", "prediction_大乐透",
                "cleanup_old_data"
            ]
            
            for task_id in task_ids:
                self.scheduler.cancel_task(task_id)
            
            logger.info("定时任务已取消")
            
        except Exception as e:
            logger.error(f"取消定时任务失败: {e}")
    
    def _initial_data_check(self):
        """初始数据检查"""
        try:
            for lottery_type in ['双色球', '大乐透']:
                # 检查数据新鲜度
                if not self.data_fetcher.is_cache_valid(lottery_type, self.config['max_data_age_hours']):
                    logger.info(f"{lottery_type}数据过期，开始获取新数据...")
                    self.force_fetch_data(lottery_type)
            
        except Exception as e:
            logger.error(f"初始数据检查失败: {e}")
    
    def _scheduled_fetch_data(self, lottery_type: str):
        """定时数据获取任务"""
        try:
            logger.info(f"定时获取{lottery_type}数据...")
            
            # 获取最新数据
            new_data = self.data_fetcher.get_latest_data(lottery_type, 5)
            
            if new_data:
                self._process_new_data(lottery_type, new_data)
                
                with self.lock:
                    self.status['last_fetch'][lottery_type] = datetime.now().isoformat()
                
                self._trigger_callbacks('new_data', {
                    'lottery_type': lottery_type,
                    'data_count': len(new_data),
                    'latest_period': new_data[0].get('period', '') if new_data else ''
                })
            
        except Exception as e:
            logger.error(f"定时数据获取失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'scheduled_fetch_data'
            })
    
    def _scheduled_analysis(self, lottery_type: str):
        """定时数据分析任务"""
        try:
            logger.info(f"定时分析{lottery_type}数据...")
            
            result = self._perform_analysis(lottery_type)
            
            if result:
                with self.lock:
                    self.status['last_analysis'][lottery_type] = datetime.now().isoformat()
                
                self._trigger_callbacks('analysis_complete', {
                    'lottery_type': lottery_type,
                    'analysis_result': result
                })
            
        except Exception as e:
            logger.error(f"定时数据分析失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'scheduled_analysis'
            })
    
    def _scheduled_prediction(self, lottery_type: str):
        """定时预测生成任务"""
        try:
            logger.info(f"定时生成{lottery_type}预测...")
            
            result = self._generate_predictions(lottery_type)
            
            if result and result.get('success'):
                with self.lock:
                    self.status['last_prediction'][lottery_type] = datetime.now().isoformat()
                
                self._trigger_callbacks('prediction_complete', {
                    'lottery_type': lottery_type,
                    'prediction_result': result
                })
            
        except Exception as e:
            logger.error(f"定时预测生成失败: {e}")
            self._trigger_callbacks('error_occurred', {
                'error': str(e),
                'operation': 'scheduled_prediction'
            })
    
    def _process_new_data(self, lottery_type: str, new_data: List[Dict]) -> bool:
        """处理新获取的数据 (500彩票网格式)"""
        try:
            # 使用批量保存方法提高效率
            self.db_manager.save_lottery_history_batch(lottery_type, new_data)
            
            logger.info(f"已批量保存{len(new_data)}条{lottery_type}数据")
            return len(new_data) > 0
            
        except Exception as e:
            logger.error(f"处理新数据失败: {e}")
            # 尝试逐条保存（兼容性处理）
            try:
                saved_count = 0
                for data in new_data:
                    try:
                        period = data.get('period', '')
                        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
                        numbers = data.get('numbers', {})
                        
                        if period and numbers:
                            self.db_manager.save_lottery_history(lottery_type, period, date, numbers)
                            saved_count += 1
                    except Exception as e:
                        logger.warning(f"保存单条数据失败: {data} - {e}")
                        continue
                
                logger.info(f"逐条保存完成: {saved_count}/{len(new_data)}条{lottery_type}数据")
                return saved_count > 0
                
            except Exception as e2:
                logger.error(f"逐条保存也失败: {e2}")
                return False
    
    def _perform_analysis(self, lottery_type: str) -> Optional[Dict]:
        """执行数据分析"""
        try:
            # 获取历史数据 (使用500彩票网格式)
            history_data = self.db_manager.get_lottery_history(lottery_type, limit=100)
            
            if not history_data:
                logger.warning(f"{lottery_type}没有历史数据可供分析")
                return None
            
            # 使用高级分析
            try:
                from analysis.advanced_analysis import AdvancedAnalysis
            except ImportError:
                # 回退到相对导入
                from ..analysis.advanced_analysis import AdvancedAnalysis
            
            analyzer = AdvancedAnalysis(lottery_type)
            
            # 执行多种分析
            analysis_results = {
                'correlation': analyzer.correlation_analysis(history_data),
                'seasonality': analyzer.seasonality_detection(history_data),
                'anomaly': analyzer.anomaly_detection(history_data),
                'pattern': analyzer.pattern_recognition(history_data),
                'clustering': analyzer.clustering_analysis(history_data)
            }
            
            # 保存分析结果
            for analysis_type, result in analysis_results.items():
                if not result.get('error'):
                    self.db_manager.save_analysis_result(
                        lottery_type, analysis_type, "最近100期", result
                    )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"执行分析失败: {e}")
            return None
    
    def _generate_predictions(self, lottery_type: str) -> Optional[Dict]:
        """生成预测"""
        try:
            # 获取历史数据
            history_data = self.db_manager.get_historical_data(lottery_type, limit=100)
            
            if not history_data:
                logger.warning(f"{lottery_type}没有历史数据可供预测")
                return None
            
            # 使用推荐引擎生成预测
            recommendation_engine = self.recommendation_engines[lottery_type]
            
            result = recommendation_engine.generate_recommendations(
                history_data, num_recommendations=5
            )
            
            if result.get('success'):
                # 保存预测记录
                for i, recommendation in enumerate(result.get('recommendations', [])):
                    prediction_data = {
                        'red_balls': recommendation['red_balls'],
                        'blue_balls': recommendation['blue_balls'],
                        'confidence': recommendation.get('confidence', 0.5),
                        'strategy': recommendation.get('strategy', 'unknown'),
                        'type': recommendation.get('type', 'recommendation')
                    }
                    
                    self.db_manager.save_prediction_record(
                        lottery_type, 
                        f"auto_prediction_{i+1}",
                        prediction_data,
                        f"RecommendationEngine_{lottery_type}"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"生成预测失败: {e}")
            return None
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            logger.info("开始清理旧数据...")
            
            # 清理过期的分析结果
            cutoff_time = datetime.now() - timedelta(days=30)
            
            conn = self.db_manager.get_connection()
            
            # 删除过期的分析结果
            cursor = conn.execute('''
                DELETE FROM analysis_results 
                WHERE created_at < ? OR (expires_at IS NOT NULL AND expires_at < ?)
            ''', (cutoff_time.isoformat(), datetime.now().isoformat()))
            
            deleted_analysis = cursor.rowcount
            
            # 清理调度器的已完成任务
            self.scheduler.cleanup_completed_tasks(max_age_hours=48)
            
            conn.commit()
            conn.close()
            
            logger.info(f"清理完成：删除了{deleted_analysis}条过期分析结果")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def _trigger_callbacks(self, event_type: str, data: Dict):
        """触发回调函数"""
        try:
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"回调函数执行失败: {e}")
                    
        except Exception as e:
            logger.error(f"触发回调失败: {e}")
    
    def export_status_report(self, filepath: str) -> bool:
        """导出状态报告"""
        try:
            report = {
                'export_time': datetime.now().isoformat(),
                'processor_status': self.get_status(),
                'scheduler_stats': self.scheduler.get_statistics(),
                'latest_results': {}
            }
            
            # 获取各彩票类型的最新结果
            for lottery_type in ['双色球', '大乐透']:
                report['latest_results'][lottery_type] = self.get_latest_results(lottery_type)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"状态报告已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出状态报告失败: {e}")
            return False
