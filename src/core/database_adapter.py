"""
数据库适配器
提供与原有JSON存储系统兼容的接口，实现无缝迁移
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, List
from .database_manager import DatabaseManager, DataMigration
import logging

logger = logging.getLogger(__name__)


class DatabaseAdapter:
    """数据库适配器 - 替代原有的JSON文件存储"""
    
    def __init__(self):
        """初始化数据库适配器"""
        self.db_manager = DatabaseManager()
        self.history_cache_file = os.path.join(
            os.path.dirname(__file__), "history_data", "history_cache.json"
        )
        
        # 如果存在旧的JSON文件，进行数据迁移
        self._migrate_json_data()
    
    def _migrate_json_data(self):
        """迁移JSON数据到SQLite"""
        if os.path.exists(self.history_cache_file):
            try:
                migration = DataMigration(self.db_manager, self.history_cache_file)
                migration.migrate_from_json()
                logger.info("JSON数据迁移完成")
            except Exception as e:
                logger.error(f"JSON数据迁移失败: {e}")
    
    def save_history_data(self, lottery_type: str, period: str, data: str):
        """
        保存历史数据到数据库
        兼容原有的save_history_data接口
        
        Args:
            lottery_type: 彩票类型
            period: 期数范围
            data: 历史数据
        """
        try:
            cache_key = f"{lottery_type}_{period}"
            self.db_manager.save_cache_data(
                cache_key=cache_key,
                lottery_type=lottery_type,
                period_range=period,
                data=data,
                expires_hours=24
            )
            logger.info(f"历史数据已保存: {cache_key}")
            
        except Exception as e:
            logger.error(f"保存历史数据失败: {e}")
            raise
    
    def get_cached_history_data(self, lottery_type: str, period: str) -> Optional[str]:
        """
        获取缓存的历史数据
        兼容原有的get_cached_history_data接口
        
        Args:
            lottery_type: 彩票类型
            period: 期数范围
            
        Returns:
            缓存的历史数据或None
        """
        try:
            cache_key = f"{lottery_type}_{period}"
            cached_data = self.db_manager.get_cache_data(cache_key)
            
            if cached_data:
                logger.info(f"使用缓存的历史数据: {cache_key}")
                return cached_data
            else:
                logger.info(f"无缓存数据: {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"获取缓存历史数据失败: {e}")
            return None
    
    def update_cache_status_callback(self, status: str, color: str):
        """
        缓存状态更新回调（用于UI更新）
        这个方法需要在主程序中设置具体的实现
        """
        pass
    
    def get_cache_status(self, lottery_type: str, period: str) -> tuple:
        """
        获取缓存状态
        
        Returns:
            (status, color) 元组
        """
        cache_key = f"{lottery_type}_{period}"
        cached_data = self.db_manager.get_cache_data(cache_key)
        
        if cached_data:
            return ("已缓存", "green")
        else:
            return ("无缓存", "red")
    
    def force_refresh_cache(self, lottery_type: str, period: str):
        """强制刷新缓存（删除现有缓存）"""
        cache_key = f"{lottery_type}_{period}"
        self.db_manager.delete_cache_data(cache_key)
        logger.info(f"已删除缓存: {cache_key}")
    
    def save_prediction_result(self, lottery_type: str, model_name: str, 
                             prediction_type: str, prediction_data: str,
                             extracted_numbers: list = None):
        """
        保存预测结果
        
        Args:
            lottery_type: 彩票类型
            model_name: AI模型名称
            prediction_type: 预测类型 (first/second)
            prediction_data: 预测结果文本
            extracted_numbers: 提取的号码列表
        """
        try:
            self.db_manager.save_prediction_record(
                lottery_type=lottery_type,
                model_name=model_name,
                prediction_type=prediction_type,
                prediction_data=prediction_data,
                numbers_data=extracted_numbers,
                target_period=None  # 可以后续扩展
            )
            logger.info(f"预测结果已保存: {lottery_type} {prediction_type}")
            
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")
    
    def get_prediction_history(self, lottery_type: str = None) -> list:
        """获取预测历史记录"""
        try:
            return self.db_manager.get_prediction_history(lottery_type)
        except Exception as e:
            logger.error(f"获取预测历史失败: {e}")
            return []
    
    def get_database_info(self) -> Dict:
        """获取数据库信息"""
        try:
            stats = self.db_manager.get_database_stats()
            
            # 格式化数据库大小
            if 'db_size' in stats:
                size_bytes = stats['db_size']
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                stats['db_size_formatted'] = size_str
            
            return stats
            
        except Exception as e:
            logger.error(f"获取数据库信息失败: {e}")
            return {}
    
    def backup_database(self, backup_dir: str = None):
        """备份数据库"""
        try:
            if backup_dir is None:
                backup_dir = os.path.join(os.path.dirname(__file__), "history_data", "backups")
            
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f"lottery_backup_{timestamp}.db")
            
            self.db_manager.backup_database(backup_file)
            return backup_file
            
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            raise
    
    def clean_old_data(self, days: int = 30):
        """清理旧数据"""
        try:
            # 清理过期缓存
            self.db_manager.clean_expired_cache()
            
            # 可以扩展：清理超过指定天数的预测记录等
            logger.info(f"已清理过期数据")
            
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def get_prediction_history(self, lottery_type: str = None, limit: int = 50) -> List[Dict]:
        """
        获取预测历史记录
        
        Args:
            lottery_type: 彩票类型筛选（可选）
            limit: 返回记录数量限制
            
        Returns:
            预测历史记录列表
        """
        try:
            conn = self.db_manager.get_connection()
            
            if lottery_type:
                cursor = conn.execute('''
                    SELECT * FROM prediction_records 
                    WHERE lottery_type = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (lottery_type, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM prediction_records 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
            
            history = []
            for row in cursor:
                record = dict(row)
                # 解析JSON数据
                if record.get('numbers_data'):
                    try:
                        record['numbers_data'] = json.loads(record['numbers_data'])
                    except json.JSONDecodeError:
                        record['numbers_data'] = []
                history.append(record)
            
            conn.close()
            return history
            
        except Exception as e:
            logger.error(f"获取预测历史失败: {e}")
            return []
    
    def backup_database(self) -> str:
        """备份数据库"""
        try:
            backup_file = self.db_manager.backup_database()
            return backup_file
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        try:
            self.db_manager.close()
        except Exception as e:
            logger.error(f"关闭数据库失败: {e}")


# 兼容性工具函数
def create_legacy_cache_structure(lottery_type: str, period: str, data: str) -> Dict:
    """
    创建与旧JSON格式兼容的缓存结构
    用于需要兼容旧代码的地方
    """
    return {
        "data": data,
        "timestamp": datetime.now().isoformat(),
        "lottery_type": lottery_type,
        "period": period
    }


def migrate_existing_json_cache():
    """
    迁移现有JSON缓存的独立函数
    可以作为一次性迁移脚本使用
    """
    json_file = os.path.join(os.path.dirname(__file__), "history_data", "history_cache.json")
    
    if not os.path.exists(json_file):
        print("没有找到现有的JSON缓存文件")
        return
    
    db_manager = DatabaseManager()
    migration = DataMigration(db_manager, json_file)
    
    try:
        migration.migrate_from_json()
        print("数据迁移完成")
        
        # 显示迁移结果
        stats = db_manager.get_database_stats()
        print(f"数据库统计: {stats}")
        
    except Exception as e:
        print(f"数据迁移失败: {e}")
    finally:
        db_manager.close()


if __name__ == "__main__":
    # 测试数据库适配器
    adapter = DatabaseAdapter()
    
    # 测试保存和获取数据
    test_data = "测试历史数据内容"
    adapter.save_history_data("双色球", "最近100期", test_data)
    
    retrieved_data = adapter.get_cached_history_data("双色球", "最近100期")
    print(f"获取数据: {retrieved_data is not None}")
    
    # 获取数据库信息
    db_info = adapter.get_database_info()
    print(f"数据库信息: {db_info}")
    
    adapter.close()
