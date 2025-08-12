"""
SQLite数据库管理模块
用于替代原有的JSON文件存储系统，提供更好的数据管理和查询性能
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径，默认为 history_data/lottery.db
        """
        if db_path is None:
            # 确保数据目录存在
            data_dir = os.path.join(os.path.dirname(__file__), "history_data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            db_path = os.path.join(data_dir, "lottery.db")
        
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用字典式访问
        return conn
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def create_tables(self):
        """创建数据库表结构（对外接口）"""
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        try:
            # 创建历史开奖数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lottery_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lottery_type TEXT NOT NULL,  -- 彩票类型：双色球、大乐透
                    period TEXT NOT NULL,        -- 期号
                    draw_date TEXT NOT NULL,     -- 开奖日期
                    numbers TEXT NOT NULL,       -- 开奖号码（JSON格式）
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lottery_type, period)
                )
            ''')
            
            # 创建缓存数据表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,  -- 缓存键
                    lottery_type TEXT NOT NULL,      -- 彩票类型
                    period_range TEXT NOT NULL,      -- 期数范围
                    data TEXT NOT NULL,              -- 缓存数据
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL   -- 过期时间
                )
            ''')
            
            # 创建预测记录表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lottery_type TEXT NOT NULL,      -- 彩票类型
                    model_name TEXT NOT NULL,        -- AI模型名称
                    prediction_type TEXT NOT NULL,   -- 预测类型：first, second
                    prediction_data TEXT NOT NULL,   -- 预测结果（JSON格式）
                    numbers_data TEXT,               -- 提取的号码（JSON格式）
                    target_period TEXT,              -- 目标期号
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建系统配置表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE NOT NULL,  -- 配置键
                    config_value TEXT NOT NULL,       -- 配置值
                    description TEXT,                 -- 配置描述
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建预测验证表（用于准确率统计）
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_verification (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,     -- 预测记录ID
                    lottery_type TEXT NOT NULL,          -- 彩票类型
                    predicted_period TEXT NOT NULL,      -- 预测期号
                    predicted_numbers TEXT NOT NULL,     -- 预测号码（JSON格式）
                    actual_numbers TEXT,                 -- 实际开奖号码（JSON格式）
                    accuracy_score REAL,                -- 准确率分数
                    hit_count INTEGER DEFAULT 0,        -- 命中号码数量
                    verification_status TEXT DEFAULT 'pending',  -- 验证状态：pending, verified, failed
                    verification_date TIMESTAMP,        -- 验证日期
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES prediction_records (id)
                )
            ''')
            
            # 创建数据分析结果表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lottery_type TEXT NOT NULL,          -- 彩票类型
                    analysis_type TEXT NOT NULL,         -- 分析类型：frequency, trend, pattern等
                    period_range TEXT NOT NULL,          -- 分析期数范围
                    analysis_data TEXT NOT NULL,         -- 分析结果（JSON格式）
                    confidence_score REAL,              -- 置信度评分
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP                 -- 分析结果过期时间
                )
            ''')
            
            # 创建用户设置表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_category TEXT NOT NULL,     -- 设置分类：visualization, analysis, export等
                    setting_key TEXT NOT NULL,          -- 设置键
                    setting_value TEXT NOT NULL,        -- 设置值
                    description TEXT,                    -- 设置描述
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(setting_category, setting_key)
                )
            ''')
            
            # 创建导出历史表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS export_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    export_type TEXT NOT NULL,          -- 导出类型：excel, pdf, image
                    export_content TEXT NOT NULL,       -- 导出内容类型：prediction, analysis, chart
                    file_path TEXT NOT NULL,            -- 导出文件路径
                    file_size INTEGER,                  -- 文件大小（字节）
                    export_params TEXT,                 -- 导出参数（JSON格式）
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引以提高查询性能
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lottery_type_period ON lottery_history(lottery_type, period)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_data(cache_key)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_data(expires_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_prediction_type ON prediction_records(lottery_type, prediction_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_verification_status ON prediction_verification(verification_status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results(lottery_type, analysis_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_expires ON analysis_results(expires_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_settings ON user_settings(setting_category, setting_key)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_export_type ON export_history(export_type, export_content)')
            
            conn.commit()
            logger.info("数据库初始化完成")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库初始化失败: {e}")
            raise
        finally:
            conn.close()
    
    def save_lottery_history(self, lottery_type: str, period: str, draw_date: str, numbers: Dict):
        """
        保存历史开奖数据 (支持500彩票网格式)
        
        Args:
            lottery_type: 彩票类型 (双色球/大乐透)
            period: 期号
            draw_date: 开奖日期
            numbers: 开奖号码字典 (500彩票网格式)
        """
        conn = self.get_connection()
        try:
            # 验证和标准化数据格式
            validated_numbers = self._validate_and_normalize_numbers(lottery_type, numbers)
            
            conn.execute('''
                INSERT OR REPLACE INTO lottery_history 
                (lottery_type, period, draw_date, numbers)
                VALUES (?, ?, ?, ?)
            ''', (lottery_type, period, draw_date, json.dumps(validated_numbers, ensure_ascii=False)))
            
            conn.commit()
            logger.info(f"保存历史数据: {lottery_type} {period} - {validated_numbers}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存历史数据失败: {e}")
            raise
        finally:
            conn.close()
    
    def _validate_and_normalize_numbers(self, lottery_type: str, numbers: Dict) -> Dict:
        """
        验证并标准化号码格式 (500彩票网格式)
        
        Args:
            lottery_type: 彩票类型
            numbers: 原始号码字典
            
        Returns:
            标准化后的号码字典
        """
        try:
            if lottery_type == '双色球':
                if 'red_balls' in numbers and 'blue_balls' in numbers:
                    red_balls = numbers['red_balls']
                    blue_balls = numbers['blue_balls']
                    
                    # 验证格式
                    if (isinstance(red_balls, list) and len(red_balls) == 6 and
                        isinstance(blue_balls, list) and len(blue_balls) == 1):
                        # 验证号码范围
                        if (all(1 <= x <= 33 for x in red_balls) and
                            all(1 <= x <= 16 for x in blue_balls)):
                            return {
                                'red_balls': sorted(red_balls),
                                'blue_balls': blue_balls
                            }
                
            elif lottery_type == '大乐透':
                if 'front_area' in numbers and 'back_area' in numbers:
                    front_area = numbers['front_area']
                    back_area = numbers['back_area']
                    
                    # 验证格式
                    if (isinstance(front_area, list) and len(front_area) == 5 and
                        isinstance(back_area, list) and len(back_area) == 2):
                        # 验证号码范围
                        if (all(1 <= x <= 35 for x in front_area) and
                            all(1 <= x <= 12 for x in back_area)):
                            return {
                                'front_area': sorted(front_area),
                                'back_area': sorted(back_area)
                            }
            
            # 如果验证失败，记录错误
            logger.warning(f"号码格式验证失败: {lottery_type} - {numbers}")
            return numbers  # 返回原始数据
            
        except Exception as e:
            logger.error(f"号码验证异常: {e}")
            return numbers
    
    def save_lottery_history_batch(self, lottery_type: str, data_list: List[Dict]):
        """
        批量保存历史开奖数据 (500彩票网格式)
        
        Args:
            lottery_type: 彩票类型
            data_list: 数据列表，每个元素包含 period, date, numbers
        """
        if not data_list:
            logger.warning("批量保存数据为空")
            return
        
        conn = self.get_connection()
        try:
            saved_count = 0
            skipped_count = 0
            
            for data in data_list:
                try:
                    period = data.get('period')
                    date = data.get('date')
                    numbers = data.get('numbers')
                    
                    if not all([period, date, numbers]):
                        logger.warning(f"数据不完整，跳过: {data}")
                        skipped_count += 1
                        continue
                    
                    # 验证和标准化数据格式
                    validated_numbers = self._validate_and_normalize_numbers(lottery_type, numbers)
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO lottery_history 
                        (lottery_type, period, draw_date, numbers)
                        VALUES (?, ?, ?, ?)
                    ''', (lottery_type, period, date, json.dumps(validated_numbers, ensure_ascii=False)))
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"保存单条数据失败: {data} - {e}")
                    skipped_count += 1
                    continue
            
            conn.commit()
            logger.info(f"批量保存完成: {lottery_type} - 成功{saved_count}期，跳过{skipped_count}期")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"批量保存失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_lottery_history(self, lottery_type: str, limit: int = 100) -> List[Dict]:
        """
        获取历史开奖数据
        
        Args:
            lottery_type: 彩票类型
            limit: 获取数量限制
            
        Returns:
            历史开奖数据列表
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT period, draw_date, numbers 
                FROM lottery_history 
                WHERE lottery_type = ? 
                ORDER BY period DESC 
                LIMIT ?
            ''', (lottery_type, limit))
            
            results = []
            for row in cursor:
                data = {
                    'period': row['period'],
                    'draw_date': row['draw_date'],
                    'numbers': json.loads(row['numbers'])
                }
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return []
        finally:
            conn.close()
    
    def save_cache_data(self, cache_key: str, lottery_type: str, period_range: str, 
                       data: str, expires_hours: int = 24):
        """
        保存缓存数据
        
        Args:
            cache_key: 缓存键
            lottery_type: 彩票类型
            period_range: 期数范围
            data: 缓存数据
            expires_hours: 过期时间（小时）
        """
        conn = self.get_connection()
        try:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            conn.execute('''
                INSERT OR REPLACE INTO cache_data 
                (cache_key, lottery_type, period_range, data, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (cache_key, lottery_type, period_range, data, expires_at.isoformat()))
            
            conn.commit()
            logger.info(f"保存缓存数据: {cache_key}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存缓存数据失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_cache_data(self, cache_key: str) -> Optional[str]:
        """
        获取缓存数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存数据或None（如果不存在或已过期）
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT data, expires_at 
                FROM cache_data 
                WHERE cache_key = ?
            ''', (cache_key,))
            
            row = cursor.fetchone()
            if row:
                expires_at = datetime.fromisoformat(row['expires_at'])
                if datetime.now() < expires_at:
                    logger.info(f"命中缓存: {cache_key}")
                    return row['data']
                else:
                    # 删除过期缓存
                    self.delete_cache_data(cache_key)
                    logger.info(f"缓存已过期: {cache_key}")
            
            return None
            
        except Exception as e:
            logger.error(f"获取缓存数据失败: {e}")
            return None
        finally:
            conn.close()
    
    def delete_cache_data(self, cache_key: str):
        """删除缓存数据"""
        conn = self.get_connection()
        try:
            conn.execute('DELETE FROM cache_data WHERE cache_key = ?', (cache_key,))
            conn.commit()
        except Exception as e:
            logger.error(f"删除缓存数据失败: {e}")
        finally:
            conn.close()
    
    def clean_expired_cache(self):
        """清理过期缓存"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                DELETE FROM cache_data 
                WHERE expires_at < CURRENT_TIMESTAMP
            ''')
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"清理过期缓存: {deleted_count}条")
                
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
        finally:
            conn.close()
    
    def save_prediction_record(self, lottery_type: str, model_name: str, 
                             prediction_type: str, prediction_data: str,
                             numbers_data: List = None, target_period: str = None):
        """
        保存预测记录
        
        Args:
            lottery_type: 彩票类型
            model_name: AI模型名称
            prediction_type: 预测类型
            prediction_data: 预测结果
            numbers_data: 提取的号码
            target_period: 目标期号
        """
        conn = self.get_connection()
        try:
            numbers_json = json.dumps(numbers_data, ensure_ascii=False) if numbers_data else None
            
            conn.execute('''
                INSERT INTO prediction_records 
                (lottery_type, model_name, prediction_type, prediction_data, numbers_data, target_period)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (lottery_type, model_name, prediction_type, prediction_data, numbers_json, target_period))
            
            conn.commit()
            logger.info(f"保存预测记录: {lottery_type} {prediction_type}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存预测记录失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_prediction_history(self, lottery_type: str = None, limit: int = 50) -> List[Dict]:
        """
        获取预测历史记录
        
        Args:
            lottery_type: 彩票类型（可选）
            limit: 获取数量限制
            
        Returns:
            预测历史记录列表
        """
        conn = self.get_connection()
        try:
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
            
            results = []
            for row in cursor:
                data = dict(row)
                if data['numbers_data']:
                    data['numbers_data'] = json.loads(data['numbers_data'])
                results.append(data)
            
            return results
            
        except Exception as e:
            logger.error(f"获取预测历史失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_config(self, config_key: str, default_value: str = None) -> Optional[str]:
        """获取系统配置"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT config_value FROM system_config WHERE config_key = ?
            ''', (config_key,))
            
            row = cursor.fetchone()
            return row['config_value'] if row else default_value
            
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            return default_value
        finally:
            conn.close()
    
    def set_config(self, config_key: str, config_value: str, description: str = None):
        """设置系统配置"""
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT OR REPLACE INTO system_config 
                (config_key, config_value, description, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (config_key, config_value, description))
            
            conn.commit()
            logger.info(f"设置配置: {config_key} = {config_value}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"设置配置失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        conn = self.get_connection()
        try:
            stats = {}
            
            # 历史数据统计
            cursor = conn.execute('SELECT lottery_type, COUNT(*) as count FROM lottery_history GROUP BY lottery_type')
            history_stats = {row['lottery_type']: row['count'] for row in cursor}
            stats['history_data'] = history_stats
            
            # 缓存数据统计
            cursor = conn.execute('SELECT COUNT(*) as total FROM cache_data WHERE expires_at > CURRENT_TIMESTAMP')
            stats['cache_count'] = cursor.fetchone()['total']
            
            # 预测记录统计
            cursor = conn.execute('SELECT COUNT(*) as total FROM prediction_records')
            stats['prediction_count'] = cursor.fetchone()['total']
            
            # 数据库文件大小
            if os.path.exists(self.db_path):
                stats['db_size'] = os.path.getsize(self.db_path)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            return {}
        finally:
            conn.close()
    
    def backup_database(self, backup_path: str):
        """备份数据库"""
        import shutil
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"数据库备份完成: {backup_path}")
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            raise
    
    def save_prediction_verification(self, prediction_id: int, lottery_type: str, 
                                   predicted_period: str, predicted_numbers: List,
                                   actual_numbers: List = None, accuracy_score: float = None):
        """
        保存预测验证记录
        
        Args:
            prediction_id: 预测记录ID
            lottery_type: 彩票类型
            predicted_period: 预测期号
            predicted_numbers: 预测号码
            actual_numbers: 实际开奖号码（可选）
            accuracy_score: 准确率分数（可选）
        """
        conn = self.get_connection()
        try:
            predicted_json = json.dumps(predicted_numbers, ensure_ascii=False)
            actual_json = json.dumps(actual_numbers, ensure_ascii=False) if actual_numbers else None
            
            # 计算命中数量
            hit_count = 0
            if actual_numbers and predicted_numbers:
                hit_count = self._calculate_hit_count(predicted_numbers, actual_numbers, lottery_type)
            
            # 确定验证状态
            status = 'verified' if actual_numbers else 'pending'
            verification_date = datetime.now().isoformat() if actual_numbers else None
            
            conn.execute('''
                INSERT INTO prediction_verification 
                (prediction_id, lottery_type, predicted_period, predicted_numbers, 
                 actual_numbers, accuracy_score, hit_count, verification_status, verification_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (prediction_id, lottery_type, predicted_period, predicted_json, 
                  actual_json, accuracy_score, hit_count, status, verification_date))
            
            conn.commit()
            logger.info(f"保存预测验证记录: {lottery_type} {predicted_period}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存预测验证记录失败: {e}")
            raise
        finally:
            conn.close()
    
    def _calculate_hit_count(self, predicted: List, actual: List, lottery_type: str) -> int:
        """计算命中号码数量"""
        try:
            if lottery_type == "双色球":
                # 双色球：6个红球 + 1个蓝球
                if len(predicted) >= 7 and len(actual) >= 7:
                    red_hits = len(set(predicted[:6]) & set(actual[:6]))
                    blue_hits = 1 if predicted[6] == actual[6] else 0
                    return red_hits + blue_hits
            elif lottery_type == "大乐透":
                # 大乐透：5个前区 + 2个后区
                if len(predicted) >= 7 and len(actual) >= 7:
                    front_hits = len(set(predicted[:5]) & set(actual[:5]))
                    back_hits = len(set(predicted[5:7]) & set(actual[5:7]))
                    return front_hits + back_hits
            return 0
        except Exception:
            return 0
    
    def save_analysis_result(self, lottery_type: str, analysis_type: str, 
                           period_range: str, analysis_data: Dict, 
                           confidence_score: float = None, expires_hours: int = 168):
        """
        保存数据分析结果
        
        Args:
            lottery_type: 彩票类型
            analysis_type: 分析类型
            period_range: 期数范围
            analysis_data: 分析结果数据
            confidence_score: 置信度评分
            expires_hours: 过期时间（小时，默认1周）
        """
        conn = self.get_connection()
        try:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            data_json = json.dumps(analysis_data, ensure_ascii=False)
            
            conn.execute('''
                INSERT OR REPLACE INTO analysis_results 
                (lottery_type, analysis_type, period_range, analysis_data, confidence_score, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (lottery_type, analysis_type, period_range, data_json, confidence_score, expires_at.isoformat()))
            
            conn.commit()
            logger.info(f"保存分析结果: {lottery_type} {analysis_type}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存分析结果失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_analysis_result(self, lottery_type: str, analysis_type: str, 
                          period_range: str) -> Optional[Dict]:
        """获取分析结果"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT analysis_data, confidence_score, created_at, expires_at 
                FROM analysis_results 
                WHERE lottery_type = ? AND analysis_type = ? AND period_range = ?
                AND expires_at > CURRENT_TIMESTAMP
                ORDER BY created_at DESC LIMIT 1
            ''', (lottery_type, analysis_type, period_range))
            
            row = cursor.fetchone()
            if row:
                return {
                    'data': json.loads(row['analysis_data']),
                    'confidence_score': row['confidence_score'],
                    'created_at': row['created_at'],
                    'expires_at': row['expires_at']
                }
            return None
            
        except Exception as e:
            logger.error(f"获取分析结果失败: {e}")
            return None
        finally:
            conn.close()
    
    def get_prediction_accuracy_stats(self, lottery_type: str = None, 
                                    days: int = 30) -> Dict:
        """获取预测准确率统计"""
        conn = self.get_connection()
        try:
            # 基础查询条件
            where_clause = "WHERE verification_status = 'verified'"
            params = []
            
            if lottery_type:
                where_clause += " AND lottery_type = ?"
                params.append(lottery_type)
            
            where_clause += " AND created_at >= ?"
            params.append((datetime.now() - timedelta(days=days)).isoformat())
            
            # 总体统计
            cursor = conn.execute(f'''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(hit_count) as avg_hit_count,
                    MAX(hit_count) as max_hit_count,
                    AVG(accuracy_score) as avg_accuracy_score
                FROM prediction_verification 
                {where_clause}
            ''', params)
            
            stats = dict(cursor.fetchone())
            
            # 按彩票类型分组统计
            cursor = conn.execute(f'''
                SELECT 
                    lottery_type,
                    COUNT(*) as count,
                    AVG(hit_count) as avg_hits,
                    AVG(accuracy_score) as avg_accuracy
                FROM prediction_verification 
                {where_clause}
                GROUP BY lottery_type
            ''', params)
            
            stats['by_lottery_type'] = {row['lottery_type']: dict(row) for row in cursor}
            
            return stats
            
        except Exception as e:
            logger.error(f"获取预测准确率统计失败: {e}")
            return {}
        finally:
            conn.close()
    
    def save_user_setting(self, category: str, key: str, value: str, description: str = None):
        """保存用户设置"""
        conn = self.get_connection()
        try:
            conn.execute('''
                INSERT OR REPLACE INTO user_settings 
                (setting_category, setting_key, setting_value, description, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (category, key, value, description))
            
            conn.commit()
            logger.info(f"保存用户设置: {category}.{key}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存用户设置失败: {e}")
            raise
        finally:
            conn.close()
    
    def get_user_setting(self, category: str, key: str, default_value: str = None) -> str:
        """获取用户设置"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                SELECT setting_value FROM user_settings 
                WHERE setting_category = ? AND setting_key = ?
            ''', (category, key))
            
            row = cursor.fetchone()
            return row['setting_value'] if row else default_value
            
        except Exception as e:
            logger.error(f"获取用户设置失败: {e}")
            return default_value
        finally:
            conn.close()
    
    def save_export_record(self, export_type: str, export_content: str, 
                         file_path: str, file_size: int = None, 
                         export_params: Dict = None):
        """保存导出记录"""
        conn = self.get_connection()
        try:
            params_json = json.dumps(export_params, ensure_ascii=False) if export_params else None
            
            conn.execute('''
                INSERT INTO export_history 
                (export_type, export_content, file_path, file_size, export_params)
                VALUES (?, ?, ?, ?, ?)
            ''', (export_type, export_content, file_path, file_size, params_json))
            
            conn.commit()
            logger.info(f"保存导出记录: {export_type} {export_content}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"保存导出记录失败: {e}")
            raise
        finally:
            conn.close()

    def close(self):
        """清理资源"""
        # 清理过期缓存
        self.clean_expired_cache()
        # 清理过期分析结果
        self._clean_expired_analysis()
        logger.info("数据库管理器已关闭")
    
    def _clean_expired_analysis(self):
        """清理过期分析结果"""
        conn = self.get_connection()
        try:
            cursor = conn.execute('''
                DELETE FROM analysis_results 
                WHERE expires_at < CURRENT_TIMESTAMP
            ''')
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"清理过期分析结果: {deleted_count}条")
                
        except Exception as e:
            logger.error(f"清理过期分析结果失败: {e}")
        finally:
            conn.close()


# 数据迁移工具
class DataMigration:
    """数据迁移工具：从JSON文件迁移到SQLite"""
    
    def __init__(self, db_manager: DatabaseManager, json_file_path: str):
        self.db_manager = db_manager
        self.json_file_path = json_file_path
    
    def migrate_from_json(self):
        """从JSON文件迁移数据到SQLite"""
        if not os.path.exists(self.json_file_path):
            logger.warning(f"JSON文件不存在: {self.json_file_path}")
            return
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            migrated_count = 0
            
            for cache_key, cache_info in json_data.items():
                if isinstance(cache_info, dict) and 'data' in cache_info:
                    lottery_type = cache_info.get('lottery_type', '')
                    period = cache_info.get('period', '')
                    data = cache_info.get('data', '')
                    timestamp = cache_info.get('timestamp', '')
                    
                    # 计算过期时间（假设原数据24小时过期）
                    if timestamp:
                        created_time = datetime.fromisoformat(timestamp)
                        expires_time = created_time + timedelta(hours=24)
                        
                        # 只迁移未过期的数据
                        if datetime.now() < expires_time:
                            self.db_manager.save_cache_data(
                                cache_key=cache_key,
                                lottery_type=lottery_type,
                                period_range=period,
                                data=data,
                                expires_hours=24
                            )
                            migrated_count += 1
            
            logger.info(f"数据迁移完成: 成功迁移 {migrated_count} 条缓存数据")
            
            # 备份原JSON文件
            backup_path = f"{self.json_file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(self.json_file_path, backup_path)
            logger.info(f"原JSON文件已备份: {backup_path}")
            
        except Exception as e:
            logger.error(f"数据迁移失败: {e}")
            raise


if __name__ == "__main__":
    # 测试数据库功能
    db = DatabaseManager()
    
    # 测试保存和获取缓存数据
    test_data = "测试缓存数据"
    db.save_cache_data("test_key", "双色球", "最近100期", test_data)
    
    retrieved_data = db.get_cache_data("test_key")
    print(f"缓存测试: {retrieved_data}")
    
    # 获取数据库统计
    stats = db.get_database_stats()
    print(f"数据库统计: {stats}")
    
    db.close()
