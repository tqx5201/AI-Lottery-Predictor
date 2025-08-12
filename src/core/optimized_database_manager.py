"""
优化的SQLite数据库管理模块
包含索引优化、连接池、查询缓存等性能优化功能
"""

import sqlite3
import json
import os
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from collections import OrderedDict
import logging

# 配置日志
logger = logging.getLogger(__name__)


class ConnectionPool:
    """数据库连接池"""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.pool = []
        self.used = set()
        self.lock = threading.Lock()
        
        # 预创建连接
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(min(3, self.max_connections)):  # 预创建3个连接
            conn = self._create_connection()
            self.pool.append(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """创建新的数据库连接"""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None  # 启用autocommit模式
        )
        conn.row_factory = sqlite3.Row
        
        # 性能优化设置
        conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式
        conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能和安全性
        conn.execute("PRAGMA cache_size=10000")  # 增加缓存页数
        conn.execute("PRAGMA temp_store=MEMORY")  # 临时表存储在内存
        conn.execute("PRAGMA mmap_size=268435456")  # 启用内存映射(256MB)
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """获取连接（上下文管理器）"""
        conn = None
        try:
            with self.lock:
                if self.pool:
                    conn = self.pool.pop()
                elif len(self.used) < self.max_connections:
                    conn = self._create_connection()
                else:
                    # 等待可用连接
                    while not self.pool and len(self.used) >= self.max_connections:
                        time.sleep(0.01)
                    if self.pool:
                        conn = self.pool.pop()
                
                if conn:
                    self.used.add(conn)
            
            yield conn
            
        finally:
            if conn:
                with self.lock:
                    self.used.discard(conn)
                    if len(self.pool) < self.max_connections // 2:
                        self.pool.append(conn)
                    else:
                        conn.close()
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.pool + list(self.used):
                try:
                    conn.close()
                except:
                    pass
            self.pool.clear()
            self.used.clear()


class QueryCache:
    """查询结果缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                # 检查是否过期
                if time.time() - self.timestamps[key] < self.ttl:
                    # 移到末尾（LRU）
                    self.cache.move_to_end(key)
                    return self.cache[key]
                else:
                    # 过期，删除
                    del self.cache[key]
                    del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        with self.lock:
            # 如果已存在，更新
            if key in self.cache:
                self.cache[key] = value
                self.timestamps[key] = time.time()
                self.cache.move_to_end(key)
                return
            
            # 检查缓存大小
            while len(self.cache) >= self.max_size:
                # 删除最旧的项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            # 添加新项
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


class OptimizedDatabaseManager:
    """优化的数据库管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化优化的数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        if db_path is None:
            # 确保数据目录存在
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "history_data")
            data_dir = os.path.abspath(data_dir)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            db_path = os.path.join(data_dir, "lottery_optimized.db")
        
        self.db_path = db_path
        self.connection_pool = ConnectionPool(db_path)
        self.query_cache = QueryCache()
        
        # 初始化数据库
        self.init_database()
        logger.info(f"优化的数据库管理器初始化完成: {db_path}")
    
    def init_database(self):
        """初始化数据库表结构和索引"""
        with self.connection_pool.get_connection() as conn:
            try:
                # 创建所有表
                self._create_tables(conn)
                
                # 创建索引
                self._create_indexes(conn)
                
                # 分析表统计信息
                conn.execute("ANALYZE")
                
                logger.info("数据库表结构和索引创建完成")
                
            except Exception as e:
                logger.error(f"数据库初始化失败: {e}")
                raise
    
    def _create_tables(self, conn: sqlite3.Connection):
        """创建数据库表"""
        tables = [
            # 历史开奖数据表
            '''
            CREATE TABLE IF NOT EXISTS lottery_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lottery_type TEXT NOT NULL,
                period TEXT NOT NULL,
                draw_date TEXT NOT NULL,
                numbers TEXT NOT NULL,
                red_numbers TEXT,  -- 红球号码（逗号分隔）
                blue_numbers TEXT, -- 蓝球号码（逗号分隔）
                sum_value INTEGER, -- 号码总和
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lottery_type, period)
            )
            ''',
            
            # 缓存数据表
            '''
            CREATE TABLE IF NOT EXISTS cache_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                lottery_type TEXT NOT NULL,
                period_range TEXT NOT NULL,
                data TEXT NOT NULL,
                data_size INTEGER, -- 数据大小
                access_count INTEGER DEFAULT 0, -- 访问次数
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            
            # 预测记录表
            '''
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lottery_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_data TEXT NOT NULL,
                numbers_data TEXT,
                target_period TEXT,
                confidence_score REAL,
                processing_time REAL, -- 处理时间（秒）
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            
            # 系统配置表
            '''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                config_type TEXT DEFAULT 'string', -- string, int, float, bool, json
                description TEXT,
                is_user_configurable BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''',
            
            # 预测验证表
            '''
            CREATE TABLE IF NOT EXISTS prediction_verification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                lottery_type TEXT NOT NULL,
                predicted_period TEXT NOT NULL,
                predicted_numbers TEXT NOT NULL,
                actual_numbers TEXT,
                accuracy_score REAL,
                hit_count INTEGER DEFAULT 0,
                red_hit_count INTEGER DEFAULT 0, -- 红球命中数
                blue_hit_count INTEGER DEFAULT 0, -- 蓝球命中数
                verification_status TEXT DEFAULT 'pending',
                verification_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES prediction_records (id)
            )
            ''',
            
            # 分析结果表
            '''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lottery_type TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                period_range TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                confidence_score REAL,
                processing_time REAL,
                data_hash TEXT, -- 数据哈希，用于检测变化
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
            ''',
            
            # 用户设置表
            '''
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_category TEXT NOT NULL,
                setting_key TEXT NOT NULL,
                setting_value TEXT NOT NULL,
                setting_type TEXT DEFAULT 'string',
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(setting_category, setting_key)
            )
            ''',
            
            # 性能监控表
            '''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                context_data TEXT, -- JSON格式的上下文信息
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        ]
        
        for table_sql in tables:
            conn.execute(table_sql)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """创建数据库索引"""
        indexes = [
            # lottery_history表索引
            "CREATE INDEX IF NOT EXISTS idx_lottery_type_date ON lottery_history(lottery_type, draw_date)",
            "CREATE INDEX IF NOT EXISTS idx_lottery_period ON lottery_history(lottery_type, period)",
            "CREATE INDEX IF NOT EXISTS idx_lottery_date ON lottery_history(draw_date)",
            "CREATE INDEX IF NOT EXISTS idx_lottery_sum ON lottery_history(sum_value)",
            
            # cache_data表索引
            "CREATE INDEX IF NOT EXISTS idx_cache_type_range ON cache_data(lottery_type, period_range)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_data(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_cache_accessed ON cache_data(last_accessed)",
            
            # prediction_records表索引
            "CREATE INDEX IF NOT EXISTS idx_pred_type_model ON prediction_records(lottery_type, model_name)",
            "CREATE INDEX IF NOT EXISTS idx_pred_date ON prediction_records(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_pred_target ON prediction_records(target_period)",
            
            # prediction_verification表索引
            "CREATE INDEX IF NOT EXISTS idx_verify_period ON prediction_verification(predicted_period)",
            "CREATE INDEX IF NOT EXISTS idx_verify_status ON prediction_verification(verification_status)",
            "CREATE INDEX IF NOT EXISTS idx_verify_score ON prediction_verification(accuracy_score)",
            
            # analysis_results表索引
            "CREATE INDEX IF NOT EXISTS idx_analysis_type_range ON analysis_results(lottery_type, analysis_type, period_range)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_expires ON analysis_results(expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_analysis_hash ON analysis_results(data_hash)",
            
            # user_settings表索引
            "CREATE INDEX IF NOT EXISTS idx_settings_category ON user_settings(setting_category)",
            
            # performance_metrics表索引
            "CREATE INDEX IF NOT EXISTS idx_metrics_name_date ON performance_metrics(metric_name, recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_date ON performance_metrics(recorded_at)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    def execute_query(self, query: str, params: tuple = None, use_cache: bool = True) -> List[sqlite3.Row]:
        """执行查询（带缓存）"""
        cache_key = None
        if use_cache:
            cache_key = f"{query}:{str(params)}"
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        start_time = time.time()
        with self.connection_pool.get_connection() as conn:
            try:
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)
                
                result = cursor.fetchall()
                
                # 记录性能指标
                execution_time = time.time() - start_time
                if execution_time > 0.1:  # 记录慢查询
                    self._record_performance_metric(
                        "slow_query_time", 
                        execution_time, 
                        "seconds",
                        {"query": query[:100], "params": str(params)[:100]}
                    )
                
                # 缓存结果
                if use_cache and cache_key:
                    self.query_cache.set(cache_key, result)
                
                return result
                
            except Exception as e:
                logger.error(f"查询执行失败: {query}, 参数: {params}, 错误: {e}")
                raise
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """执行更新操作"""
        with self.connection_pool.get_connection() as conn:
            try:
                if params:
                    cursor = conn.execute(query, params)
                else:
                    cursor = conn.execute(query)
                
                # 清除相关缓存
                self.query_cache.clear()
                
                return cursor.rowcount
                
            except Exception as e:
                logger.error(f"更新执行失败: {query}, 参数: {params}, 错误: {e}")
                raise
    
    def get_lottery_history(self, lottery_type: str, limit: int = None, 
                           start_date: str = None, end_date: str = None) -> List[Dict]:
        """获取历史数据（优化查询）"""
        query = """
            SELECT lottery_type, period, draw_date, numbers, red_numbers, blue_numbers, sum_value
            FROM lottery_history 
            WHERE lottery_type = ?
        """
        params = [lottery_type]
        
        if start_date:
            query += " AND draw_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND draw_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY draw_date DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        rows = self.execute_query(query, tuple(params))
        
        result = []
        for row in rows:
            try:
                numbers = json.loads(row['numbers'])
                result.append({
                    'lottery_type': row['lottery_type'],
                    'period': row['period'],
                    'date': row['draw_date'],
                    'numbers': numbers,
                    'red_numbers': row['red_numbers'],
                    'blue_numbers': row['blue_numbers'],
                    'sum_value': row['sum_value']
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"解析历史数据失败: {e}")
                continue
        
        return result
    
    def save_lottery_data(self, lottery_type: str, period: str, draw_date: str, numbers: Dict):
        """保存彩票数据（优化插入）"""
        try:
            # 提取红球和蓝球
            red_numbers = None
            blue_numbers = None
            sum_value = 0
            
            if lottery_type == "双色球":
                red_numbers = ",".join(map(str, numbers.get('red', [])))
                blue_numbers = ",".join(map(str, numbers.get('blue', [])))
                sum_value = sum(numbers.get('red', [])) + sum(numbers.get('blue', []))
            elif lottery_type == "大乐透":
                red_numbers = ",".join(map(str, numbers.get('front', [])))
                blue_numbers = ",".join(map(str, numbers.get('back', [])))
                sum_value = sum(numbers.get('front', [])) + sum(numbers.get('back', []))
            
            query = """
                INSERT OR REPLACE INTO lottery_history 
                (lottery_type, period, draw_date, numbers, red_numbers, blue_numbers, sum_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                lottery_type, period, draw_date, 
                json.dumps(numbers, ensure_ascii=False),
                red_numbers, blue_numbers, sum_value
            )
            
            self.execute_update(query, params)
            
        except Exception as e:
            logger.error(f"保存彩票数据失败: {e}")
            raise
    
    def _record_performance_metric(self, metric_name: str, value: float, 
                                  unit: str = None, context: Dict = None):
        """记录性能指标"""
        try:
            query = """
                INSERT INTO performance_metrics 
                (metric_name, metric_value, metric_unit, context_data)
                VALUES (?, ?, ?, ?)
            """
            
            context_json = json.dumps(context, ensure_ascii=False) if context else None
            params = (metric_name, value, unit, context_json)
            
            self.execute_update(query, params)
            
        except Exception as e:
            logger.warning(f"记录性能指标失败: {e}")
    
    def get_performance_metrics(self, metric_name: str = None, 
                               hours: int = 24) -> List[Dict]:
        """获取性能指标"""
        query = """
            SELECT metric_name, metric_value, metric_unit, context_data, recorded_at
            FROM performance_metrics
            WHERE recorded_at >= datetime('now', '-{} hours')
        """.format(hours)
        
        params = []
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        query += " ORDER BY recorded_at DESC"
        
        rows = self.execute_query(query, tuple(params) if params else None)
        
        result = []
        for row in rows:
            context = None
            if row['context_data']:
                try:
                    context = json.loads(row['context_data'])
                except json.JSONDecodeError:
                    pass
            
            result.append({
                'metric_name': row['metric_name'],
                'value': row['metric_value'],
                'unit': row['metric_unit'],
                'context': context,
                'recorded_at': row['recorded_at']
            })
        
        return result
    
    def cleanup_expired_data(self):
        """清理过期数据"""
        try:
            # 清理过期缓存
            self.execute_update(
                "DELETE FROM cache_data WHERE expires_at < datetime('now')"
            )
            
            # 清理过期分析结果
            self.execute_update(
                "DELETE FROM analysis_results WHERE expires_at < datetime('now')"
            )
            
            # 清理旧的性能指标（保留30天）
            self.execute_update(
                "DELETE FROM performance_metrics WHERE recorded_at < datetime('now', '-30 days')"
            )
            
            # 优化数据库
            with self.connection_pool.get_connection() as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
            
            logger.info("数据库清理完成")
            
        except Exception as e:
            logger.error(f"数据库清理失败: {e}")
    
    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        stats = {}
        
        try:
            # 表记录数统计
            tables = [
                'lottery_history', 'cache_data', 'prediction_records',
                'prediction_verification', 'analysis_results', 'user_settings',
                'performance_metrics'
            ]
            
            for table in tables:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = self.execute_query(count_query, use_cache=False)
                stats[f'{table}_count'] = result[0]['count'] if result else 0
            
            # 数据库文件大小
            if os.path.exists(self.db_path):
                stats['db_file_size'] = os.path.getsize(self.db_path)
            
            # 缓存统计
            stats['query_cache_size'] = len(self.query_cache.cache)
            stats['connection_pool_size'] = len(self.connection_pool.pool)
            stats['active_connections'] = len(self.connection_pool.used)
            
        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def close(self):
        """关闭数据库管理器"""
        try:
            self.connection_pool.close_all()
            self.query_cache.clear()
            logger.info("数据库管理器已关闭")
        except Exception as e:
            logger.error(f"关闭数据库管理器失败: {e}")
