"""
AI彩票预测系统 - 自动数据分析模块
智能分析历史开奖数据，发现规律和趋势
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
import json
import math
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import hashlib
from multiprocessing import cpu_count

try:
    from core.database_manager import DatabaseManager
except ImportError:
    from ..core.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    analysis_type: str
    confidence_score: float
    patterns: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime


class LotteryAnalysis:
    """彩票数据分析类"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        初始化分析模块
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager or DatabaseManager()
        
        # 性能优化配置
        self.max_workers = min(4, cpu_count())  # 并行处理线程数
        self.cache_timeout_hours = 6  # 缓存超时时间
        self.memory_cache = {}  # 内存缓存
        self.cache_max_size = 100  # 最大缓存条目数
        
        # 彩票规则配置
        self.lottery_config = {
            "双色球": {
                "red_count": 6,
                "blue_count": 1,
                "red_range": (1, 33),
                "blue_range": (1, 16),
                "total_numbers": 7
            },
            "大乐透": {
                "front_count": 5,
                "back_count": 2,
                "front_range": (1, 35),
                "back_range": (1, 12),
                "total_numbers": 7
            }
        }
        
        # 分析权重配置
        self.analysis_weights = {
            'frequency': 0.25,    # 频率分析权重
            'trend': 0.20,        # 趋势分析权重
            'pattern': 0.20,      # 形态分析权重
            'missing': 0.15,      # 遗漏分析权重
            'correlation': 0.10,  # 相关性分析权重
            'statistical': 0.10   # 统计分析权重
        }
    
    def comprehensive_analysis(self, lottery_type: str, period_range: str = "最近100期",
                             force_refresh: bool = False, use_parallel: bool = True) -> Dict[str, Any]:
        """
        综合数据分析
        
        Args:
            lottery_type: 彩票类型
            period_range: 期数范围
            force_refresh: 是否强制刷新分析
            
        Returns:
            综合分析结果
        """
        try:
            # 检查内存缓存
            cache_key = self._generate_cache_key(lottery_type, period_range, 'comprehensive')
            if not force_refresh and cache_key in self.memory_cache:
                cached_data, cache_time = self.memory_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_timeout_hours * 3600:
                    logger.info(f"使用内存缓存的综合分析结果: {lottery_type}")
                    return cached_data
            
            # 检查数据库缓存
            if not force_refresh:
                cached_result = self.db_manager.get_analysis_result(
                    lottery_type, 'comprehensive', period_range
                )
                if cached_result:
                    # 更新内存缓存
                    self._update_memory_cache(cache_key, cached_result)
                    logger.info(f"使用数据库缓存的综合分析结果: {lottery_type}")
                    return cached_result
            
            # 获取历史数据
            history_data = self._get_history_data(lottery_type, period_range)
            
            if not history_data:
                logger.warning(f"历史数据为空，返回基础分析结果: {lottery_type}")
                # 如果没有历史数据，返回基础的分析结果而不是错误
                return {
                    'lottery_type': lottery_type,
                    'period_range': period_range,
                    'analysis_date': datetime.now().isoformat(),
                    'data_count': 0,
                    'confidence_score': 0.0,
                    'recommendations': ['暂无历史数据，无法进行深度分析', '建议获取更多历史数据后重试'],
                    'scores': {
                        'regularity_score': 0.0,
                        'randomness_score': 50.0,
                        'hotness_score': 0.0,
                        'stability_score': 0.0,
                        'overall_score': 12.5
                    },
                    'error': '无历史数据'
                }
            
            # 执行各项分析（支持并行处理）
            analysis_results = {}
            
            if use_parallel and len(history_data) > 50:  # 数据量较大时使用并行处理
                analysis_results = self._parallel_analysis(history_data, lottery_type)
            else:
                # 顺序执行分析
                analysis_tasks = [
                    ('frequency', self._analyze_frequency),
                    ('trend', self._analyze_trend),
                    ('pattern', self._analyze_pattern),
                    ('missing', self._analyze_missing),
                    ('correlation', self._analyze_correlation),
                    ('statistical', self._analyze_statistical),
                    ('hot_cold', self._analyze_hot_cold),
                    ('sum_analysis', self._analyze_sum),
                    ('span', self._analyze_span),
                    ('consecutive', self._analyze_consecutive),
                    ('repeat', self._analyze_repeat),
                    ('distribution', self._analyze_distribution_advanced),  # 新增分析
                    ('volatility', self._analyze_volatility),  # 新增分析
                    ('cycle', self._analyze_cycle_patterns)  # 新增分析
                ]
                
                for analysis_name, analysis_func in analysis_tasks:
                    try:
                        analysis_results[analysis_name] = analysis_func(history_data, lottery_type)
                    except Exception as e:
                        logger.error(f"{analysis_name}分析失败: {e}")
                        analysis_results[analysis_name] = {'error': str(e)}
            
            # 计算综合置信度
            confidence_score = self._calculate_comprehensive_confidence(analysis_results)
            
            # 生成综合建议
            recommendations = self._generate_recommendations(analysis_results, lottery_type)
            
            # 生成综合评分
            scores = self._calculate_comprehensive_scores(analysis_results)
            
            # 整合结果
            comprehensive_result = {
                'lottery_type': lottery_type,
                'period_range': period_range,
                'analysis_date': datetime.now().isoformat(),
                'data_count': len(history_data),
                'confidence_score': confidence_score,
                'recommendations': recommendations,
                'scores': scores,
                **analysis_results
            }
            
            # 保存分析结果到数据库和内存缓存
            serializable_result = self._make_json_serializable(comprehensive_result)
            self.db_manager.save_analysis_result(
                lottery_type=lottery_type,
                analysis_type='comprehensive',
                period_range=period_range,
                analysis_data=serializable_result,
                confidence_score=confidence_score,
                expires_hours=self.cache_timeout_hours
            )
            
            # 更新内存缓存
            self._update_memory_cache(cache_key, comprehensive_result)
            
            logger.info(f"综合分析完成: {lottery_type}, 置信度: {confidence_score:.2f}")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"综合分析失败: {e}")
            return {
                'error': f'分析失败: {str(e)}',
                'confidence_score': 0
            }
    
    def _get_history_data(self, lottery_type: str, period_range: str) -> List[Dict]:
        """获取历史数据"""
        try:
            # 解析期数范围
            period_map = {
                "最近50期": 50,
                "最近100期": 100,
                "最近200期": 200,
                "最近500期": 500,
                "最近1000期": 1000
            }
            limit = period_map.get(period_range, 100)
            
            # 从数据库获取历史数据
            history_records = self.db_manager.get_lottery_history(lottery_type, limit)
            
            if not history_records:
                logger.warning(f"未获取到历史数据: {lottery_type} {period_range}")
                # 如果没有历史数据，返回空列表但不影响其他分析
                return []
            
            return history_records
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return []
    
    def _analyze_frequency(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """频率分析"""
        try:
            config = self.lottery_config[lottery_type]
            
            if lottery_type == "双色球":
                red_freq = Counter()
                blue_freq = Counter()
                
                for record in history_data:
                    numbers = record.get('numbers', {})
                    if isinstance(numbers, str):
                        numbers = json.loads(numbers)
                    
                    red_nums = numbers.get('red', '').split(',')
                    blue_num = numbers.get('blue', '')
                    
                    for num in red_nums:
                        if num.strip().isdigit():
                            red_freq[int(num.strip())] += 1
                    
                    if blue_num.strip().isdigit():
                        blue_freq[int(blue_num.strip())] += 1
                
                return {
                    'red_frequency': dict(red_freq),
                    'blue_frequency': dict(blue_freq),
                    'hot_red': red_freq.most_common(10),
                    'cold_red': red_freq.most_common()[-10:],
                    'hot_blue': blue_freq.most_common(5),
                    'cold_blue': blue_freq.most_common()[-5:],
                    'analysis_quality': 'high' if len(history_data) >= 100 else 'medium'
                }
            
            else:  # 大乐透
                front_freq = Counter()
                back_freq = Counter()
                
                for record in history_data:
                    numbers = record.get('numbers', {})
                    if isinstance(numbers, str):
                        numbers = json.loads(numbers)
                    
                    front_nums = numbers.get('front', '').split(',')
                    back_nums = numbers.get('back', '').split(',')
                    
                    for num in front_nums:
                        if num.strip().isdigit():
                            front_freq[int(num.strip())] += 1
                    
                    for num in back_nums:
                        if num.strip().isdigit():
                            back_freq[int(num.strip())] += 1
                
                return {
                    'front_frequency': dict(front_freq),
                    'back_frequency': dict(back_freq),
                    'hot_front': front_freq.most_common(10),
                    'cold_front': front_freq.most_common()[-10:],
                    'hot_back': back_freq.most_common(5),
                    'cold_back': back_freq.most_common()[-5:],
                    'analysis_quality': 'high' if len(history_data) >= 100 else 'medium'
                }
                
        except Exception as e:
            logger.error(f"频率分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_trend(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """趋势分析"""
        try:
            if len(history_data) < 20:
                return {'error': '数据不足，无法进行趋势分析'}
            
            trends = {
                'increasing_numbers': [],
                'decreasing_numbers': [],
                'stable_numbers': [],
                'trend_strength': 0.0
            }
            
            # 分析最近20期的趋势
            recent_data = history_data[:20]
            
            if lottery_type == "双色球":
                # 分析红球趋势
                red_trends = defaultdict(list)
                
                for i, record in enumerate(recent_data):
                    numbers = record.get('numbers', {})
                    if isinstance(numbers, str):
                        numbers = json.loads(numbers)
                    
                    red_nums = numbers.get('red', '').split(',')
                    for num in red_nums:
                        if num.strip().isdigit():
                            red_trends[int(num.strip())].append(i)
                
                # 计算趋势
                for num, positions in red_trends.items():
                    if len(positions) >= 3:
                        # 计算趋势斜率
                        x = np.array(positions)
                        y = np.arange(len(positions))
                        slope, _, r_value, _, _ = stats.linregress(x, y)
                        
                        if slope < -0.1 and r_value < -0.5:
                            trends['increasing_numbers'].append((num, abs(slope)))
                        elif slope > 0.1 and r_value > 0.5:
                            trends['decreasing_numbers'].append((num, slope))
                        else:
                            trends['stable_numbers'].append((num, slope))
            
            else:  # 大乐透
                # 类似的逻辑处理大乐透前区和后区
                pass
            
            # 计算趋势强度
            total_numbers = len(trends['increasing_numbers']) + len(trends['decreasing_numbers']) + len(trends['stable_numbers'])
            if total_numbers > 0:
                trends['trend_strength'] = (len(trends['increasing_numbers']) + len(trends['decreasing_numbers'])) / total_numbers
            
            return trends
            
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_pattern(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """形态分析（奇偶、大小、质合等）"""
        try:
            patterns = {
                'odd_even_distribution': {},
                'big_small_distribution': {},
                'prime_composite_distribution': {},
                'pattern_frequency': {}
            }
            
            config = self.lottery_config[lottery_type]
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = [int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()]
                    
                    if len(red_nums) == 6:
                        # 奇偶分析
                        odd_count = sum(1 for num in red_nums if num % 2 == 1)
                        even_count = 6 - odd_count
                        odd_even_pattern = f"{odd_count}奇{even_count}偶"
                        patterns['odd_even_distribution'][odd_even_pattern] = patterns['odd_even_distribution'].get(odd_even_pattern, 0) + 1
                        
                        # 大小分析
                        big_count = sum(1 for num in red_nums if num > 16.5)
                        small_count = 6 - big_count
                        big_small_pattern = f"{big_count}大{small_count}小"
                        patterns['big_small_distribution'][big_small_pattern] = patterns['big_small_distribution'].get(big_small_pattern, 0) + 1
                        
                        # 质合分析
                        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
                        prime_count = sum(1 for num in red_nums if num in primes)
                        composite_count = 6 - prime_count
                        prime_pattern = f"{prime_count}质{composite_count}合"
                        patterns['prime_composite_distribution'][prime_pattern] = patterns['prime_composite_distribution'].get(prime_pattern, 0) + 1
                
                else:  # 大乐透
                    front_nums = [int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()]
                    
                    if len(front_nums) == 5:
                        # 类似的分析逻辑
                        pass
            
            # 找出最常见的形态模式
            patterns['most_common_odd_even'] = max(patterns['odd_even_distribution'].items(), key=lambda x: x[1]) if patterns['odd_even_distribution'] else None
            patterns['most_common_big_small'] = max(patterns['big_small_distribution'].items(), key=lambda x: x[1]) if patterns['big_small_distribution'] else None
            
            return patterns
            
        except Exception as e:
            logger.error(f"形态分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_missing(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """遗漏分析"""
        try:
            config = self.lottery_config[lottery_type]
            missing_analysis = {}
            
            if lottery_type == "双色球":
                # 分析红球和蓝球的遗漏情况
                red_missing = {i: 0 for i in range(1, 34)}
                blue_missing = {i: 0 for i in range(1, 17)}
                
                red_last_appear = {i: len(history_data) for i in range(1, 34)}
                blue_last_appear = {i: len(history_data) for i in range(1, 17)}
                
                for idx, record in enumerate(history_data):
                    numbers = record.get('numbers', {})
                    if isinstance(numbers, str):
                        numbers = json.loads(numbers)
                    
                    red_nums = [int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()]
                    blue_num = numbers.get('blue', '')
                    
                    # 更新红球遗漏
                    for num in red_nums:
                        if num in red_last_appear:
                            red_missing[num] = idx - red_last_appear[num] + 1
                            red_last_appear[num] = idx
                    
                    # 更新蓝球遗漏
                    if blue_num.strip().isdigit():
                        blue_int = int(blue_num.strip())
                        if blue_int in blue_last_appear:
                            blue_missing[blue_int] = idx - blue_last_appear[blue_int] + 1
                            blue_last_appear[blue_int] = idx
                
                # 计算当前遗漏值
                current_red_missing = {num: len(history_data) - red_last_appear[num] for num in range(1, 34)}
                current_blue_missing = {num: len(history_data) - blue_last_appear[num] for num in range(1, 17)}
                
                missing_analysis = {
                    'red_missing': current_red_missing,
                    'blue_missing': current_blue_missing,
                    'max_red_missing': max(current_red_missing.values()),
                    'max_blue_missing': max(current_blue_missing.values()),
                    'high_missing_red': [num for num, missing in current_red_missing.items() if missing > 20],
                    'high_missing_blue': [num for num, missing in current_blue_missing.items() if missing > 10]
                }
            
            else:  # 大乐透
                # 类似的逻辑处理大乐透
                pass
            
            return missing_analysis
            
        except Exception as e:
            logger.error(f"遗漏分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_correlation(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """相关性分析"""
        try:
            if len(history_data) < 50:
                return {'error': '数据不足，无法进行相关性分析'}
            
            correlation_analysis = {}
            
            # 构建数据矩阵
            data_matrix = []
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = [int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()]
                    if len(red_nums) == 6:
                        # 创建号码向量（1-33的二进制表示）
                        vector = [1 if i in red_nums else 0 for i in range(1, 34)]
                        data_matrix.append(vector)
                
                else:  # 大乐透
                    front_nums = [int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()]
                    if len(front_nums) == 5:
                        # 创建前区号码向量
                        vector = [1 if i in front_nums else 0 for i in range(1, 36)]
                        data_matrix.append(vector)
            
            if len(data_matrix) >= 20:
                # 计算相关系数矩阵
                df = pd.DataFrame(data_matrix)
                corr_matrix = df.corr()
                
                # 找出高相关性的号码对
                high_correlations = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.3:  # 相关性阈值
                            high_correlations.append({
                                'number1': i + 1,
                                'number2': j + 1,
                                'correlation': corr_value
                            })
                
                correlation_analysis = {
                    'high_correlations': high_correlations,
                    'correlation_strength': 'high' if len(high_correlations) > 10 else 'low',
                    'most_correlated_pairs': sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True)[:5]
                }
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"相关性分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_statistical(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """统计学分析"""
        try:
            statistical_analysis = {}
            
            # 收集数据
            all_numbers = []
            sums = []
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = [int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()]
                    if len(red_nums) == 6:
                        all_numbers.extend(red_nums)
                        sums.append(sum(red_nums))
                
                else:  # 大乐透
                    front_nums = [int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()]
                    if len(front_nums) == 5:
                        all_numbers.extend(front_nums)
                        sums.append(sum(front_nums))
            
            if all_numbers and sums:
                # 基本统计
                statistical_analysis = {
                    'mean': np.mean(all_numbers),
                    'median': np.median(all_numbers),
                    'std': np.std(all_numbers),
                    'variance': np.var(all_numbers),
                    'sum_mean': np.mean(sums),
                    'sum_std': np.std(sums),
                    'sum_range': (min(sums), max(sums)),
                    'distribution_type': self._analyze_distribution(all_numbers)
                }
                
                # 正态性检验
                if len(all_numbers) > 50:
                    _, p_value = stats.normaltest(all_numbers)
                    statistical_analysis['normality_p_value'] = p_value
                    statistical_analysis['is_normal'] = p_value > 0.05
            
            return statistical_analysis
            
        except Exception as e:
            logger.error(f"统计学分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_distribution(self, numbers: List[int]) -> str:
        """分析数字分布类型"""
        try:
            # 简单的分布类型判断
            mean_val = np.mean(numbers)
            median_val = np.median(numbers)
            
            if abs(mean_val - median_val) < 1:
                return "近似正态分布"
            elif mean_val > median_val:
                return "右偏分布"
            else:
                return "左偏分布"
                
        except:
            return "未知分布"
    
    def _analyze_hot_cold(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """冷热号分析"""
        try:
            frequency_result = self._analyze_frequency(history_data, lottery_type)
            
            if 'error' in frequency_result:
                return frequency_result
            
            hot_cold_analysis = {}
            
            if lottery_type == "双色球":
                red_freq = frequency_result.get('red_frequency', {})
                blue_freq = frequency_result.get('blue_frequency', {})
                
                # 计算平均频率
                avg_red_freq = np.mean(list(red_freq.values())) if red_freq else 0
                avg_blue_freq = np.mean(list(blue_freq.values())) if blue_freq else 0
                
                # 分类冷热号
                hot_red = [num for num, freq in red_freq.items() if freq > avg_red_freq * 1.2]
                cold_red = [num for num, freq in red_freq.items() if freq < avg_red_freq * 0.8]
                hot_blue = [num for num, freq in blue_freq.items() if freq > avg_blue_freq * 1.2]
                cold_blue = [num for num, freq in blue_freq.items() if freq < avg_blue_freq * 0.8]
                
                hot_cold_analysis = {
                    'hot': {
                        'red': hot_red,
                        'blue': hot_blue
                    },
                    'cold': {
                        'red': cold_red,
                        'blue': cold_blue
                    },
                    'avg_frequency': {
                        'red': avg_red_freq,
                        'blue': avg_blue_freq
                    }
                }
            
            else:  # 大乐透
                front_freq = frequency_result.get('front_frequency', {})
                back_freq = frequency_result.get('back_frequency', {})
                
                avg_front_freq = np.mean(list(front_freq.values())) if front_freq else 0
                avg_back_freq = np.mean(list(back_freq.values())) if back_freq else 0
                
                hot_front = [num for num, freq in front_freq.items() if freq > avg_front_freq * 1.2]
                cold_front = [num for num, freq in front_freq.items() if freq < avg_front_freq * 0.8]
                hot_back = [num for num, freq in back_freq.items() if freq > avg_back_freq * 1.2]
                cold_back = [num for num, freq in back_freq.items() if freq < avg_back_freq * 0.8]
                
                hot_cold_analysis = {
                    'hot': {
                        'front': hot_front,
                        'back': hot_back
                    },
                    'cold': {
                        'front': cold_front,
                        'back': cold_back
                    },
                    'avg_frequency': {
                        'front': avg_front_freq,
                        'back': avg_back_freq
                    }
                }
            
            return hot_cold_analysis
            
        except Exception as e:
            logger.error(f"冷热号分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_sum(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """和值分析"""
        try:
            sums = []
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = [int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()]
                    if len(red_nums) == 6:
                        sums.append(sum(red_nums))
                
                else:  # 大乐透
                    front_nums = [int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()]
                    if len(front_nums) == 5:
                        sums.append(sum(front_nums))
            
            if not sums:
                return {'error': '无有效和值数据'}
            
            # 和值统计分析
            sum_analysis = {
                'recent_sums': sums[:20],  # 最近20期和值
                'mean_sum': np.mean(sums),
                'median_sum': np.median(sums),
                'min_sum': min(sums),
                'max_sum': max(sums),
                'std_sum': np.std(sums),
                'sum_distribution': dict(Counter(sums)),
                'sum_trend': 'increasing' if len(sums) > 10 and np.corrcoef(range(10), sums[:10])[0,1] > 0.3 else 'stable'
            }
            
            # 和值区间分析
            if lottery_type == "双色球":
                # 双色球红球和值理论范围：21-183
                ranges = {
                    'low': (21, 90),
                    'medium': (91, 140),
                    'high': (141, 183)
                }
            else:
                # 大乐透前区和值理论范围：15-155
                ranges = {
                    'low': (15, 70),
                    'medium': (71, 120),
                    'high': (121, 155)
                }
            
            range_counts = {range_name: 0 for range_name in ranges.keys()}
            for sum_val in sums:
                for range_name, (min_val, max_val) in ranges.items():
                    if min_val <= sum_val <= max_val:
                        range_counts[range_name] += 1
                        break
            
            sum_analysis['range_distribution'] = range_counts
            
            return sum_analysis
            
        except Exception as e:
            logger.error(f"和值分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_span(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """跨度分析"""
        try:
            spans = []
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = [int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()]
                    if len(red_nums) == 6:
                        spans.append(max(red_nums) - min(red_nums))
                
                else:  # 大乐透
                    front_nums = [int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()]
                    if len(front_nums) == 5:
                        spans.append(max(front_nums) - min(front_nums))
            
            if not spans:
                return {'error': '无有效跨度数据'}
            
            span_analysis = {
                'recent_spans': spans[:20],
                'mean_span': np.mean(spans),
                'median_span': np.median(spans),
                'min_span': min(spans),
                'max_span': max(spans),
                'std_span': np.std(spans),
                'span_distribution': dict(Counter(spans))
            }
            
            # 跨度区间分析
            span_ranges = {
                'small': (0, 15),
                'medium': (16, 25),
                'large': (26, 35)
            }
            
            range_counts = {range_name: 0 for range_name in span_ranges.keys()}
            for span in spans:
                for range_name, (min_val, max_val) in span_ranges.items():
                    if min_val <= span <= max_val:
                        range_counts[range_name] += 1
                        break
            
            span_analysis['range_distribution'] = range_counts
            
            return span_analysis
            
        except Exception as e:
            logger.error(f"跨度分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_consecutive(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """连号分析"""
        try:
            consecutive_counts = []
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = sorted([int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()])
                    if len(red_nums) == 6:
                        consecutive_count = self._count_consecutive(red_nums)
                        consecutive_counts.append(consecutive_count)
                
                else:  # 大乐透
                    front_nums = sorted([int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()])
                    if len(front_nums) == 5:
                        consecutive_count = self._count_consecutive(front_nums)
                        consecutive_counts.append(consecutive_count)
            
            if not consecutive_counts:
                return {'error': '无有效连号数据'}
            
            consecutive_analysis = {
                'consecutive_distribution': dict(Counter(consecutive_counts)),
                'avg_consecutive': np.mean(consecutive_counts),
                'max_consecutive': max(consecutive_counts),
                'no_consecutive_rate': consecutive_counts.count(0) / len(consecutive_counts) * 100
            }
            
            return consecutive_analysis
            
        except Exception as e:
            logger.error(f"连号分析失败: {e}")
            return {'error': str(e)}
    
    def _count_consecutive(self, numbers: List[int]) -> int:
        """计算连号个数"""
        if len(numbers) < 2:
            return 0
        
        consecutive_count = 0
        current_count = 1
        
        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i-1] + 1:
                current_count += 1
            else:
                if current_count >= 2:
                    consecutive_count += current_count
                current_count = 1
        
        if current_count >= 2:
            consecutive_count += current_count
        
        return consecutive_count
    
    def _analyze_repeat(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """重号分析（与上期重复的号码）"""
        try:
            repeat_counts = []
            
            prev_numbers = None
            
            for record in history_data:
                numbers = record.get('numbers', {})
                if isinstance(numbers, str):
                    numbers = json.loads(numbers)
                
                if lottery_type == "双色球":
                    red_nums = set([int(num.strip()) for num in numbers.get('red', '').split(',') if num.strip().isdigit()])
                    
                    if prev_numbers is not None and len(red_nums) == 6:
                        repeat_count = len(red_nums & prev_numbers)
                        repeat_counts.append(repeat_count)
                    
                    if len(red_nums) == 6:
                        prev_numbers = red_nums
                
                else:  # 大乐透
                    front_nums = set([int(num.strip()) for num in numbers.get('front', '').split(',') if num.strip().isdigit()])
                    
                    if prev_numbers is not None and len(front_nums) == 5:
                        repeat_count = len(front_nums & prev_numbers)
                        repeat_counts.append(repeat_count)
                    
                    if len(front_nums) == 5:
                        prev_numbers = front_nums
            
            if not repeat_counts:
                return {'error': '无有效重号数据'}
            
            repeat_analysis = {
                'repeat_distribution': dict(Counter(repeat_counts)),
                'avg_repeat': np.mean(repeat_counts),
                'max_repeat': max(repeat_counts),
                'no_repeat_rate': repeat_counts.count(0) / len(repeat_counts) * 100
            }
            
            return repeat_analysis
            
        except Exception as e:
            logger.error(f"重号分析失败: {e}")
            return {'error': str(e)}
    
    def _calculate_comprehensive_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """计算综合置信度"""
        try:
            confidence_scores = []
            
            # 根据各项分析的完整性和质量计算置信度
            for analysis_type, weight in self.analysis_weights.items():
                if analysis_type in analysis_results:
                    result = analysis_results[analysis_type]
                    if 'error' not in result:
                        # 根据分析类型计算具体置信度
                        if analysis_type == 'frequency':
                            quality = result.get('analysis_quality', 'medium')
                            score = 0.9 if quality == 'high' else 0.7
                        elif analysis_type == 'statistical':
                            score = 0.8 if 'mean' in result else 0.5
                        else:
                            score = 0.7  # 默认分数
                        
                        confidence_scores.append(score * weight)
            
            overall_confidence = sum(confidence_scores) / sum(self.analysis_weights.values()) * 100
            return round(overall_confidence, 2)
            
        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return 50.0  # 默认置信度
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], lottery_type: str) -> List[str]:
        """生成分析建议"""
        recommendations = []
        
        try:
            # 基于频率分析的建议
            if 'frequency' in analysis_results and 'error' not in analysis_results['frequency']:
                freq_result = analysis_results['frequency']
                if lottery_type == "双色球":
                    hot_red = freq_result.get('hot_red', [])
                    cold_red = freq_result.get('cold_red', [])
                    if hot_red:
                        recommendations.append(f"考虑关注热门红球号码: {[num for num, _ in hot_red[:5]]}")
                    if cold_red:
                        recommendations.append(f"关注冷门红球号码的补出机会: {[num for num, _ in cold_red[:3]]}")
            
            # 基于遗漏分析的建议
            if 'missing' in analysis_results and 'error' not in analysis_results['missing']:
                missing_result = analysis_results['missing']
                high_missing = missing_result.get('high_missing_red', [])
                if high_missing:
                    recommendations.append(f"重点关注高遗漏号码: {high_missing[:3]}")
            
            # 基于形态分析的建议
            if 'pattern' in analysis_results and 'error' not in analysis_results['pattern']:
                pattern_result = analysis_results['pattern']
                common_odd_even = pattern_result.get('most_common_odd_even')
                if common_odd_even:
                    recommendations.append(f"建议采用常见奇偶形态: {common_odd_even[0]}")
            
            # 基于和值分析的建议
            if 'sum_analysis' in analysis_results and 'error' not in analysis_results['sum_analysis']:
                sum_result = analysis_results['sum_analysis']
                mean_sum = sum_result.get('mean_sum', 0)
                if mean_sum > 0:
                    recommendations.append(f"建议选择和值在 {int(mean_sum-10)} 到 {int(mean_sum+10)} 之间的号码组合")
            
            # 通用建议
            recommendations.extend([
                "建议结合多种分析方法进行选号",
                "注意号码分布的均匀性，避免过于集中",
                "定期关注号码走势变化，及时调整策略",
                "理性投注，切勿沉迷"
            ])
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            recommendations.append("分析建议生成异常，请参考基础分析结果")
        
        return recommendations
    
    def _calculate_comprehensive_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """计算综合评分"""
        scores = {
            'regularity_score': 50,    # 规律性评分
            'randomness_score': 50,    # 随机性评分
            'hotness_score': 50,       # 热度评分
            'stability_score': 50,     # 稳定性评分
            'overall_score': 50        # 综合评分
        }
        
        try:
            # 基于各项分析结果计算评分
            if 'frequency' in analysis_results and 'error' not in analysis_results['frequency']:
                # 规律性评分基于频率分布的均匀程度
                freq_result = analysis_results['frequency']
                if 'red_frequency' in freq_result:
                    freqs = list(freq_result['red_frequency'].values())
                    if freqs:
                        cv = np.std(freqs) / np.mean(freqs) if np.mean(freqs) > 0 else 1
                        scores['regularity_score'] = max(0, min(100, 100 - cv * 50))
            
            if 'statistical' in analysis_results and 'error' not in analysis_results['statistical']:
                # 随机性评分基于统计特征
                stat_result = analysis_results['statistical']
                if stat_result.get('is_normal', False):
                    scores['randomness_score'] = 80
                else:
                    scores['randomness_score'] = 60
            
            # 计算综合评分
            scores['overall_score'] = np.mean(list(scores.values()))
            
        except Exception as e:
            logger.error(f"计算评分失败: {e}")
        
        return {k: round(v, 1) for k, v in scores.items()}
    
    def generate_analysis_report(self, lottery_type: str, period_range: str = "最近100期") -> str:
        """
        生成分析报告
        
        Args:
            lottery_type: 彩票类型
            period_range: 期数范围
            
        Returns:
            分析报告文本
        """
        try:
            # 获取综合分析结果
            analysis_result = self.comprehensive_analysis(lottery_type, period_range)
            
            if 'error' in analysis_result:
                return f"❌ 分析报告生成失败: {analysis_result['error']}"
            
            # 生成报告
            report = f"""
📊 {lottery_type}数据分析报告
{'='*60}

🎯 分析概况:
• 彩票类型: {lottery_type}
• 分析期数: {period_range}
• 数据样本: {analysis_result.get('data_count', 0)}期
• 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• 置信度评分: {analysis_result.get('confidence_score', 0):.1f}/100

📈 频率分析:
"""
            
            # 添加频率分析结果
            freq_result = analysis_result.get('frequency', {})
            if 'error' not in freq_result:
                if lottery_type == "双色球":
                    hot_red = freq_result.get('hot_red', [])
                    hot_blue = freq_result.get('hot_blue', [])
                    report += f"• 热门红球: {[f'{num}({freq}次)' for num, freq in hot_red[:5]]}\n"
                    report += f"• 热门蓝球: {[f'{num}({freq}次)' for num, freq in hot_blue[:3]]}\n"
                else:
                    hot_front = freq_result.get('hot_front', [])
                    hot_back = freq_result.get('hot_back', [])
                    report += f"• 热门前区: {[f'{num}({freq}次)' for num, freq in hot_front[:5]]}\n"
                    report += f"• 热门后区: {[f'{num}({freq}次)' for num, freq in hot_back[:3]]}\n"
            
            # 添加冷热号分析
            hot_cold_result = analysis_result.get('hot_cold', {})
            if 'error' not in hot_cold_result:
                report += f"\n🌡️ 冷热号分析:\n"
                hot_nums = hot_cold_result.get('hot', {})
                cold_nums = hot_cold_result.get('cold', {})
                if lottery_type == "双色球":
                    report += f"• 热号红球: {hot_nums.get('red', [])}\n"
                    report += f"• 冷号红球: {cold_nums.get('red', [])}\n"
                else:
                    report += f"• 热号前区: {hot_nums.get('front', [])}\n"
                    report += f"• 冷号前区: {cold_nums.get('front', [])}\n"
            
            # 添加和值分析
            sum_result = analysis_result.get('sum_analysis', {})
            if 'error' not in sum_result:
                report += f"\n➕ 和值分析:\n"
                report += f"• 平均和值: {sum_result.get('mean_sum', 0):.1f}\n"
                report += f"• 和值范围: {sum_result.get('min_sum', 0)} - {sum_result.get('max_sum', 0)}\n"
                report += f"• 建议和值区间: {sum_result.get('mean_sum', 0)-10:.0f} - {sum_result.get('mean_sum', 0)+10:.0f}\n"
            
            # 添加遗漏分析
            missing_result = analysis_result.get('missing', {})
            if 'error' not in missing_result:
                report += f"\n⏰ 遗漏分析:\n"
                if lottery_type == "双色球":
                    high_missing = missing_result.get('high_missing_red', [])
                    max_missing = missing_result.get('max_red_missing', 0)
                    report += f"• 最大遗漏: {max_missing}期\n"
                    report += f"• 高遗漏号码: {high_missing}\n"
            
            # 添加综合评分
            scores = analysis_result.get('scores', {})
            if scores:
                report += f"\n🎯 综合评分:\n"
                report += f"• 规律性: {scores.get('regularity_score', 0):.1f}/100\n"
                report += f"• 随机性: {scores.get('randomness_score', 0):.1f}/100\n"
                report += f"• 热度指数: {scores.get('hotness_score', 0):.1f}/100\n"
                report += f"• 稳定性: {scores.get('stability_score', 0):.1f}/100\n"
                report += f"• 综合评分: {scores.get('overall_score', 0):.1f}/100\n"
            
            # 添加建议
            recommendations = analysis_result.get('recommendations', [])
            if recommendations:
                report += f"\n💡 分析建议:\n"
                for i, rec in enumerate(recommendations, 1):
                    report += f"{i}. {rec}\n"
            
            # 添加免责声明
            report += f"\n⚠️ 重要提醒:\n"
            report += "• 本分析仅供参考，不构成投注建议\n"
            report += "• 彩票具有随机性，历史数据不能预测未来结果\n"
            report += "• 请理性投注，量力而行\n"
            report += "• 数据分析有助于了解历史规律，但不保证准确性\n"
            
            return report
            
        except Exception as e:
            logger.error(f"生成分析报告失败: {e}")
            return f"❌ 分析报告生成失败: {str(e)}"
    
    def _generate_cache_key(self, lottery_type: str, period_range: str, analysis_type: str) -> str:
        """生成缓存键"""
        key_str = f"{lottery_type}_{period_range}_{analysis_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_memory_cache(self, cache_key: str, data: Any):
        """更新内存缓存"""
        try:
            # 检查缓存大小，如果超过限制则清理旧缓存
            if len(self.memory_cache) >= self.cache_max_size:
                # 删除最旧的缓存项
                oldest_key = min(self.memory_cache.keys(), 
                               key=lambda k: self.memory_cache[k][1])
                del self.memory_cache[oldest_key]
            
            self.memory_cache[cache_key] = (data, datetime.now())
            
        except Exception as e:
            logger.warning(f"更新内存缓存失败: {e}")
    
    def _parallel_analysis(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """并行执行分析任务"""
        analysis_results = {}
        
        # 定义分析任务
        analysis_tasks = [
            ('frequency', self._analyze_frequency),
            ('trend', self._analyze_trend),
            ('pattern', self._analyze_pattern),
            ('missing', self._analyze_missing),
            ('correlation', self._analyze_correlation),
            ('statistical', self._analyze_statistical),
            ('hot_cold', self._analyze_hot_cold),
            ('sum_analysis', self._analyze_sum),
            ('span', self._analyze_span),
            ('consecutive', self._analyze_consecutive),
            ('repeat', self._analyze_repeat),
            ('distribution', self._analyze_distribution_advanced),
            ('volatility', self._analyze_volatility),
            ('cycle', self._analyze_cycle_patterns)
        ]
        
        # 智能选择执行方式：数据量小时使用顺序执行，数据量大时使用并行执行
        data_size = len(history_data)
        use_parallel = data_size > 200 and len(analysis_tasks) > 8
        
        if use_parallel:
            try:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(analysis_tasks))) as executor:
                    # 提交所有任务
                    future_to_analysis = {
                        executor.submit(analysis_func, history_data, lottery_type): analysis_name
                        for analysis_name, analysis_func in analysis_tasks
                    }
                    
                    # 收集结果
                    for future in as_completed(future_to_analysis):
                        analysis_name = future_to_analysis[future]
                        try:
                            result = future.result(timeout=30)  # 30秒超时
                            analysis_results[analysis_name] = result
                        except Exception as e:
                            logger.error(f"并行执行{analysis_name}分析失败: {e}")
                            analysis_results[analysis_name] = {'error': str(e)}
            
            except Exception as e:
                logger.error(f"并行分析执行失败: {e}")
                use_parallel = False  # 降级到顺序执行
        
        if not use_parallel:
            # 顺序执行（小数据集或并行失败时）
            for analysis_name, analysis_func in analysis_tasks:
                try:
                    analysis_results[analysis_name] = analysis_func(history_data, lottery_type)
                except Exception as func_e:
                    logger.error(f"{analysis_name}分析失败: {func_e}")
                    analysis_results[analysis_name] = {'error': str(func_e)}
        
        return analysis_results
    
    def _analyze_distribution_advanced(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """高级分布分析"""
        try:
            logger.info("开始进行高级分布分析...")
            
            all_numbers = []
            position_analysis = {}
            
            for data in history_data:
                red_balls, blue_balls = self._extract_numbers_from_data(data)
                if red_balls and blue_balls:
                    all_numbers.extend(red_balls)
                    
                    # 按位置分析
                    sorted_reds = sorted(red_balls)
                    for i, num in enumerate(sorted_reds):
                        if i not in position_analysis:
                            position_analysis[i] = []
                        position_analysis[i].append(num)
            
            if not all_numbers:
                return {'error': '无有效数据进行高级分布分析'}
            
            # 分布特征分析
            distribution_features = {
                'skewness': float(stats.skew(all_numbers)),  # 偏度
                'kurtosis': float(stats.kurtosis(all_numbers)),  # 峰度
                'entropy': self._calculate_entropy(all_numbers),  # 信息熵
                'uniformity_test': self._test_uniformity(all_numbers),  # 均匀性检验
                'position_preferences': {}  # 位置偏好
            }
            
            # 位置偏好分析
            for position, nums in position_analysis.items():
                if nums:
                    distribution_features['position_preferences'][f'position_{position+1}'] = {
                        'mean': float(np.mean(nums)),
                        'std': float(np.std(nums)),
                        'most_common': Counter(nums).most_common(5)
                    }
            
            # 区域分布分析
            if lottery_type == "双色球":
                range_size = 33
            else:
                range_size = 35
            
            zone_size = range_size // 3
            zones = {
                'low': (1, zone_size),
                'middle': (zone_size + 1, zone_size * 2),
                'high': (zone_size * 2 + 1, range_size)
            }
            
            zone_distribution = {}
            for zone_name, (start, end) in zones.items():
                zone_count = sum(1 for num in all_numbers if start <= num <= end)
                zone_distribution[zone_name] = {
                    'count': zone_count,
                    'percentage': round((zone_count / len(all_numbers)) * 100, 2),
                    'range': f"{start}-{end}"
                }
            
            distribution_features['zone_distribution'] = zone_distribution
            
            return {
                'analysis_type': 'distribution_advanced',
                'distribution_features': distribution_features,
                'sample_size': len(all_numbers),
                'analysis_quality': 'high' if len(history_data) >= 100 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"高级分布分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_volatility(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """波动性分析"""
        try:
            logger.info("开始进行波动性分析...")
            
            sums = []
            means = []
            ranges = []
            
            for data in history_data:
                red_balls, blue_balls = self._extract_numbers_from_data(data)
                if red_balls:
                    sums.append(sum(red_balls))
                    means.append(np.mean(red_balls))
                    ranges.append(max(red_balls) - min(red_balls))
            
            if len(sums) < 10:
                return {'error': '数据不足，无法进行波动性分析'}
            
            # 计算各种波动性指标
            volatility_metrics = {
                'sum_volatility': {
                    'std': float(np.std(sums)),
                    'cv': float(np.std(sums) / np.mean(sums)) if np.mean(sums) > 0 else 0,
                    'rolling_std': self._calculate_rolling_std(sums, window=10)
                },
                'mean_volatility': {
                    'std': float(np.std(means)),
                    'cv': float(np.std(means) / np.mean(means)) if np.mean(means) > 0 else 0,
                    'rolling_std': self._calculate_rolling_std(means, window=10)
                },
                'range_volatility': {
                    'std': float(np.std(ranges)),
                    'cv': float(np.std(ranges) / np.mean(ranges)) if np.mean(ranges) > 0 else 0,
                    'rolling_std': self._calculate_rolling_std(ranges, window=10)
                }
            }
            
            # 波动性等级评估
            sum_cv = volatility_metrics['sum_volatility']['cv']
            if sum_cv < 0.1:
                volatility_level = 'low'
            elif sum_cv < 0.2:
                volatility_level = 'medium'
            else:
                volatility_level = 'high'
            
            return {
                'analysis_type': 'volatility',
                'volatility_metrics': volatility_metrics,
                'volatility_level': volatility_level,
                'stability_score': round(1 / (1 + sum_cv), 3),
                'sample_size': len(sums)
            }
            
        except Exception as e:
            logger.error(f"波动性分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_cycle_patterns(self, history_data: List[Dict], lottery_type: str) -> Dict[str, Any]:
        """周期模式分析"""
        try:
            logger.info("开始进行周期模式分析...")
            
            if len(history_data) < 30:
                return {'error': '数据不足，无法进行周期模式分析'}
            
            # 按开奖日期分析周期
            date_patterns = {}
            number_cycles = defaultdict(list)
            
            for i, data in enumerate(history_data):
                red_balls, blue_balls = self._extract_numbers_from_data(data)
                if red_balls:
                    # 记录每个号码的出现位置
                    for num in red_balls:
                        number_cycles[num].append(i)
                
                # 日期模式分析
                date_str = data.get('date', '')
                if date_str:
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        weekday = date_obj.strftime('%A')
                        month = date_obj.month
                        
                        if weekday not in date_patterns:
                            date_patterns[weekday] = {'count': 0, 'numbers': []}
                        date_patterns[weekday]['count'] += 1
                        date_patterns[weekday]['numbers'].extend(red_balls)
                    except:
                        continue
            
            # 分析号码出现周期
            cycle_analysis = {}
            for num, positions in number_cycles.items():
                if len(positions) >= 3:
                    intervals = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                    cycle_analysis[num] = {
                        'avg_interval': round(np.mean(intervals), 2),
                        'std_interval': round(np.std(intervals), 2),
                        'last_appearance': positions[-1],
                        'predicted_next': positions[-1] + round(np.mean(intervals))
                    }
            
            # 周期强度评估
            cycle_strength = self._evaluate_cycle_strength(number_cycles, history_data)
            
            return {
                'analysis_type': 'cycle_patterns',
                'date_patterns': date_patterns,
                'number_cycles': dict(list(cycle_analysis.items())[:20]),  # 前20个号码
                'cycle_strength': cycle_strength,
                'total_periods': len(history_data)
            }
            
        except Exception as e:
            logger.error(f"周期模式分析失败: {e}")
            return {'error': str(e)}
    
    def _extract_numbers_from_data(self, data: Dict) -> Tuple[List[int], List[int]]:
        """从数据中提取号码（优化版本）"""
        try:
            numbers = data.get('numbers', {})
            if isinstance(numbers, str):
                numbers = json.loads(numbers)
            
            if isinstance(numbers, dict):
                if 'red' in numbers and 'blue' in numbers:
                    # 双色球格式
                    red_str = numbers.get('red', '')
                    blue_str = numbers.get('blue', '')
                    
                    red_balls = [int(x.strip()) for x in red_str.split(',') if x.strip().isdigit()]
                    blue_balls = [int(blue_str.strip())] if blue_str.strip().isdigit() else []
                    
                elif 'red_balls' in numbers and 'blue_balls' in numbers:
                    # 标准格式
                    red_balls = numbers.get('red_balls', [])
                    blue_balls = numbers.get('blue_balls', [])
                    
                elif 'front' in numbers and 'back' in numbers:
                    # 大乐透格式
                    front_str = numbers.get('front', '')
                    back_str = numbers.get('back', '')
                    
                    red_balls = [int(x.strip()) for x in front_str.split(',') if x.strip().isdigit()]
                    blue_balls = [int(x.strip()) for x in back_str.split(',') if x.strip().isdigit()]
                    
                else:
                    return [], []
                
                return red_balls, blue_balls
            
            return [], []
            
        except Exception as e:
            logger.debug(f"提取号码失败: {e}")
            return [], []
    
    def _calculate_entropy(self, numbers: List[int]) -> float:
        """计算信息熵"""
        try:
            counter = Counter(numbers)
            total = len(numbers)
            entropy = 0
            
            for count in counter.values():
                probability = count / total
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            return round(entropy, 4)
        except:
            return 0.0
    
    def _test_uniformity(self, numbers: List[int]) -> Dict[str, Any]:
        """均匀性检验"""
        try:
            # 卡方检验
            counter = Counter(numbers)
            observed = list(counter.values())
            expected = [len(numbers) / len(set(numbers))] * len(set(numbers))
            
            chi2_stat, p_value = stats.chisquare(observed, expected)
            
            return {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'is_uniform': p_value > 0.05,
                'confidence_level': '95%'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_rolling_std(self, values: List[float], window: int = 10) -> List[float]:
        """计算滚动标准差"""
        try:
            if len(values) < window:
                return []
            
            rolling_stds = []
            for i in range(window - 1, len(values)):
                window_values = values[i - window + 1:i + 1]
                rolling_stds.append(float(np.std(window_values)))
            
            return rolling_stds
        except:
            return []
    
    def _evaluate_cycle_strength(self, number_cycles: Dict, history_data: List[Dict]) -> Dict[str, Any]:
        """评估周期强度"""
        try:
            cycle_scores = []
            
            for num, positions in number_cycles.items():
                if len(positions) >= 3:
                    intervals = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
                    
                    # 周期性越强，变异系数越小
                    cycle_score = 1 / (1 + cv) if cv != float('inf') else 0
                    cycle_scores.append(cycle_score)
            
            if cycle_scores:
                avg_cycle_strength = np.mean(cycle_scores)
                if avg_cycle_strength > 0.7:
                    strength_level = 'strong'
                elif avg_cycle_strength > 0.4:
                    strength_level = 'moderate'
                else:
                    strength_level = 'weak'
            else:
                avg_cycle_strength = 0
                strength_level = 'none'
            
            return {
                'average_strength': round(avg_cycle_strength, 3),
                'strength_level': strength_level,
                'numbers_with_cycles': len(cycle_scores)
            }
        except:
            return {'average_strength': 0, 'strength_level': 'unknown', 'numbers_with_cycles': 0}
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, (int, float, str, type(None))):
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)


# 使用示例
if __name__ == "__main__":
    # 创建分析实例
    analyzer = LotteryAnalysis()
    
    # 执行综合分析
    result = analyzer.comprehensive_analysis("双色球", "最近100期")
    print("分析结果:", json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    # 生成分析报告
    report = analyzer.generate_analysis_report("双色球", "最近100期")
    print("\n分析报告:")
    print(report)
