"""
基础预测器 - 所有预测模型的基类
增强版本：支持特征缓存、性能监控和自适应优化
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """
    预测器基类，定义所有预测模型的通用接口
    """
    
    def __init__(self, lottery_type: str, model_name: str):
        """
        初始化预测器
        
        Args:
            lottery_type: 彩票类型 ('双色球' 或 '大乐透')
            model_name: 模型名称
        """
        self.lottery_type = lottery_type
        self.model_name = model_name
        self.is_trained = False
        self.feature_columns = []
        self.target_columns = []
        
        # 性能优化配置
        self.feature_cache = {}  # 特征缓存
        self.cache_max_size = 50
        self.performance_metrics = {
            'training_time': 0,
            'prediction_time': 0,
            'accuracy_history': [],
            'last_updated': None
        }
        
        # 自适应配置
        self.adaptive_threshold = 0.1  # 自适应阈值
        self.min_confidence = 0.3
        self.feature_importance_cache = {}
        
        # 设置彩票规则
        self._setup_lottery_rules()
        
    def _setup_lottery_rules(self):
        """设置彩票规则"""
        if self.lottery_type == '双色球':
            self.red_range = (1, 33)
            self.blue_range = (1, 16)
            self.red_count = 6
            self.blue_count = 1
        elif self.lottery_type == '大乐透':
            self.red_range = (1, 35)
            self.blue_range = (1, 12)
            self.red_count = 5
            self.blue_count = 2
        else:
            raise ValueError(f"不支持的彩票类型: {self.lottery_type}")
    
    @abstractmethod
    def prepare_features(self, history_data: List[Dict]) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            特征DataFrame
        """
        pass
    
    @abstractmethod
    def train(self, history_data: List[Dict]) -> bool:
        """
        训练模型
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            训练是否成功
        """
        pass
    
    @abstractmethod
    def predict(self, recent_data: Optional[List[Dict]] = None) -> Dict:
        """
        进行预测
        
        Args:
            recent_data: 最近的历史数据（可选）
            
        Returns:
            预测结果字典
        """
        pass
    
    def extract_numbers_from_data(self, data: Dict) -> Tuple[List[int], List[int]]:
        """
        从开奖数据中提取号码
        
        Args:
            data: 开奖数据字典
            
        Returns:
            (红球号码列表, 蓝球号码列表)
        """
        try:
            if 'numbers' in data:
                numbers = data['numbers']
            elif 'numbers_data' in data:
                numbers = data['numbers_data']
            else:
                raise ValueError("数据中未找到开奖号码")
            
            if self.lottery_type == '双色球':
                red_balls = numbers.get('red_balls', [])
                blue_balls = numbers.get('blue_balls', [])
            else:  # 大乐透
                red_balls = numbers.get('front_area', [])
                blue_balls = numbers.get('back_area', [])
            
            return red_balls, blue_balls
            
        except Exception as e:
            logger.error(f"提取号码失败: {e}")
            return [], []
    
    def validate_prediction(self, red_balls: List[int], blue_balls: List[int]) -> bool:
        """
        验证预测结果是否符合彩票规则
        
        Args:
            red_balls: 红球号码
            blue_balls: 蓝球号码
            
        Returns:
            是否有效
        """
        # 检查红球
        if len(red_balls) != self.red_count:
            return False
        if not all(self.red_range[0] <= num <= self.red_range[1] for num in red_balls):
            return False
        if len(set(red_balls)) != len(red_balls):  # 检查重复
            return False
        
        # 检查蓝球
        if len(blue_balls) != self.blue_count:
            return False
        if not all(self.blue_range[0] <= num <= self.blue_range[1] for num in blue_balls):
            return False
        if len(set(blue_balls)) != len(blue_balls):  # 检查重复
            return False
        
        return True
    
    def calculate_features(self, history_data: List[Dict]) -> pd.DataFrame:
        """
        计算基础特征
        
        Args:
            history_data: 历史数据
            
        Returns:
            特征DataFrame
        """
        features = []
        
        for i, data in enumerate(history_data):
            red_balls, blue_balls = self.extract_numbers_from_data(data)
            
            if not red_balls or not blue_balls:
                continue
            
            feature_dict = {
                'period': data.get('period', ''),
                'date': data.get('date', ''),
            }
            
            # 基础号码特征
            for j, num in enumerate(red_balls):
                feature_dict[f'red_{j+1}'] = num
            for j, num in enumerate(blue_balls):
                feature_dict[f'blue_{j+1}'] = num
            
            # 统计特征
            feature_dict.update(self._calculate_statistical_features(red_balls, blue_balls))
            
            # 历史频率特征
            if i > 0:  # 需要历史数据计算频率
                feature_dict.update(self._calculate_frequency_features(
                    red_balls, blue_balls, history_data[:i]
                ))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_statistical_features(self, red_balls: List[int], blue_balls: List[int]) -> Dict:
        """计算统计特征"""
        features = {}
        
        # 红球统计
        features['red_sum'] = sum(red_balls)
        features['red_mean'] = np.mean(red_balls)
        features['red_std'] = np.std(red_balls)
        features['red_range'] = max(red_balls) - min(red_balls)
        features['red_odd_count'] = sum(1 for num in red_balls if num % 2 == 1)
        features['red_even_count'] = sum(1 for num in red_balls if num % 2 == 0)
        
        # 大小比例 (以中位数为界)
        red_mid = (self.red_range[0] + self.red_range[1]) // 2
        features['red_big_count'] = sum(1 for num in red_balls if num > red_mid)
        features['red_small_count'] = sum(1 for num in red_balls if num <= red_mid)
        
        # 连号统计
        sorted_reds = sorted(red_balls)
        consecutive_count = 0
        for i in range(len(sorted_reds) - 1):
            if sorted_reds[i+1] - sorted_reds[i] == 1:
                consecutive_count += 1
        features['red_consecutive_count'] = consecutive_count
        
        # 蓝球统计
        features['blue_sum'] = sum(blue_balls)
        features['blue_mean'] = np.mean(blue_balls)
        if len(blue_balls) > 1:
            features['blue_std'] = np.std(blue_balls)
            features['blue_range'] = max(blue_balls) - min(blue_balls)
        else:
            features['blue_std'] = 0
            features['blue_range'] = 0
        
        return features
    
    def _calculate_frequency_features(self, red_balls: List[int], blue_balls: List[int], 
                                    history_data: List[Dict]) -> Dict:
        """计算频率特征"""
        features = {}
        
        # 统计历史出现频率
        red_freq = {}
        blue_freq = {}
        
        for data in history_data:
            hist_red, hist_blue = self.extract_numbers_from_data(data)
            for num in hist_red:
                red_freq[num] = red_freq.get(num, 0) + 1
            for num in hist_blue:
                blue_freq[num] = blue_freq.get(num, 0) + 1
        
        # 当前号码的历史频率
        features['red_avg_freq'] = np.mean([red_freq.get(num, 0) for num in red_balls])
        features['blue_avg_freq'] = np.mean([blue_freq.get(num, 0) for num in blue_balls])
        
        # 冷热号特征
        total_periods = len(history_data)
        if total_periods > 0:
            red_avg_appear = total_periods * self.red_count / (self.red_range[1] - self.red_range[0] + 1)
            blue_avg_appear = total_periods * self.blue_count / (self.blue_range[1] - self.blue_range[0] + 1)
            
            features['red_hot_count'] = sum(1 for num in red_balls 
                                          if red_freq.get(num, 0) > red_avg_appear)
            features['red_cold_count'] = sum(1 for num in red_balls 
                                           if red_freq.get(num, 0) < red_avg_appear)
            features['blue_hot_count'] = sum(1 for num in blue_balls 
                                           if blue_freq.get(num, 0) > blue_avg_appear)
            features['blue_cold_count'] = sum(1 for num in blue_balls 
                                            if blue_freq.get(num, 0) < blue_avg_appear)
        
        return features
    
    def get_cache_key(self, data: Any) -> str:
        """生成数据的缓存键"""
        try:
            data_str = str(sorted(data)) if isinstance(data, (list, tuple)) else str(data)
            return hashlib.md5(data_str.encode()).hexdigest()[:16]
        except:
            return str(hash(str(data)))[:16]
    
    def cache_features(self, cache_key: str, features: pd.DataFrame):
        """缓存特征数据"""
        try:
            if len(self.feature_cache) >= self.cache_max_size:
                # 删除最旧的缓存
                oldest_key = min(self.feature_cache.keys(), 
                               key=lambda k: self.feature_cache[k]['timestamp'])
                del self.feature_cache[oldest_key]
            
            self.feature_cache[cache_key] = {
                'features': features.copy(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"缓存特征失败: {e}")
    
    def get_cached_features(self, cache_key: str, max_age: int = 3600) -> Optional[pd.DataFrame]:
        """获取缓存的特征数据"""
        try:
            if cache_key in self.feature_cache:
                cached_data = self.feature_cache[cache_key]
                if time.time() - cached_data['timestamp'] <= max_age:
                    return cached_data['features'].copy()
                else:
                    # 过期的缓存，删除
                    del self.feature_cache[cache_key]
        except Exception as e:
            logger.warning(f"获取缓存特征失败: {e}")
        
        return None
    
    def update_performance_metrics(self, training_time: float = None, 
                                 prediction_time: float = None, 
                                 accuracy: float = None):
        """更新性能指标"""
        try:
            if training_time is not None:
                self.performance_metrics['training_time'] = training_time
            
            if prediction_time is not None:
                self.performance_metrics['prediction_time'] = prediction_time
            
            if accuracy is not None:
                self.performance_metrics['accuracy_history'].append({
                    'accuracy': accuracy,
                    'timestamp': datetime.now().isoformat()
                })
                # 保持历史记录不超过100条
                if len(self.performance_metrics['accuracy_history']) > 100:
                    self.performance_metrics['accuracy_history'] = \
                        self.performance_metrics['accuracy_history'][-100:]
            
            self.performance_metrics['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.warning(f"更新性能指标失败: {e}")
    
    def get_adaptive_confidence(self, base_confidence: float) -> float:
        """基于历史性能自适应调整置信度"""
        try:
            if not self.performance_metrics['accuracy_history']:
                return max(self.min_confidence, base_confidence)
            
            # 计算最近10次预测的平均准确率
            recent_accuracies = [
                item['accuracy'] for item in 
                self.performance_metrics['accuracy_history'][-10:]
            ]
            
            if recent_accuracies:
                avg_accuracy = np.mean(recent_accuracies)
                trend = np.mean(np.diff(recent_accuracies)) if len(recent_accuracies) > 1 else 0
                
                # 根据趋势调整置信度
                adjustment = trend * 0.5  # 趋势权重
                adjusted_confidence = base_confidence * avg_accuracy + adjustment
                
                return max(self.min_confidence, min(1.0, adjusted_confidence))
            
            return max(self.min_confidence, base_confidence)
            
        except Exception as e:
            logger.warning(f"自适应置信度计算失败: {e}")
            return max(self.min_confidence, base_confidence)
    
    def optimize_features(self, features: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """特征优化和选择"""
        try:
            if features.empty:
                return features
            
            # 移除高相关性特征
            optimized_features = self._remove_highly_correlated_features(features)
            
            # 特征标准化
            optimized_features = self._normalize_features(optimized_features)
            
            # 如果有目标变量，进行特征选择
            if target is not None and len(target) == len(optimized_features):
                optimized_features = self._select_important_features(optimized_features, target)
            
            return optimized_features
            
        except Exception as e:
            logger.warning(f"特征优化失败: {e}")
            return features
    
    def _remove_highly_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """移除高相关性特征"""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return features
            
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 找到高相关性的特征对
            high_corr_pairs = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > threshold)
            ]
            
            # 移除高相关性特征
            features_to_keep = [col for col in features.columns if col not in high_corr_pairs]
            
            if features_to_keep:
                return features[features_to_keep]
            else:
                return features
                
        except Exception as e:
            logger.warning(f"移除高相关性特征失败: {e}")
            return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty:
                return features
            
            # 使用Z-score标准化
            normalized = features.copy()
            for col in numeric_features.columns:
                if features[col].std() > 0:  # 避免除零错误
                    normalized[col] = (features[col] - features[col].mean()) / features[col].std()
            
            return normalized
            
        except Exception as e:
            logger.warning(f"特征标准化失败: {e}")
            return features
    
    def _select_important_features(self, features: pd.DataFrame, target: pd.Series, top_k: int = 20) -> pd.DataFrame:
        """选择重要特征"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            
            numeric_features = features.select_dtypes(include=[np.number])
            if numeric_features.empty or len(numeric_features.columns) <= top_k:
                return features
            
            selector = SelectKBest(score_func=f_regression, k=min(top_k, len(numeric_features.columns)))
            selector.fit(numeric_features, target)
            
            selected_features = numeric_features.columns[selector.get_support()].tolist()
            non_numeric_features = [col for col in features.columns if col not in numeric_features.columns]
            
            all_selected = selected_features + non_numeric_features
            return features[all_selected]
            
        except Exception as e:
            logger.warning(f"特征选择失败: {e}")
            return features
    
    def parallel_predict(self, data_batches: List[List[Dict]]) -> List[Dict]:
        """并行预测多个数据批次"""
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(data_batches))) as executor:
                futures = [executor.submit(self.predict, batch) for batch in data_batches]
                results = [future.result() for future in futures]
            return results
        except Exception as e:
            logger.error(f"并行预测失败: {e}")
            return [self.predict(batch) for batch in data_batches]
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'lottery_type': self.lottery_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'performance_metrics': self.performance_metrics,
            'cache_size': len(self.feature_cache),
            'created_at': datetime.now().isoformat()
        }
