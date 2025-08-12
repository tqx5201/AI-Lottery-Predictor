"""
高级数据分析模块 - 深度分析彩票历史数据
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from collections import Counter
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedAnalysis:
    """
    高级数据分析器
    提供深度数据挖掘和模式识别功能
    """
    
    def __init__(self, lottery_type: str):
        """
        初始化高级分析器
        
        Args:
            lottery_type: 彩票类型 ('双色球' 或 '大乐透')
        """
        self.lottery_type = lottery_type
        
        # 设置彩票规则
        if lottery_type == '双色球':
            self.red_range = (1, 33)
            self.blue_range = (1, 16)
            self.red_count = 6
            self.blue_count = 1
        elif lottery_type == '大乐透':
            self.red_range = (1, 35)
            self.blue_range = (1, 12)
            self.red_count = 5
            self.blue_count = 2
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")
    
    def correlation_analysis(self, history_data: List[Dict]) -> Dict:
        """
        关联性分析 - 分析号码之间的关联性
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            关联性分析结果
        """
        try:
            logger.info("开始进行关联性分析...")
            
            # 准备数据
            red_numbers_matrix = []
            blue_numbers_matrix = []
            
            for data in history_data:
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls and blue_balls:
                    # 创建one-hot编码
                    red_encoded = [1 if i in red_balls else 0 
                                 for i in range(self.red_range[0], self.red_range[1] + 1)]
                    blue_encoded = [1 if i in blue_balls else 0 
                                  for i in range(self.blue_range[0], self.blue_range[1] + 1)]
                    
                    red_numbers_matrix.append(red_encoded)
                    blue_numbers_matrix.append(blue_encoded)
            
            if len(red_numbers_matrix) < 10:
                return {'error': '数据不足，无法进行关联性分析'}
            
            red_df = pd.DataFrame(red_numbers_matrix, 
                                columns=[f'红{i}' for i in range(self.red_range[0], self.red_range[1] + 1)])
            blue_df = pd.DataFrame(blue_numbers_matrix,
                                 columns=[f'蓝{i}' for i in range(self.blue_range[0], self.blue_range[1] + 1)])
            
            # 计算相关性矩阵
            red_corr = red_df.corr()
            blue_corr = blue_df.corr()
            
            # 找到高相关性的号码对
            red_high_corr = []
            blue_high_corr = []
            
            threshold = 0.3  # 相关性阈值
            
            for i in range(len(red_corr.columns)):
                for j in range(i + 1, len(red_corr.columns)):
                    corr_value = red_corr.iloc[i, j]
                    if abs(corr_value) > threshold:
                        red_high_corr.append({
                            'number1': i + self.red_range[0],
                            'number2': j + self.red_range[0],
                            'correlation': round(corr_value, 3),
                            'type': 'positive' if corr_value > 0 else 'negative'
                        })
            
            for i in range(len(blue_corr.columns)):
                for j in range(i + 1, len(blue_corr.columns)):
                    corr_value = blue_corr.iloc[i, j]
                    if abs(corr_value) > threshold:
                        blue_high_corr.append({
                            'number1': i + self.blue_range[0],
                            'number2': j + self.blue_range[0],
                            'correlation': round(corr_value, 3),
                            'type': 'positive' if corr_value > 0 else 'negative'
                        })
            
            # 红蓝球之间的关联性
            cross_correlation = []
            for red_idx in range(len(red_df.columns)):
                for blue_idx in range(len(blue_df.columns)):
                    corr_value = red_df.iloc[:, red_idx].corr(blue_df.iloc[:, blue_idx])
                    if abs(corr_value) > threshold:
                        cross_correlation.append({
                            'red_number': red_idx + self.red_range[0],
                            'blue_number': blue_idx + self.blue_range[0],
                            'correlation': round(corr_value, 3),
                            'type': 'positive' if corr_value > 0 else 'negative'
                        })
            
            result = {
                'analysis_type': 'correlation_analysis',
                'lottery_type': self.lottery_type,
                'data_period': len(history_data),
                'red_correlations': sorted(red_high_corr, key=lambda x: abs(x['correlation']), reverse=True),
                'blue_correlations': sorted(blue_high_corr, key=lambda x: abs(x['correlation']), reverse=True),
                'cross_correlations': sorted(cross_correlation, key=lambda x: abs(x['correlation']), reverse=True),
                'summary': {
                    'red_high_corr_pairs': len(red_high_corr),
                    'blue_high_corr_pairs': len(blue_high_corr),
                    'cross_corr_pairs': len(cross_correlation),
                    'threshold': threshold
                },
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.info(f"关联性分析完成: 红球相关对{len(red_high_corr)}个，蓝球相关对{len(blue_high_corr)}个")
            return result
            
        except Exception as e:
            logger.error(f"关联性分析失败: {e}")
            return {'error': str(e)}
    
    def seasonality_detection(self, history_data: List[Dict]) -> Dict:
        """
        季节性检测 - 检测号码出现的季节性模式
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            季节性分析结果
        """
        try:
            logger.info("开始进行季节性检测...")
            
            # 按月份统计号码出现频率
            monthly_stats = {}
            for month in range(1, 13):
                monthly_stats[month] = {
                    'red_frequency': Counter(),
                    'blue_frequency': Counter(),
                    'total_draws': 0
                }
            
            for data in history_data:
                # 解析日期
                date_str = data.get('date', '')
                if not date_str:
                    continue
                
                try:
                    # 尝试多种日期格式
                    if '-' in date_str:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    elif '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
                    else:
                        continue
                    
                    month = date_obj.month
                    red_balls, blue_balls = self._extract_numbers(data)
                    
                    if red_balls and blue_balls:
                        monthly_stats[month]['total_draws'] += 1
                        for num in red_balls:
                            monthly_stats[month]['red_frequency'][num] += 1
                        for num in blue_balls:
                            monthly_stats[month]['blue_frequency'][num] += 1
                
                except ValueError:
                    continue
            
            # 分析季节性模式
            seasonal_patterns = {
                'monthly_red_trends': {},
                'monthly_blue_trends': {},
                'seasonal_summary': {}
            }
            
            # 计算每个月的热门号码
            for month in range(1, 13):
                if monthly_stats[month]['total_draws'] > 0:
                    # 红球月度趋势
                    red_freq = monthly_stats[month]['red_frequency']
                    total_red = sum(red_freq.values())
                    red_normalized = {num: count / total_red for num, count in red_freq.items()}
                    
                    # 蓝球月度趋势
                    blue_freq = monthly_stats[month]['blue_frequency']
                    total_blue = sum(blue_freq.values())
                    blue_normalized = {num: count / total_blue for num, count in blue_freq.items()}
                    
                    seasonal_patterns['monthly_red_trends'][month] = {
                        'top_numbers': sorted(red_normalized.items(), 
                                            key=lambda x: x[1], reverse=True)[:10],
                        'total_draws': monthly_stats[month]['total_draws']
                    }
                    
                    seasonal_patterns['monthly_blue_trends'][month] = {
                        'top_numbers': sorted(blue_normalized.items(), 
                                            key=lambda x: x[1], reverse=True)[:5],
                        'total_draws': monthly_stats[month]['total_draws']
                    }
            
            # 季节分析（春夏秋冬）
            seasons = {
                'spring': [3, 4, 5],    # 春季
                'summer': [6, 7, 8],    # 夏季
                'autumn': [9, 10, 11],  # 秋季
                'winter': [12, 1, 2]    # 冬季
            }
            
            for season_name, months in seasons.items():
                season_red_freq = Counter()
                season_blue_freq = Counter()
                season_total = 0
                
                for month in months:
                    if month in monthly_stats:
                        season_total += monthly_stats[month]['total_draws']
                        for num, count in monthly_stats[month]['red_frequency'].items():
                            season_red_freq[num] += count
                        for num, count in monthly_stats[month]['blue_frequency'].items():
                            season_blue_freq[num] += count
                
                if season_total > 0:
                    seasonal_patterns['seasonal_summary'][season_name] = {
                        'red_hot_numbers': [num for num, _ in season_red_freq.most_common(10)],
                        'blue_hot_numbers': [num for num, _ in season_blue_freq.most_common(5)],
                        'total_draws': season_total
                    }
            
            result = {
                'analysis_type': 'seasonality_detection',
                'lottery_type': self.lottery_type,
                'data_period': len(history_data),
                'monthly_patterns': seasonal_patterns['monthly_red_trends'],
                'seasonal_patterns': seasonal_patterns['seasonal_summary'],
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.info("季节性检测完成")
            return result
            
        except Exception as e:
            logger.error(f"季节性检测失败: {e}")
            return {'error': str(e)}
    
    def anomaly_detection(self, history_data: List[Dict]) -> Dict:
        """
        异常检测 - 检测异常的开奖模式
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            异常检测结果
        """
        try:
            logger.info("开始进行异常检测...")
            
            if len(history_data) < 20:
                return {'error': '数据不足，无法进行异常检测'}
            
            # 提取特征
            features = []
            periods = []
            
            for data in history_data:
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls and blue_balls:
                    feature_vector = self._calculate_number_features(red_balls, blue_balls)
                    features.append(feature_vector)
                    periods.append(data.get('period', ''))
            
            if len(features) < 10:
                return {'error': '有效数据不足'}
            
            features_array = np.array(features)
            
            # Z-score异常检测
            z_scores = np.abs(stats.zscore(features_array, axis=0))
            anomaly_threshold = 2.5
            
            anomalies = []
            for i, period in enumerate(periods):
                max_z_score = np.max(z_scores[i])
                if max_z_score > anomaly_threshold:
                    anomaly_features = []
                    for j, z_score in enumerate(z_scores[i]):
                        if z_score > anomaly_threshold:
                            feature_names = [
                                'red_sum', 'red_mean', 'red_std', 'red_range',
                                'red_odd_count', 'red_consecutive', 'blue_sum', 'blue_range'
                            ]
                            if j < len(feature_names):
                                anomaly_features.append({
                                    'feature': feature_names[j],
                                    'z_score': round(z_score, 3),
                                    'value': features[i][j]
                                })
                    
                    anomalies.append({
                        'period': period,
                        'max_z_score': round(max_z_score, 3),
                        'anomaly_features': anomaly_features,
                        'red_balls': self._extract_numbers(history_data[i])[0],
                        'blue_balls': self._extract_numbers(history_data[i])[1]
                    })
            
            # 聚类异常检测（如果sklearn可用）
            cluster_anomalies = []
            if SKLEARN_AVAILABLE and len(features) >= 20:
                try:
                    # 标准化特征
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features_array)
                    
                    # 使用DBSCAN检测异常
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    cluster_labels = dbscan.fit_predict(features_scaled)
                    
                    # 标记为-1的点是异常点
                    for i, label in enumerate(cluster_labels):
                        if label == -1:
                            cluster_anomalies.append({
                                'period': periods[i],
                                'red_balls': self._extract_numbers(history_data[i])[0],
                                'blue_balls': self._extract_numbers(history_data[i])[1],
                                'cluster_label': int(label)
                            })
                
                except Exception as e:
                    logger.warning(f"聚类异常检测失败: {e}")
            
            # 计算异常统计
            anomaly_stats = {
                'total_periods': len(history_data),
                'z_score_anomalies': len(anomalies),
                'cluster_anomalies': len(cluster_anomalies),
                'anomaly_rate': round(len(anomalies) / len(history_data) * 100, 2)
            }
            
            result = {
                'analysis_type': 'anomaly_detection',
                'lottery_type': self.lottery_type,
                'data_period': len(history_data),
                'z_score_anomalies': sorted(anomalies, key=lambda x: x['max_z_score'], reverse=True),
                'cluster_anomalies': cluster_anomalies,
                'anomaly_statistics': anomaly_stats,
                'threshold_used': anomaly_threshold,
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.info(f"异常检测完成: 发现{len(anomalies)}个Z-score异常，{len(cluster_anomalies)}个聚类异常")
            return result
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return {'error': str(e)}
    
    def pattern_recognition(self, history_data: List[Dict]) -> Dict:
        """
        模式识别 - 识别历史数据中的重复模式
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            模式识别结果
        """
        try:
            logger.info("开始进行模式识别...")
            
            if len(history_data) < 30:
                return {'error': '数据不足，无法进行模式识别'}
            
            # 提取号码模式
            red_patterns = []
            blue_patterns = []
            
            for data in history_data:
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls and blue_balls:
                    # 红球模式：排序后的号码组合
                    red_pattern = tuple(sorted(red_balls))
                    red_patterns.append(red_pattern)
                    
                    # 蓝球模式
                    blue_pattern = tuple(sorted(blue_balls))
                    blue_patterns.append(blue_pattern)
            
            # 寻找重复模式
            red_pattern_counts = Counter(red_patterns)
            blue_pattern_counts = Counter(blue_patterns)
            
            # 重复的红球组合
            repeated_red_patterns = [(pattern, count) for pattern, count in red_pattern_counts.items() if count > 1]
            repeated_blue_patterns = [(pattern, count) for pattern, count in blue_pattern_counts.items() if count > 1]
            
            # 分析号码间隔模式
            interval_patterns = self._analyze_interval_patterns(history_data)
            
            # 分析和值模式
            sum_patterns = self._analyze_sum_patterns(history_data)
            
            # 分析连号模式
            consecutive_patterns = self._analyze_consecutive_patterns(history_data)
            
            result = {
                'analysis_type': 'pattern_recognition',
                'lottery_type': self.lottery_type,
                'data_period': len(history_data),
                'repeated_red_combinations': sorted(repeated_red_patterns, key=lambda x: x[1], reverse=True)[:20],
                'repeated_blue_combinations': sorted(repeated_blue_patterns, key=lambda x: x[1], reverse=True)[:20],
                'interval_patterns': interval_patterns,
                'sum_patterns': sum_patterns,
                'consecutive_patterns': consecutive_patterns,
                'pattern_statistics': {
                    'unique_red_combinations': len(red_pattern_counts),
                    'repeated_red_combinations': len(repeated_red_patterns),
                    'unique_blue_combinations': len(blue_pattern_counts),
                    'repeated_blue_combinations': len(repeated_blue_patterns),
                    'repetition_rate': round(len(repeated_red_patterns) / len(red_pattern_counts) * 100, 2)
                },
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.info(f"模式识别完成: 发现{len(repeated_red_patterns)}个重复红球模式")
            return result
            
        except Exception as e:
            logger.error(f"模式识别失败: {e}")
            return {'error': str(e)}
    
    def clustering_analysis(self, history_data: List[Dict]) -> Dict:
        """
        聚类分析 - 将历史开奖数据进行聚类分析
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            聚类分析结果
        """
        try:
            logger.info("开始进行聚类分析...")
            
            if not SKLEARN_AVAILABLE:
                return {'error': 'scikit-learn未安装，无法进行聚类分析'}
            
            if len(history_data) < 20:
                return {'error': '数据不足，无法进行聚类分析'}
            
            # 准备特征数据
            features = []
            periods = []
            
            for data in history_data:
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls and blue_balls:
                    feature_vector = self._calculate_comprehensive_features(red_balls, blue_balls)
                    features.append(feature_vector)
                    periods.append(data.get('period', ''))
            
            if len(features) < 10:
                return {'error': '有效特征数据不足'}
            
            features_array = np.array(features)
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # 确定最优聚类数量
            optimal_clusters = self._find_optimal_clusters(features_scaled)
            
            # K-means聚类
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # 分析聚类结果
            cluster_analysis = {}
            for cluster_id in range(optimal_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_periods = [periods[i] for i in cluster_indices]
                cluster_data = [history_data[i] for i in cluster_indices]
                
                # 分析该聚类的特征
                cluster_red_balls = []
                cluster_blue_balls = []
                
                for data in cluster_data:
                    red_balls, blue_balls = self._extract_numbers(data)
                    cluster_red_balls.extend(red_balls)
                    cluster_blue_balls.extend(blue_balls)
                
                cluster_analysis[cluster_id] = {
                    'size': len(cluster_indices),
                    'periods': cluster_periods[:10],  # 显示前10期
                    'common_red_numbers': [num for num, _ in Counter(cluster_red_balls).most_common(10)],
                    'common_blue_numbers': [num for num, _ in Counter(cluster_blue_balls).most_common(5)],
                    'cluster_features': self._analyze_cluster_characteristics(cluster_data)
                }
            
            # PCA降维可视化数据
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            result = {
                'analysis_type': 'clustering_analysis',
                'lottery_type': self.lottery_type,
                'data_period': len(history_data),
                'optimal_clusters': optimal_clusters,
                'cluster_analysis': cluster_analysis,
                'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'silhouette_score': silhouette_score(features_scaled, cluster_labels),
                'cluster_distribution': dict(Counter(cluster_labels)),
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.info(f"聚类分析完成: {optimal_clusters}个聚类，轮廓系数{result['silhouette_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            return {'error': str(e)}
    
    def _extract_numbers(self, data: Dict) -> Tuple[List[int], List[int]]:
        """提取号码数据"""
        try:
            if 'numbers' in data:
                numbers = data['numbers']
            elif 'numbers_data' in data:
                numbers = data['numbers_data']
            else:
                return [], []
            
            if self.lottery_type == '双色球':
                red_balls = numbers.get('red_balls', [])
                blue_balls = numbers.get('blue_balls', [])
            else:  # 大乐透
                red_balls = numbers.get('front_area', [])
                blue_balls = numbers.get('back_area', [])
            
            return red_balls, blue_balls
            
        except Exception:
            return [], []
    
    def _calculate_number_features(self, red_balls: List[int], blue_balls: List[int]) -> List[float]:
        """计算号码特征向量"""
        features = []
        
        # 红球特征
        features.append(sum(red_balls))  # 和值
        features.append(np.mean(red_balls))  # 平均值
        features.append(np.std(red_balls))  # 标准差
        features.append(max(red_balls) - min(red_balls))  # 范围
        features.append(sum(1 for x in red_balls if x % 2 == 1))  # 奇数个数
        
        # 连号统计
        sorted_reds = sorted(red_balls)
        consecutive_count = 0
        for i in range(len(sorted_reds) - 1):
            if sorted_reds[i+1] - sorted_reds[i] == 1:
                consecutive_count += 1
        features.append(consecutive_count)
        
        # 蓝球特征
        features.append(sum(blue_balls))  # 蓝球和值
        if len(blue_balls) > 1:
            features.append(max(blue_balls) - min(blue_balls))  # 蓝球范围
        else:
            features.append(0)
        
        return features
    
    def _calculate_comprehensive_features(self, red_balls: List[int], blue_balls: List[int]) -> List[float]:
        """计算更全面的特征向量"""
        features = self._calculate_number_features(red_balls, blue_balls)
        
        # 额外特征
        # 区间分布
        red_mid = (self.red_range[0] + self.red_range[1]) // 2
        features.append(sum(1 for x in red_balls if x <= red_mid))  # 小号个数
        features.append(sum(1 for x in red_balls if x > red_mid))   # 大号个数
        
        # 尾数分布
        tail_counts = Counter([x % 10 for x in red_balls])
        features.append(len(tail_counts))  # 不同尾数个数
        
        # AC值（号码复杂度）
        ac_value = 0
        for i in range(len(red_balls)):
            for j in range(i + 1, len(red_balls)):
                ac_value += abs(red_balls[i] - red_balls[j])
        features.append(ac_value)
        
        return features
    
    def _analyze_interval_patterns(self, history_data: List[Dict]) -> Dict:
        """分析号码间隔模式"""
        intervals = []
        
        for data in history_data:
            red_balls, _ = self._extract_numbers(data)
            if red_balls:
                sorted_reds = sorted(red_balls)
                for i in range(len(sorted_reds) - 1):
                    interval = sorted_reds[i+1] - sorted_reds[i]
                    intervals.append(interval)
        
        interval_counts = Counter(intervals)
        
        return {
            'most_common_intervals': interval_counts.most_common(10),
            'average_interval': round(np.mean(intervals), 2) if intervals else 0,
            'interval_distribution': dict(interval_counts)
        }
    
    def _analyze_sum_patterns(self, history_data: List[Dict]) -> Dict:
        """分析和值模式"""
        red_sums = []
        blue_sums = []
        
        for data in history_data:
            red_balls, blue_balls = self._extract_numbers(data)
            if red_balls:
                red_sums.append(sum(red_balls))
            if blue_balls:
                blue_sums.append(sum(blue_balls))
        
        return {
            'red_sum_stats': {
                'mean': round(np.mean(red_sums), 2) if red_sums else 0,
                'std': round(np.std(red_sums), 2) if red_sums else 0,
                'min': min(red_sums) if red_sums else 0,
                'max': max(red_sums) if red_sums else 0,
                'most_common': Counter(red_sums).most_common(10)
            },
            'blue_sum_stats': {
                'mean': round(np.mean(blue_sums), 2) if blue_sums else 0,
                'std': round(np.std(blue_sums), 2) if blue_sums else 0,
                'min': min(blue_sums) if blue_sums else 0,
                'max': max(blue_sums) if blue_sums else 0,
                'most_common': Counter(blue_sums).most_common(5)
            }
        }
    
    def _analyze_consecutive_patterns(self, history_data: List[Dict]) -> Dict:
        """分析连号模式"""
        consecutive_counts = []
        
        for data in history_data:
            red_balls, _ = self._extract_numbers(data)
            if red_balls:
                sorted_reds = sorted(red_balls)
                consecutive = 0
                for i in range(len(sorted_reds) - 1):
                    if sorted_reds[i+1] - sorted_reds[i] == 1:
                        consecutive += 1
                consecutive_counts.append(consecutive)
        
        consecutive_counter = Counter(consecutive_counts)
        
        return {
            'distribution': dict(consecutive_counter),
            'average_consecutive': round(np.mean(consecutive_counts), 2) if consecutive_counts else 0,
            'max_consecutive': max(consecutive_counts) if consecutive_counts else 0,
            'no_consecutive_rate': round(consecutive_counter.get(0, 0) / len(consecutive_counts) * 100, 2) if consecutive_counts else 0
        }
    
    def _find_optimal_clusters(self, features_scaled: np.ndarray) -> int:
        """寻找最优聚类数量"""
        max_clusters = min(10, len(features_scaled) // 3)
        silhouette_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # 选择轮廓系数最高的聚类数
        optimal_idx = np.argmax(silhouette_scores)
        return optimal_idx + 2
    
    def _analyze_cluster_characteristics(self, cluster_data: List[Dict]) -> Dict:
        """分析聚类特征"""
        red_all = []
        blue_all = []
        red_sums = []
        blue_sums = []
        
        for data in cluster_data:
            red_balls, blue_balls = self._extract_numbers(data)
            if red_balls and blue_balls:
                red_all.extend(red_balls)
                blue_all.extend(blue_balls)
                red_sums.append(sum(red_balls))
                blue_sums.append(sum(blue_balls))
        
        return {
            'red_sum_avg': round(np.mean(red_sums), 2) if red_sums else 0,
            'blue_sum_avg': round(np.mean(blue_sums), 2) if blue_sums else 0,
            'red_frequency_top5': [num for num, _ in Counter(red_all).most_common(5)],
            'blue_frequency_top3': [num for num, _ in Counter(blue_all).most_common(3)]
        }
