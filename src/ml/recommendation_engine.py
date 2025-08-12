"""
智能推荐系统 - 基于多种策略的号码推荐引擎
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from collections import Counter
import random

from .model_manager import ModelManager

# 条件导入分析模块
try:
    from analysis.advanced_analysis import AdvancedAnalysis
except ImportError:
    # 如果从ml模块直接导入失败，尝试相对导入
    try:
        from ..analysis.advanced_analysis import AdvancedAnalysis
    except ImportError:
        AdvancedAnalysis = None

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    智能推荐系统
    综合多种策略和模型结果，提供智能化的号码推荐
    """
    
    def __init__(self, lottery_type: str, model_manager: Optional[ModelManager] = None):
        """
        初始化推荐引擎
        
        Args:
            lottery_type: 彩票类型 ('双色球' 或 '大乐透')
            model_manager: 模型管理器实例
        """
        self.lottery_type = lottery_type
        self.model_manager = model_manager or ModelManager()
        self.advanced_analysis = AdvancedAnalysis(lottery_type)
        
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
        
        # 推荐策略配置
        self.strategies = {
            'frequency_based': {'weight': 0.25, 'enabled': True},
            'pattern_based': {'weight': 0.20, 'enabled': True},
            'ml_ensemble': {'weight': 0.30, 'enabled': True},
            'trend_following': {'weight': 0.15, 'enabled': True},
            'anti_trend': {'weight': 0.10, 'enabled': True}
        }
    
    def generate_recommendations(self, history_data: List[Dict], 
                               num_recommendations: int = 5,
                               strategy_weights: Optional[Dict] = None) -> Dict:
        """
        生成智能推荐
        
        Args:
            history_data: 历史开奖数据
            num_recommendations: 推荐组合数量
            strategy_weights: 自定义策略权重
            
        Returns:
            推荐结果字典
        """
        try:
            logger.info(f"开始生成{num_recommendations}组{self.lottery_type}推荐...")
            
            if len(history_data) < 10:
                return {
                    'success': False,
                    'error': '历史数据不足，无法生成推荐',
                    'recommendations': []
                }
            
            # 更新策略权重
            if strategy_weights:
                for strategy, weight in strategy_weights.items():
                    if strategy in self.strategies:
                        self.strategies[strategy]['weight'] = weight
            
            # 生成各策略的推荐
            strategy_recommendations = {}
            
            # 1. 频率策略
            if self.strategies['frequency_based']['enabled']:
                strategy_recommendations['frequency_based'] = self._frequency_strategy(history_data)
            
            # 2. 模式策略
            if self.strategies['pattern_based']['enabled']:
                strategy_recommendations['pattern_based'] = self._pattern_strategy(history_data)
            
            # 3. 机器学习集成
            if self.strategies['ml_ensemble']['enabled']:
                strategy_recommendations['ml_ensemble'] = self._ml_ensemble_strategy(history_data)
            
            # 4. 趋势跟随
            if self.strategies['trend_following']['enabled']:
                strategy_recommendations['trend_following'] = self._trend_following_strategy(history_data)
            
            # 5. 反趋势策略
            if self.strategies['anti_trend']['enabled']:
                strategy_recommendations['anti_trend'] = self._anti_trend_strategy(history_data)
            
            # 综合推荐
            final_recommendations = self._combine_strategies(
                strategy_recommendations, num_recommendations
            )
            
            # 风险评估
            risk_assessment = self._assess_risk(final_recommendations, history_data)
            
            # 计算推荐置信度
            confidence_scores = self._calculate_confidence(
                final_recommendations, strategy_recommendations, history_data
            )
            
            result = {
                'success': True,
                'lottery_type': self.lottery_type,
                'recommendations': final_recommendations,
                'strategy_details': strategy_recommendations,
                'risk_assessment': risk_assessment,
                'confidence_scores': confidence_scores,
                'strategy_weights': {k: v['weight'] for k, v in self.strategies.items() if v['enabled']},
                'generation_time': datetime.now().isoformat(),
                'data_period': len(history_data)
            }
            
            logger.info(f"推荐生成完成，共{len(final_recommendations)}组推荐")
            return result
            
        except Exception as e:
            logger.error(f"推荐生成失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }
    
    def _frequency_strategy(self, history_data: List[Dict]) -> Dict:
        """基于频率的推荐策略"""
        try:
            red_freq = Counter()
            blue_freq = Counter()
            
            # 统计最近期数的频率
            recent_data = history_data[-50:] if len(history_data) > 50 else history_data
            
            for data in recent_data:
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls and blue_balls:
                    for num in red_balls:
                        red_freq[num] += 1
                    for num in blue_balls:
                        blue_freq[num] += 1
            
            # 热号推荐（高频号码）
            hot_red = [num for num, _ in red_freq.most_common(15)]
            hot_blue = [num for num, _ in blue_freq.most_common(8)]
            
            # 冷号推荐（低频号码）
            all_red = set(range(self.red_range[0], self.red_range[1] + 1))
            all_blue = set(range(self.blue_range[0], self.blue_range[1] + 1))
            
            appeared_red = set(red_freq.keys())
            appeared_blue = set(blue_freq.keys())
            
            cold_red = list(all_red - appeared_red)
            cold_blue = list(all_blue - appeared_blue)
            
            # 如果冷号不足，补充低频号码
            if len(cold_red) < 10:
                cold_red.extend([num for num, _ in red_freq.most_common()[-10:]])
            
            if len(cold_blue) < 5:
                cold_blue.extend([num for num, _ in blue_freq.most_common()[-5:]])
            
            # 生成推荐组合
            recommendations = []
            
            # 热号组合
            if len(hot_red) >= self.red_count and len(hot_blue) >= self.blue_count:
                red_combo = random.sample(hot_red[:12], self.red_count)
                blue_combo = random.sample(hot_blue[:6], self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'hot_numbers',
                    'confidence': 0.7
                })
            
            # 冷号组合
            if len(cold_red) >= self.red_count and len(cold_blue) >= self.blue_count:
                red_combo = random.sample(cold_red[:10], self.red_count)
                blue_combo = random.sample(cold_blue[:4], self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'cold_numbers',
                    'confidence': 0.5
                })
            
            # 混合组合
            if len(hot_red) >= 3 and len(cold_red) >= 3:
                red_combo = (random.sample(hot_red[:8], min(3, self.red_count // 2)) + 
                           random.sample(cold_red[:8], self.red_count - min(3, self.red_count // 2)))
                blue_combo = random.sample(hot_blue[:4] + cold_blue[:2], self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'mixed_frequency',
                    'confidence': 0.6
                })
            
            return {
                'strategy': 'frequency_based',
                'recommendations': recommendations,
                'hot_red_numbers': hot_red,
                'hot_blue_numbers': hot_blue,
                'cold_red_numbers': cold_red[:10],
                'cold_blue_numbers': cold_blue[:5]
            }
            
        except Exception as e:
            logger.error(f"频率策略失败: {e}")
            return {'strategy': 'frequency_based', 'recommendations': [], 'error': str(e)}
    
    def _pattern_strategy(self, history_data: List[Dict]) -> Dict:
        """基于模式的推荐策略"""
        try:
            # 分析号码间隔模式
            intervals = []
            sum_values = []
            
            for data in history_data[-30:]:  # 分析最近30期
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls:
                    sorted_reds = sorted(red_balls)
                    # 计算间隔
                    for i in range(len(sorted_reds) - 1):
                        intervals.append(sorted_reds[i+1] - sorted_reds[i])
                    sum_values.append(sum(red_balls))
            
            # 统计最常见的间隔
            interval_freq = Counter(intervals)
            common_intervals = [interval for interval, _ in interval_freq.most_common(5)]
            
            # 统计和值分布
            sum_mean = np.mean(sum_values) if sum_values else 100
            sum_std = np.std(sum_values) if sum_values else 20
            
            recommendations = []
            
            # 基于常见间隔生成组合
            for _ in range(2):
                red_combo = []
                start_num = random.randint(self.red_range[0], self.red_range[0] + 10)
                red_combo.append(start_num)
                
                current_num = start_num
                while len(red_combo) < self.red_count:
                    interval = random.choice(common_intervals) if common_intervals else random.randint(1, 5)
                    next_num = current_num + interval
                    if next_num <= self.red_range[1] and next_num not in red_combo:
                        red_combo.append(next_num)
                        current_num = next_num
                    else:
                        # 重新选择
                        available_nums = [n for n in range(self.red_range[0], self.red_range[1] + 1) 
                                        if n not in red_combo]
                        if available_nums:
                            next_num = random.choice(available_nums)
                            red_combo.append(next_num)
                            current_num = next_num
                        else:
                            break
                
                # 确保有足够的号码
                while len(red_combo) < self.red_count:
                    available_nums = [n for n in range(self.red_range[0], self.red_range[1] + 1) 
                                    if n not in red_combo]
                    if available_nums:
                        red_combo.append(random.choice(available_nums))
                    else:
                        break
                
                # 蓝球随机选择
                blue_combo = random.sample(range(self.blue_range[0], self.blue_range[1] + 1), 
                                         self.blue_count)
                
                if len(red_combo) == self.red_count:
                    recommendations.append({
                        'red_balls': sorted(red_combo[:self.red_count]),
                        'blue_balls': blue_combo,
                        'type': 'interval_pattern',
                        'confidence': 0.6
                    })
            
            # 基于和值范围生成组合
            target_sum = random.normalvariate(sum_mean, sum_std)
            target_sum = max(self.red_range[0] * self.red_count, 
                           min(self.red_range[1] * self.red_count, target_sum))
            
            red_combo = self._generate_sum_based_combination(int(target_sum))
            if red_combo:
                blue_combo = random.sample(range(self.blue_range[0], self.blue_range[1] + 1), 
                                         self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'sum_pattern',
                    'confidence': 0.55,
                    'target_sum': int(target_sum)
                })
            
            return {
                'strategy': 'pattern_based',
                'recommendations': recommendations,
                'common_intervals': common_intervals,
                'sum_statistics': {
                    'mean': round(sum_mean, 2),
                    'std': round(sum_std, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"模式策略失败: {e}")
            return {'strategy': 'pattern_based', 'recommendations': [], 'error': str(e)}
    
    def _ml_ensemble_strategy(self, history_data: List[Dict]) -> Dict:
        """机器学习集成策略"""
        try:
            if not self.model_manager:
                return {'strategy': 'ml_ensemble', 'recommendations': [], 'error': '模型管理器未初始化'}
            
            # 获取集成预测结果
            ensemble_result = self.model_manager.ensemble_predict(self.lottery_type, history_data)
            
            recommendations = []
            
            if ensemble_result.get('success'):
                # 主推荐：集成结果
                recommendations.append({
                    'red_balls': ensemble_result['red_balls'],
                    'blue_balls': ensemble_result['blue_balls'],
                    'type': 'ml_ensemble',
                    'confidence': ensemble_result.get('confidence', 0.7),
                    'models_used': ensemble_result.get('models_used', [])
                })
                
                # 基于个别模型的推荐
                individual_predictions = ensemble_result.get('individual_predictions', {})
                for model_name, prediction in list(individual_predictions.items())[:2]:
                    recommendations.append({
                        'red_balls': prediction['red_balls'],
                        'blue_balls': prediction['blue_balls'],
                        'type': f'ml_model_{model_name}',
                        'confidence': 0.6,
                        'source_model': model_name
                    })
            
            # 如果ML预测失败，生成随机推荐
            if not recommendations:
                for _ in range(2):
                    red_combo = random.sample(range(self.red_range[0], self.red_range[1] + 1), 
                                            self.red_count)
                    blue_combo = random.sample(range(self.blue_range[0], self.blue_range[1] + 1), 
                                             self.blue_count)
                    recommendations.append({
                        'red_balls': sorted(red_combo),
                        'blue_balls': blue_combo,
                        'type': 'ml_fallback',
                        'confidence': 0.4
                    })
            
            return {
                'strategy': 'ml_ensemble',
                'recommendations': recommendations,
                'ensemble_details': ensemble_result
            }
            
        except Exception as e:
            logger.error(f"ML集成策略失败: {e}")
            return {'strategy': 'ml_ensemble', 'recommendations': [], 'error': str(e)}
    
    def _trend_following_strategy(self, history_data: List[Dict]) -> Dict:
        """趋势跟随策略"""
        try:
            # 分析最近的趋势
            recent_periods = min(20, len(history_data))
            recent_data = history_data[-recent_periods:]
            
            red_trends = Counter()
            blue_trends = Counter()
            
            # 加权统计（越近的期数权重越大）
            for i, data in enumerate(recent_data):
                weight = (i + 1) / len(recent_data)  # 线性递增权重
                red_balls, blue_balls = self._extract_numbers(data)
                
                if red_balls and blue_balls:
                    for num in red_balls:
                        red_trends[num] += weight
                    for num in blue_balls:
                        blue_trends[num] += weight
            
            # 获取上升趋势的号码
            trending_red = [num for num, _ in red_trends.most_common(12)]
            trending_blue = [num for num, _ in blue_trends.most_common(6)]
            
            recommendations = []
            
            # 强趋势组合
            if len(trending_red) >= self.red_count and len(trending_blue) >= self.blue_count:
                red_combo = random.sample(trending_red[:8], self.red_count)
                blue_combo = random.sample(trending_blue[:4], self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'strong_trend',
                    'confidence': 0.65
                })
            
            # 中等趋势组合
            if len(trending_red) >= 8:
                red_combo = random.sample(trending_red[4:12], self.red_count)
                blue_combo = random.sample(trending_blue[2:] if len(trending_blue) > 2 else trending_blue, 
                                         self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'medium_trend',
                    'confidence': 0.55
                })
            
            return {
                'strategy': 'trend_following',
                'recommendations': recommendations,
                'trending_red_numbers': trending_red,
                'trending_blue_numbers': trending_blue
            }
            
        except Exception as e:
            logger.error(f"趋势跟随策略失败: {e}")
            return {'strategy': 'trend_following', 'recommendations': [], 'error': str(e)}
    
    def _anti_trend_strategy(self, history_data: List[Dict]) -> Dict:
        """反趋势策略"""
        try:
            # 找到最近较少出现的号码
            recent_periods = min(15, len(history_data))
            recent_data = history_data[-recent_periods:]
            
            red_freq = Counter()
            blue_freq = Counter()
            
            for data in recent_data:
                red_balls, blue_balls = self._extract_numbers(data)
                if red_balls and blue_balls:
                    for num in red_balls:
                        red_freq[num] += 1
                    for num in blue_balls:
                        blue_freq[num] += 1
            
            # 获取所有号码
            all_red = set(range(self.red_range[0], self.red_range[1] + 1))
            all_blue = set(range(self.blue_range[0], self.blue_range[1] + 1))
            
            # 低频或未出现的号码
            low_freq_red = [num for num in all_red if red_freq.get(num, 0) <= 1]
            low_freq_blue = [num for num in all_blue if blue_freq.get(num, 0) <= 1]
            
            # 补充次低频号码
            if len(low_freq_red) < self.red_count * 2:
                additional_red = [num for num, count in red_freq.items() if count <= 2]
                low_freq_red.extend(additional_red)
            
            if len(low_freq_blue) < self.blue_count * 2:
                additional_blue = [num for num, count in blue_freq.items() if count <= 1]
                low_freq_blue.extend(additional_blue)
            
            recommendations = []
            
            # 反趋势组合
            if len(low_freq_red) >= self.red_count and len(low_freq_blue) >= self.blue_count:
                red_combo = random.sample(low_freq_red[:15], self.red_count)
                blue_combo = random.sample(low_freq_blue[:8], self.blue_count)
                recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'anti_trend',
                    'confidence': 0.45
                })
            
            return {
                'strategy': 'anti_trend',
                'recommendations': recommendations,
                'low_frequency_red': low_freq_red[:10],
                'low_frequency_blue': low_freq_blue[:5]
            }
            
        except Exception as e:
            logger.error(f"反趋势策略失败: {e}")
            return {'strategy': 'anti_trend', 'recommendations': [], 'error': str(e)}
    
    def _combine_strategies(self, strategy_recommendations: Dict, 
                          num_recommendations: int) -> List[Dict]:
        """综合各策略推荐"""
        all_recommendations = []
        
        # 收集所有推荐
        for strategy_name, strategy_result in strategy_recommendations.items():
            if 'recommendations' in strategy_result:
                for rec in strategy_result['recommendations']:
                    rec['strategy'] = strategy_name
                    rec['weight'] = self.strategies[strategy_name]['weight']
                    all_recommendations.append(rec)
        
        # 按置信度和权重排序
        all_recommendations.sort(
            key=lambda x: x.get('confidence', 0.5) * x.get('weight', 0.5), 
            reverse=True
        )
        
        # 去重并选择最佳推荐
        final_recommendations = []
        used_combinations = set()
        
        for rec in all_recommendations:
            combo_key = (tuple(rec['red_balls']), tuple(rec['blue_balls']))
            if combo_key not in used_combinations:
                used_combinations.add(combo_key)
                final_recommendations.append(rec)
                
                if len(final_recommendations) >= num_recommendations:
                    break
        
        # 如果推荐不足，生成补充推荐
        while len(final_recommendations) < num_recommendations:
            red_combo = random.sample(range(self.red_range[0], self.red_range[1] + 1), 
                                    self.red_count)
            blue_combo = random.sample(range(self.blue_range[0], self.blue_range[1] + 1), 
                                     self.blue_count)
            
            combo_key = (tuple(sorted(red_combo)), tuple(blue_combo))
            if combo_key not in used_combinations:
                used_combinations.add(combo_key)
                final_recommendations.append({
                    'red_balls': sorted(red_combo),
                    'blue_balls': blue_combo,
                    'type': 'random_supplement',
                    'confidence': 0.3,
                    'strategy': 'random'
                })
        
        return final_recommendations[:num_recommendations]
    
    def _assess_risk(self, recommendations: List[Dict], history_data: List[Dict]) -> Dict:
        """评估推荐的风险"""
        try:
            risk_scores = []
            
            for rec in recommendations:
                risk_score = 0
                
                # 1. 重复风险 - 检查是否与历史组合重复
                combo_key = (tuple(rec['red_balls']), tuple(rec['blue_balls']))
                for data in history_data[-100:]:  # 检查最近100期
                    red_balls, blue_balls = self._extract_numbers(data)
                    if red_balls and blue_balls:
                        hist_key = (tuple(sorted(red_balls)), tuple(sorted(blue_balls)))
                        if combo_key == hist_key:
                            risk_score += 0.8  # 完全重复，高风险
                            break
                        
                        # 部分重复检查
                        red_overlap = len(set(rec['red_balls']) & set(red_balls))
                        blue_overlap = len(set(rec['blue_balls']) & set(blue_balls))
                        
                        if red_overlap >= self.red_count - 1 and blue_overlap == self.blue_count:
                            risk_score += 0.3  # 高度相似
                
                # 2. 分布风险 - 号码分布是否极端
                red_balls = rec['red_balls']
                
                # 检查连号过多
                consecutive_count = 0
                for i in range(len(red_balls) - 1):
                    if red_balls[i+1] - red_balls[i] == 1:
                        consecutive_count += 1
                
                if consecutive_count >= 3:
                    risk_score += 0.2
                
                # 检查号码分布是否过于集中
                number_range = max(red_balls) - min(red_balls)
                expected_range = (self.red_range[1] - self.red_range[0]) * 0.3
                
                if number_range < expected_range:
                    risk_score += 0.2
                
                # 3. 奇偶比例风险
                odd_count = sum(1 for num in red_balls if num % 2 == 1)
                if odd_count == 0 or odd_count == len(red_balls):
                    risk_score += 0.1
                
                risk_scores.append(min(1.0, risk_score))
            
            return {
                'individual_risks': risk_scores,
                'average_risk': round(np.mean(risk_scores), 3),
                'high_risk_count': sum(1 for score in risk_scores if score > 0.6),
                'risk_categories': {
                    'low': sum(1 for score in risk_scores if score <= 0.3),
                    'medium': sum(1 for score in risk_scores if 0.3 < score <= 0.6),
                    'high': sum(1 for score in risk_scores if score > 0.6)
                }
            }
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence(self, final_recommendations: List[Dict],
                            strategy_recommendations: Dict, 
                            history_data: List[Dict]) -> List[float]:
        """计算推荐置信度"""
        try:
            confidence_scores = []
            
            for rec in final_recommendations:
                base_confidence = rec.get('confidence', 0.5)
                strategy_weight = rec.get('weight', 0.5)
                
                # 基于策略权重调整置信度
                adjusted_confidence = base_confidence * (0.7 + 0.3 * strategy_weight)
                
                # 基于数据质量调整
                data_quality_factor = min(1.0, len(history_data) / 100)
                adjusted_confidence *= (0.8 + 0.2 * data_quality_factor)
                
                confidence_scores.append(round(min(1.0, adjusted_confidence), 3))
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return [0.5] * len(final_recommendations)
    
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
    
    def _generate_sum_based_combination(self, target_sum: int) -> List[int]:
        """基于目标和值生成号码组合"""
        try:
            red_combo = []
            remaining_sum = target_sum
            remaining_count = self.red_count
            
            for i in range(self.red_count):
                if remaining_count == 1:
                    # 最后一个号码
                    last_num = remaining_sum
                    if self.red_range[0] <= last_num <= self.red_range[1] and last_num not in red_combo:
                        red_combo.append(last_num)
                    break
                else:
                    # 计算可能的范围
                    min_remaining = sum(range(self.red_range[0], self.red_range[0] + remaining_count - 1))
                    max_remaining = sum(range(self.red_range[1] - remaining_count + 2, self.red_range[1] + 1))
                    
                    min_current = max(self.red_range[0], remaining_sum - max_remaining)
                    max_current = min(self.red_range[1], remaining_sum - min_remaining)
                    
                    if min_current <= max_current:
                        # 从可用号码中随机选择
                        available_nums = [n for n in range(min_current, max_current + 1) 
                                        if n not in red_combo]
                        if available_nums:
                            selected_num = random.choice(available_nums)
                            red_combo.append(selected_num)
                            remaining_sum -= selected_num
                            remaining_count -= 1
                        else:
                            break
                    else:
                        break
            
            return red_combo if len(red_combo) == self.red_count else None
            
        except Exception:
            return None
    
    def update_strategy_weights(self, new_weights: Dict[str, float]) -> bool:
        """更新策略权重"""
        try:
            total_weight = sum(new_weights.values())
            if total_weight <= 0:
                return False
            
            # 标准化权重
            for strategy, weight in new_weights.items():
                if strategy in self.strategies:
                    self.strategies[strategy]['weight'] = weight / total_weight
            
            logger.info("策略权重已更新")
            return True
            
        except Exception as e:
            logger.error(f"更新策略权重失败: {e}")
            return False
    
    def get_strategy_performance(self, history_data: List[Dict], 
                               verification_data: List[Dict]) -> Dict:
        """评估各策略的历史性能"""
        try:
            # 这里可以实现策略回测功能
            # 基于历史数据评估各策略的表现
            logger.info("策略性能评估功能待实现")
            return {'message': '策略性能评估功能待实现'}
            
        except Exception as e:
            logger.error(f"策略性能评估失败: {e}")
            return {'error': str(e)}
