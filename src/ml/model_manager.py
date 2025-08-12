"""
模型管理器 - 统一管理所有机器学习预测模型
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from collections import defaultdict

from .base_predictor import BasePredictor
from .random_forest_predictor import RandomForestPredictor

# 条件导入可选依赖
try:
    from .xgboost_predictor import XGBoostPredictor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBoostPredictor = None
    XGBOOST_AVAILABLE = False

try:
    from .lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except (ImportError, NameError):
    LSTMPredictor = None
    LSTM_AVAILABLE = False

try:
    from .advanced_predictor import AdvancedPredictor
    ADVANCED_AVAILABLE = True
except ImportError:
    AdvancedPredictor = None
    ADVANCED_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelManager:
    """
    模型管理器
    负责创建、训练、保存、加载和管理所有预测模型
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型保存目录
        """
        self.models_dir = models_dir
        self.models = {}  # 存储所有模型实例
        self.model_configs = {}  # 存储模型配置
        
        # 创建模型目录
        os.makedirs(models_dir, exist_ok=True)
        
        # 支持的模型类型
        self.supported_models = {
            'RandomForest': RandomForestPredictor,
        }
        
        if XGBOOST_AVAILABLE:
            self.supported_models['XGBoost'] = XGBoostPredictor
        
        if LSTM_AVAILABLE:
            self.supported_models['LSTM'] = LSTMPredictor
        
        if ADVANCED_AVAILABLE:
            self.supported_models['Advanced'] = AdvancedPredictor
        
        logger.info(f"模型管理器初始化完成，支持的模型: {list(self.supported_models.keys())}")
    
    def auto_select_model(self, lottery_type: str, history_data: List[Dict]) -> str:
        """
        根据数据特征自动选择最佳模型
        
        Args:
            lottery_type: 彩票类型
            history_data: 历史数据
            
        Returns:
            推荐的模型类型
        """
        try:
            data_size = len(history_data)
            
            # 根据数据量选择模型
            if data_size < 50:
                return 'RandomForest'  # 小数据集使用随机森林
            elif data_size < 200:
                if XGBOOST_AVAILABLE:
                    return 'XGBoost'  # 中等数据集使用XGBoost
                else:
                    return 'RandomForest'
            else:
                if ADVANCED_AVAILABLE:
                    return 'Advanced'  # 大数据集使用高级预测器
                elif LSTM_AVAILABLE:
                    return 'LSTM'  # 备选LSTM
                elif XGBOOST_AVAILABLE:
                    return 'XGBoost'  # 备选XGBoost
                else:
                    return 'RandomForest'  # 最后备选
                    
        except Exception as e:
            logger.warning(f"自动模型选择失败: {e}")
            return 'RandomForest'  # 默认选择
    
    def create_model(self, model_type: str, lottery_type: str, **kwargs) -> Optional[BasePredictor]:
        """
        创建预测模型
        
        Args:
            model_type: 模型类型 ('RandomForest', 'XGBoost', 'LSTM', 'Advanced')
            lottery_type: 彩票类型 ('双色球', '大乐透')
            **kwargs: 模型特定的参数
            
        Returns:
            创建的模型实例
        """
        try:
            if model_type not in self.supported_models:
                logger.error(f"不支持的模型类型: {model_type}")
                return None
            
            if lottery_type not in ['双色球', '大乐透']:
                logger.error(f"不支持的彩票类型: {lottery_type}")
                return None
            
            model_class = self.supported_models[model_type]
            model = model_class(lottery_type, **kwargs)
            
            model_key = f"{model_type}_{lottery_type}"
            self.models[model_key] = model
            
            # 保存模型配置
            self.model_configs[model_key] = {
                'model_type': model_type,
                'lottery_type': lottery_type,
                'parameters': kwargs,
                'created_at': datetime.now().isoformat(),
                'trained': False
            }
            
            logger.info(f"模型创建成功: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"模型创建失败: {e}")
            return None
    
    def train_model(self, model_key: str, history_data: List[Dict]) -> bool:
        """
        训练指定模型
        
        Args:
            model_key: 模型键名
            history_data: 历史训练数据
            
        Returns:
            训练是否成功
        """
        try:
            if model_key not in self.models:
                logger.error(f"模型不存在: {model_key}")
                return False
            
            model = self.models[model_key]
            success = model.train(history_data)
            
            if success:
                # 更新配置
                self.model_configs[model_key]['trained'] = True
                self.model_configs[model_key]['trained_at'] = datetime.now().isoformat()
                self.model_configs[model_key]['training_data_size'] = len(history_data)
                
                logger.info(f"模型训练成功: {model_key}")
            else:
                logger.error(f"模型训练失败: {model_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"模型训练异常: {e}")
            return False
    
    def predict_with_model(self, model_key: str, recent_data: List[Dict]) -> Dict:
        """
        使用指定模型进行预测
        
        Args:
            model_key: 模型键名
            recent_data: 最近的历史数据
            
        Returns:
            预测结果
        """
        try:
            if model_key not in self.models:
                return {
                    'success': False,
                    'error': f'模型不存在: {model_key}',
                    'model_key': model_key
                }
            
            model = self.models[model_key]
            
            if not model.is_trained:
                return {
                    'success': False,
                    'error': f'模型未训练: {model_key}',
                    'model_key': model_key
                }
            
            result = model.predict(recent_data)
            result['model_key'] = model_key
            
            return result
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_key': model_key
            }
    
    def smart_ensemble_predict(self, lottery_type: str, recent_data: List[Dict],
                              adaptive_weights: bool = True) -> Dict:
        """
        智能集成预测 - 支持自适应权重调整的集成预测
        
        Args:
            lottery_type: 彩票类型
            recent_data: 最近的历史数据
            adaptive_weights: 是否使用自适应权重
            
        Returns:
            智能集成预测结果
        """
        try:
            # 获取该彩票类型的所有已训练模型
            trained_models = []
            for model_key, model in self.models.items():
                if lottery_type in model_key and model.is_trained:
                    trained_models.append(model_key)
            
            if not trained_models:
                return {
                    'success': False,
                    'error': f'没有已训练的{lottery_type}模型',
                    'lottery_type': lottery_type
                }
            
            # 获取所有模型的预测结果
            predictions = {}
            confidences = {}
            performance_scores = {}
            
            for model_key in trained_models:
                result = self.predict_with_model(model_key, recent_data)
                if result.get('success'):
                    predictions[model_key] = {
                        'red_balls': result['red_balls'],
                        'blue_balls': result['blue_balls']
                    }
                    confidences[model_key] = result.get('confidence', 0.5)
                    
                    # 获取模型性能分数
                    model = self.models[model_key]
                    if hasattr(model, 'performance_metrics'):
                        metrics = model.performance_metrics
                        # 综合性能评分
                        perf_score = self._calculate_performance_score(metrics)
                        performance_scores[model_key] = perf_score
                    else:
                        performance_scores[model_key] = 0.5
            
            if not predictions:
                return {
                    'success': False,
                    'error': '所有模型预测失败',
                    'lottery_type': lottery_type
                }
            
            # 计算自适应权重
            if adaptive_weights:
                weights = self._calculate_adaptive_weights(
                    confidences, performance_scores, trained_models
                )
            else:
                weights = {model_key: 1.0 for model_key in predictions.keys()}
            
            # 标准化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {k: w / total_weight for k, w in weights.items()}
            else:
                normalized_weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
            
            # 智能集成预测
            ensemble_result = self._smart_ensemble_predictions(
                predictions, normalized_weights, confidences, lottery_type
            )
            
            ensemble_result.update({
                'success': True,
                'model_name': f'SmartEnsemble_{lottery_type}',
                'lottery_type': lottery_type,
                'prediction_time': datetime.now().isoformat(),
                'models_used': list(predictions.keys()),
                'adaptive_weights': normalized_weights,
                'performance_scores': performance_scores,
                'individual_predictions': predictions,
                'ensemble_method': 'smart_adaptive'
            })
            
            logger.info(f"智能集成预测完成: {lottery_type}, 使用模型: {list(predictions.keys())}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"智能集成预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'lottery_type': lottery_type
            }
    
    def ensemble_predict(self, lottery_type: str, recent_data: List[Dict], 
                        weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        集成预测 - 使用多个模型进行预测并综合结果
        
        Args:
            lottery_type: 彩票类型
            recent_data: 最近的历史数据
            weights: 模型权重字典
            
        Returns:
            集成预测结果
        """
        try:
            # 获取该彩票类型的所有已训练模型
            trained_models = []
            for model_key, model in self.models.items():
                if lottery_type in model_key and model.is_trained:
                    trained_models.append(model_key)
            
            if not trained_models:
                return {
                    'success': False,
                    'error': f'没有已训练的{lottery_type}模型',
                    'lottery_type': lottery_type
                }
            
            # 获取所有模型的预测结果
            predictions = {}
            confidences = {}
            
            for model_key in trained_models:
                result = self.predict_with_model(model_key, recent_data)
                if result.get('success'):
                    predictions[model_key] = {
                        'red_balls': result['red_balls'],
                        'blue_balls': result['blue_balls']
                    }
                    confidences[model_key] = result.get('confidence', 0.5)
            
            if not predictions:
                return {
                    'success': False,
                    'error': '所有模型预测失败',
                    'lottery_type': lottery_type
                }
            
            # 设置默认权重
            if weights is None:
                weights = {model_key: 1.0 for model_key in predictions.keys()}
            
            # 标准化权重
            total_weight = sum(weights.get(k, 0) for k in predictions.keys())
            if total_weight > 0:
                normalized_weights = {k: weights.get(k, 0) / total_weight 
                                    for k in predictions.keys()}
            else:
                normalized_weights = {k: 1.0 / len(predictions) 
                                    for k in predictions.keys()}
            
            # 集成预测
            ensemble_result = self._ensemble_predictions(
                predictions, normalized_weights, confidences, lottery_type
            )
            
            ensemble_result.update({
                'success': True,
                'model_name': f'Ensemble_{lottery_type}',
                'lottery_type': lottery_type,
                'prediction_time': datetime.now().isoformat(),
                'models_used': list(predictions.keys()),
                'model_weights': normalized_weights,
                'individual_predictions': predictions
            })
            
            logger.info(f"集成预测完成: {lottery_type}, 使用模型: {list(predictions.keys())}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'lottery_type': lottery_type
            }
    
    def _ensemble_predictions(self, predictions: Dict, weights: Dict, 
                            confidences: Dict, lottery_type: str) -> Dict:
        """
        集成多个模型的预测结果
        """
        # 获取彩票规则
        if lottery_type == '双色球':
            red_range = (1, 33)
            blue_range = (1, 16)
            red_count = 6
            blue_count = 1
        else:  # 大乐透
            red_range = (1, 35)
            blue_range = (1, 12)
            red_count = 5
            blue_count = 2
        
        # 统计每个号码的加权投票
        red_votes = {}
        blue_votes = {}
        
        for model_key, pred in predictions.items():
            weight = weights.get(model_key, 0)
            confidence = confidences.get(model_key, 0.5)
            
            # 计算实际权重（权重 × 置信度）
            actual_weight = weight * confidence
            
            # 红球投票
            for num in pred['red_balls']:
                red_votes[num] = red_votes.get(num, 0) + actual_weight
            
            # 蓝球投票
            for num in pred['blue_balls']:
                blue_votes[num] = blue_votes.get(num, 0) + actual_weight
        
        # 选择得票最高的号码
        red_sorted = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)
        blue_sorted = sorted(blue_votes.items(), key=lambda x: x[1], reverse=True)
        
        # 获取最终预测结果
        ensemble_red = [num for num, _ in red_sorted[:red_count]]
        ensemble_blue = [num for num, _ in blue_sorted[:blue_count]]
        
        # 如果数量不足，随机补充
        if len(ensemble_red) < red_count:
            used_red = set(ensemble_red)
            for num in range(red_range[0], red_range[1] + 1):
                if num not in used_red:
                    ensemble_red.append(num)
                    if len(ensemble_red) >= red_count:
                        break
        
        if len(ensemble_blue) < blue_count:
            used_blue = set(ensemble_blue)
            for num in range(blue_range[0], blue_range[1] + 1):
                if num not in used_blue:
                    ensemble_blue.append(num)
                    if len(ensemble_blue) >= blue_count:
                        break
        
        # 计算集成置信度
        ensemble_confidence = sum(confidences[k] * weights[k] 
                                for k in predictions.keys()) / len(predictions)
        
        return {
            'red_balls': sorted(ensemble_red[:red_count]),
            'blue_balls': ensemble_blue[:blue_count],
            'confidence': round(ensemble_confidence, 3),
            'red_votes': dict(red_sorted),
            'blue_votes': dict(blue_sorted)
        }
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """计算模型性能综合评分"""
        try:
            score = 0.5  # 基础分数
            
            # 基于训练时间的评分（训练时间越短越好）
            training_time = metrics.get('training_time', 0)
            if training_time > 0:
                time_score = min(1.0, 60 / training_time)  # 60秒内完成训练得满分
                score += time_score * 0.1
            
            # 基于预测时间的评分（预测时间越短越好）
            prediction_time = metrics.get('prediction_time', 0)
            if prediction_time > 0:
                pred_time_score = min(1.0, 1 / prediction_time)  # 1秒内完成预测得满分
                score += pred_time_score * 0.1
            
            # 基于历史准确率的评分
            accuracy_history = metrics.get('accuracy_history', [])
            if accuracy_history:
                recent_accuracies = [item['accuracy'] for item in accuracy_history[-10:]]
                avg_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                score += avg_accuracy * 0.4
            
            # 基于模型特定指标的评分
            if 'red_cv_score' in metrics:
                cv_score = 1 / (1 + metrics['red_cv_score'])  # CV分数越小越好
                score += cv_score * 0.2
            
            if 'optimization_score' in metrics:
                opt_score = 1 / (1 + metrics['optimization_score'])
                score += opt_score * 0.1
            
            return min(1.0, max(0.1, score))
            
        except Exception as e:
            logger.warning(f"计算性能评分失败: {e}")
            return 0.5
    
    def _calculate_adaptive_weights(self, confidences: Dict[str, float], 
                                  performance_scores: Dict[str, float],
                                  model_keys: List[str]) -> Dict[str, float]:
        """计算自适应权重"""
        try:
            weights = {}
            
            for model_key in model_keys:
                confidence = confidences.get(model_key, 0.5)
                performance = performance_scores.get(model_key, 0.5)
                
                # 综合置信度和性能评分
                base_weight = (confidence * 0.6 + performance * 0.4)
                
                # 根据模型类型调整权重
                if 'Advanced' in model_key:
                    type_bonus = 1.2  # 高级模型获得额外权重
                elif 'XGBoost' in model_key or 'LSTM' in model_key:
                    type_bonus = 1.1  # 先进模型获得少量额外权重
                else:
                    type_bonus = 1.0  # 基础模型保持原权重
                
                weights[model_key] = base_weight * type_bonus
            
            return weights
            
        except Exception as e:
            logger.warning(f"计算自适应权重失败: {e}")
            return {key: 1.0 for key in model_keys}
    
    def _smart_ensemble_predictions(self, predictions: Dict, weights: Dict,
                                  confidences: Dict, lottery_type: str) -> Dict:
        """智能集成多个模型的预测结果"""
        # 获取彩票规则
        if lottery_type == '双色球':
            red_range = (1, 33)
            blue_range = (1, 16)
            red_count = 6
            blue_count = 1
        else:  # 大乐透
            red_range = (1, 35)
            blue_range = (1, 12)
            red_count = 5
            blue_count = 2
        
        # 加权投票，考虑置信度和性能
        red_votes = defaultdict(float)
        blue_votes = defaultdict(float)
        
        for model_key, pred in predictions.items():
            weight = weights.get(model_key, 0)
            confidence = confidences.get(model_key, 0.5)
            
            # 计算实际权重（权重 × 置信度 × 额外的智能因子）
            smart_factor = self._calculate_smart_factor(pred, lottery_type)
            actual_weight = weight * confidence * smart_factor
            
            # 红球投票
            for num in pred['red_balls']:
                red_votes[num] += actual_weight
            
            # 蓝球投票
            for num in pred['blue_balls']:
                blue_votes[num] += actual_weight
        
        # 智能选择：不仅考虑票数，还考虑分布合理性
        red_candidates = sorted(red_votes.items(), key=lambda x: x[1], reverse=True)
        blue_candidates = sorted(blue_votes.items(), key=lambda x: x[1], reverse=True)
        
        # 选择最终预测结果，确保分布合理
        ensemble_red = self._smart_select_numbers(red_candidates, red_count, red_range)
        ensemble_blue = self._smart_select_numbers(blue_candidates, blue_count, blue_range)
        
        # 计算智能集成置信度
        ensemble_confidence = self._calculate_smart_confidence(
            predictions, weights, confidences
        )
        
        return {
            'red_balls': sorted(ensemble_red),
            'blue_balls': ensemble_blue,
            'confidence': round(ensemble_confidence, 3),
            'red_votes': dict(red_candidates),
            'blue_votes': dict(blue_candidates),
            'smart_factors': {k: self._calculate_smart_factor(v, lottery_type) 
                            for k, v in predictions.items()}
        }
    
    def _calculate_smart_factor(self, prediction: Dict, lottery_type: str) -> float:
        """计算智能因子，评估预测的合理性"""
        try:
            red_balls = prediction['red_balls']
            blue_balls = prediction['blue_balls']
            
            smart_factor = 1.0
            
            # 1. 分布合理性检查
            if len(red_balls) > 1:
                red_range = max(red_balls) - min(red_balls)
                expected_range = 20 if lottery_type == '双色球' else 22
                
                # 分布过于集中或过于分散都会降低智能因子
                range_ratio = red_range / expected_range
                if 0.3 <= range_ratio <= 1.5:
                    smart_factor *= 1.1  # 合理分布加分
                else:
                    smart_factor *= 0.9  # 不合理分布减分
            
            # 2. 奇偶比例检查
            odd_count = sum(1 for num in red_balls if num % 2 == 1)
            expected_odd = len(red_balls) // 2
            
            if abs(odd_count - expected_odd) <= 1:
                smart_factor *= 1.05  # 合理奇偶比例加分
            
            # 3. 连号检查
            sorted_reds = sorted(red_balls)
            consecutive_count = 0
            for i in range(len(sorted_reds) - 1):
                if sorted_reds[i+1] - sorted_reds[i] == 1:
                    consecutive_count += 1
            
            if consecutive_count <= 2:  # 适量连号
                smart_factor *= 1.02
            elif consecutive_count > 4:  # 过多连号
                smart_factor *= 0.95
            
            return max(0.5, min(1.5, smart_factor))
            
        except Exception as e:
            logger.warning(f"计算智能因子失败: {e}")
            return 1.0
    
    def _smart_select_numbers(self, candidates: List[Tuple[int, float]], 
                            count: int, number_range: Tuple[int, int]) -> List[int]:
        """智能选择号码，确保分布合理"""
        try:
            selected = []
            used_numbers = set()
            
            # 首先选择票数最高的号码
            for num, votes in candidates:
                if len(selected) >= count:
                    break
                if num not in used_numbers:
                    selected.append(num)
                    used_numbers.add(num)
            
            # 如果数量不足，智能补充
            while len(selected) < count:
                # 寻找未被选中但合理的号码
                for num in range(number_range[0], number_range[1] + 1):
                    if num not in used_numbers:
                        # 检查添加这个号码是否合理
                        temp_selected = selected + [num]
                        if self._is_reasonable_combination(temp_selected):
                            selected.append(num)
                            used_numbers.add(num)
                            break
                else:
                    # 如果找不到合理的号码，随机选择
                    remaining = [n for n in range(number_range[0], number_range[1] + 1) 
                               if n not in used_numbers]
                    if remaining:
                        import random
                        selected.append(random.choice(remaining))
                        used_numbers.add(selected[-1])
                    else:
                        break
            
            return selected[:count]
            
        except Exception as e:
            logger.warning(f"智能选择号码失败: {e}")
            return [candidates[i][0] for i in range(min(count, len(candidates)))]
    
    def _is_reasonable_combination(self, numbers: List[int]) -> bool:
        """检查号码组合是否合理"""
        try:
            if len(numbers) <= 1:
                return True
            
            # 检查分布是否过于集中
            num_range = max(numbers) - min(numbers)
            if num_range < len(numbers):  # 过于集中
                return False
            
            # 检查是否有太多连号
            sorted_nums = sorted(numbers)
            consecutive_count = 0
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i+1] - sorted_nums[i] == 1:
                    consecutive_count += 1
            
            if consecutive_count > len(numbers) // 2:  # 连号过多
                return False
            
            return True
            
        except:
            return True
    
    def _calculate_smart_confidence(self, predictions: Dict, weights: Dict, 
                                  confidences: Dict) -> float:
        """计算智能集成置信度"""
        try:
            # 基础置信度：加权平均
            base_confidence = sum(confidences[k] * weights[k] for k in predictions.keys()) / len(predictions)
            
            # 一致性奖励：如果多个模型预测相似，增加置信度
            consistency_bonus = self._calculate_consistency_bonus(predictions)
            
            # 多样性惩罚：如果模型过于相似，降低置信度
            diversity_factor = self._calculate_diversity_factor(predictions)
            
            smart_confidence = base_confidence * (1 + consistency_bonus) * diversity_factor
            
            return max(0.1, min(1.0, smart_confidence))
            
        except Exception as e:
            logger.warning(f"计算智能置信度失败: {e}")
            return 0.5
    
    def _calculate_consistency_bonus(self, predictions: Dict) -> float:
        """计算一致性奖励"""
        try:
            if len(predictions) < 2:
                return 0
            
            # 计算预测之间的相似度
            similarities = []
            pred_list = list(predictions.values())
            
            for i in range(len(pred_list)):
                for j in range(i + 1, len(pred_list)):
                    red_overlap = len(set(pred_list[i]['red_balls']) & set(pred_list[j]['red_balls']))
                    blue_overlap = len(set(pred_list[i]['blue_balls']) & set(pred_list[j]['blue_balls']))
                    
                    total_overlap = red_overlap + blue_overlap
                    max_overlap = len(pred_list[i]['red_balls']) + len(pred_list[i]['blue_balls'])
                    
                    similarity = total_overlap / max_overlap if max_overlap > 0 else 0
                    similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            # 适中的相似度获得最高奖励
            if 0.3 <= avg_similarity <= 0.7:
                return 0.1  # 10%奖励
            else:
                return 0  # 无奖励
                
        except:
            return 0
    
    def _calculate_diversity_factor(self, predictions: Dict) -> float:
        """计算多样性因子"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            # 检查模型类型多样性
            model_types = set()
            for model_key in predictions.keys():
                if 'RandomForest' in model_key:
                    model_types.add('RandomForest')
                elif 'XGBoost' in model_key:
                    model_types.add('XGBoost')
                elif 'LSTM' in model_key:
                    model_types.add('LSTM')
                elif 'Advanced' in model_key:
                    model_types.add('Advanced')
            
            diversity_factor = min(1.2, 1.0 + len(model_types) * 0.05)  # 最多20%奖励
            
            return diversity_factor
            
        except:
            return 1.0
    
    def save_model(self, model_key: str) -> bool:
        """
        保存指定模型
        
        Args:
            model_key: 模型键名
            
        Returns:
            保存是否成功
        """
        try:
            if model_key not in self.models:
                logger.error(f"模型不存在: {model_key}")
                return False
            
            model = self.models[model_key]
            filepath = os.path.join(self.models_dir, f"{model_key}.pkl")
            
            success = model.save_model(filepath)
            
            if success:
                # 保存模型配置
                config_path = os.path.join(self.models_dir, f"{model_key}_config.json")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.model_configs[model_key], f, 
                            ensure_ascii=False, indent=2)
                
                logger.info(f"模型保存成功: {model_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, model_key: str) -> bool:
        """
        加载指定模型
        
        Args:
            model_key: 模型键名
            
        Returns:
            加载是否成功
        """
        try:
            filepath = os.path.join(self.models_dir, f"{model_key}.pkl")
            config_path = os.path.join(self.models_dir, f"{model_key}_config.json")
            
            if not os.path.exists(filepath):
                logger.error(f"模型文件不存在: {filepath}")
                return False
            
            if not os.path.exists(config_path):
                logger.error(f"模型配置文件不存在: {config_path}")
                return False
            
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 创建模型实例
            model_type = config['model_type']
            lottery_type = config['lottery_type']
            parameters = config.get('parameters', {})
            
            model = self.create_model(model_type, lottery_type, **parameters)
            if model is None:
                return False
            
            # 加载模型数据
            success = model.load_model(filepath)
            
            if success:
                self.model_configs[model_key] = config
                logger.info(f"模型加载成功: {model_key}")
            else:
                # 如果加载失败，从管理器中移除
                if model_key in self.models:
                    del self.models[model_key]
                if model_key in self.model_configs:
                    del self.model_configs[model_key]
            
            return success
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def list_models(self) -> Dict[str, Dict]:
        """
        列出所有模型及其状态
        
        Returns:
            模型状态字典
        """
        model_status = {}
        
        for model_key, config in self.model_configs.items():
            model = self.models.get(model_key)
            
            status = {
                'model_type': config['model_type'],
                'lottery_type': config['lottery_type'],
                'created_at': config['created_at'],
                'trained': config.get('trained', False),
                'loaded': model is not None,
                'is_trained': model.is_trained if model else False
            }
            
            if config.get('trained_at'):
                status['trained_at'] = config['trained_at']
            
            if config.get('training_data_size'):
                status['training_data_size'] = config['training_data_size']
            
            if model and hasattr(model, 'performance_metrics'):
                status['performance_metrics'] = model.performance_metrics
            
            model_status[model_key] = status
        
        return model_status
    
    def delete_model(self, model_key: str) -> bool:
        """
        删除指定模型
        
        Args:
            model_key: 模型键名
            
        Returns:
            删除是否成功
        """
        try:
            # 从内存中删除
            if model_key in self.models:
                del self.models[model_key]
            
            if model_key in self.model_configs:
                del self.model_configs[model_key]
            
            # 删除文件
            filepath = os.path.join(self.models_dir, f"{model_key}.pkl")
            config_path = os.path.join(self.models_dir, f"{model_key}_config.json")
            
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if os.path.exists(config_path):
                os.remove(config_path)
            
            # 删除LSTM相关文件
            red_model_path = filepath.replace('.pkl', '_red_model.h5')
            blue_model_path = filepath.replace('.pkl', '_blue_model.h5')
            
            if os.path.exists(red_model_path):
                os.remove(red_model_path)
            
            if os.path.exists(blue_model_path):
                os.remove(blue_model_path)
            
            logger.info(f"模型删除成功: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"模型删除失败: {e}")
            return False
    
    def get_model(self, model_key: str) -> Optional[BasePredictor]:
        """
        获取指定模型实例
        
        Args:
            model_key: 模型键名
            
        Returns:
            模型实例
        """
        return self.models.get(model_key)
    
    def auto_load_models(self) -> int:
        """
        自动加载模型目录中的所有模型
        
        Returns:
            成功加载的模型数量
        """
        loaded_count = 0
        
        try:
            if not os.path.exists(self.models_dir):
                return 0
            
            # 查找所有配置文件
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_config.json'):
                    model_key = filename.replace('_config.json', '')
                    
                    if self.load_model(model_key):
                        loaded_count += 1
            
            logger.info(f"自动加载完成，成功加载 {loaded_count} 个模型")
            return loaded_count
            
        except Exception as e:
            logger.error(f"自动加载模型失败: {e}")
            return loaded_count
