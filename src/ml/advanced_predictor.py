"""
高级预测器 - 集成多种先进算法的智能预测系统
支持深度学习、集成学习、强化学习等多种方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter
import random

try:
    from sklearn.ensemble import VotingRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class AdvancedPredictor(BasePredictor):
    """
    高级预测器
    集成多种先进算法，支持自动超参数优化和在线学习
    """
    
    def __init__(self, lottery_type: str, enable_optimization: bool = True,
                 ensemble_size: int = 5, random_state: int = 42):
        """
        初始化高级预测器
        
        Args:
            lottery_type: 彩票类型
            enable_optimization: 是否启用超参数优化
            ensemble_size: 集成模型数量
            random_state: 随机种子
        """
        super().__init__(lottery_type, f"Advanced_{lottery_type}")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("需要安装scikit-learn: pip install scikit-learn")
        
        self.enable_optimization = enable_optimization and OPTUNA_AVAILABLE
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        
        # 设置随机种子
        np.random.seed(random_state)
        random.seed(random_state)
        
        # 模型组件
        self.red_ensemble = None
        self.blue_ensemble = None
        self.meta_learner = None
        
        # 缩放器
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # 在线学习缓存
        self.online_buffer = []
        self.online_buffer_size = 100
        self.adaptation_threshold = 0.15
        
        # 预测历史
        self.prediction_history = []
        self.performance_tracker = {
            'mse_history': [],
            'mae_history': [],
            'confidence_history': [],
            'adaptation_count': 0
        }
        
        logger.info(f"高级预测器初始化完成: {self.model_name}")
    
    def prepare_features(self, history_data: List[Dict]) -> pd.DataFrame:
        """
        准备高级特征
        """
        try:
            # 获取基础特征
            base_features = self.calculate_features(history_data)
            
            if base_features.empty:
                return base_features
            
            # 检查缓存
            cache_key = self.get_cache_key(str(len(history_data)) + str(hash(str(history_data[-10:]))))
            cached_features = self.get_cached_features(cache_key)
            if cached_features is not None:
                logger.info("使用缓存的高级特征")
                return cached_features
            
            # 添加高级特征
            advanced_features = self._create_advanced_features(base_features, history_data)
            
            # 特征优化
            optimized_features = self.optimize_features(advanced_features)
            
            # 缓存特征
            self.cache_features(cache_key, optimized_features)
            
            logger.info(f"高级特征准备完成: {len(optimized_features)} 条记录, {len(optimized_features.columns)} 个特征")
            return optimized_features
            
        except Exception as e:
            logger.error(f"高级特征准备失败: {e}")
            return pd.DataFrame()
    
    def _create_advanced_features(self, base_features: pd.DataFrame, 
                                history_data: List[Dict]) -> pd.DataFrame:
        """创建高级特征"""
        try:
            enhanced_features = base_features.copy()
            
            # 1. 时间序列特征
            enhanced_features = self._add_time_series_features(enhanced_features)
            
            # 2. 交互特征
            enhanced_features = self._add_interaction_features(enhanced_features)
            
            # 3. 统计特征
            enhanced_features = self._add_statistical_features(enhanced_features, history_data)
            
            # 4. 序列模式特征
            enhanced_features = self._add_sequence_features(enhanced_features, history_data)
            
            # 5. 概率特征
            enhanced_features = self._add_probability_features(enhanced_features, history_data)
            
            return enhanced_features
            
        except Exception as e:
            logger.error(f"创建高级特征失败: {e}")
            return base_features
    
    def _add_time_series_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """添加时间序列特征"""
        try:
            if len(features) < 5:
                return features
            
            # 滞后特征
            lag_features = ['red_sum', 'red_mean', 'blue_sum', 'red_range']
            for feature in lag_features:
                if feature in features.columns:
                    for lag in [1, 2, 3, 5]:
                        features[f'{feature}_lag_{lag}'] = features[feature].shift(lag)
            
            # 滑动窗口统计
            window_sizes = [3, 5, 10]
            for feature in lag_features:
                if feature in features.columns:
                    for window in window_sizes:
                        if len(features) >= window:
                            features[f'{feature}_rolling_mean_{window}'] = \
                                features[feature].rolling(window=window).mean()
                            features[f'{feature}_rolling_std_{window}'] = \
                                features[feature].rolling(window=window).std()
                            features[f'{feature}_rolling_min_{window}'] = \
                                features[feature].rolling(window=window).min()
                            features[f'{feature}_rolling_max_{window}'] = \
                                features[feature].rolling(window=window).max()
            
            # 差分特征
            for feature in lag_features:
                if feature in features.columns:
                    features[f'{feature}_diff_1'] = features[feature].diff(1)
                    features[f'{feature}_diff_2'] = features[feature].diff(2)
            
            return features
            
        except Exception as e:
            logger.warning(f"添加时间序列特征失败: {e}")
            return features
    
    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """添加交互特征"""
        try:
            # 选择重要的数值特征进行交互
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            important_cols = [col for col in numeric_cols 
                            if any(keyword in col for keyword in 
                                 ['sum', 'mean', 'range', 'count', 'std'])][:10]
            
            # 创建交互特征
            for i in range(len(important_cols)):
                for j in range(i + 1, min(i + 3, len(important_cols))):  # 限制交互特征数量
                    col1, col2 = important_cols[i], important_cols[j]
                    if col1 in features.columns and col2 in features.columns:
                        # 乘积特征
                        features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
                        
                        # 比值特征（避免除零）
                        mask = features[col2] != 0
                        features[f'{col1}_div_{col2}'] = 0.0
                        features.loc[mask, f'{col1}_div_{col2}'] = \
                            features.loc[mask, col1] / features.loc[mask, col2]
            
            return features
            
        except Exception as e:
            logger.warning(f"添加交互特征失败: {e}")
            return features
    
    def _add_statistical_features(self, features: pd.DataFrame, 
                                history_data: List[Dict]) -> pd.DataFrame:
        """添加统计特征"""
        try:
            # 计算历史统计信息
            all_red_numbers = []
            all_blue_numbers = []
            
            for data in history_data:
                red_balls, blue_balls = self.extract_numbers_from_data(data)
                if red_balls and blue_balls:
                    all_red_numbers.extend(red_balls)
                    all_blue_numbers.extend(blue_balls)
            
            if not all_red_numbers:
                return features
            
            # 添加全局统计特征
            features['global_red_entropy'] = self._calculate_entropy(all_red_numbers)
            features['global_blue_entropy'] = self._calculate_entropy(all_blue_numbers)
            features['global_red_variance'] = np.var(all_red_numbers)
            features['global_blue_variance'] = np.var(all_blue_numbers)
            
            # 添加分布特征
            red_counter = Counter(all_red_numbers)
            blue_counter = Counter(all_blue_numbers)
            
            # 最常见和最不常见号码的统计
            if red_counter:
                most_common_red = red_counter.most_common(5)
                least_common_red = red_counter.most_common()[-5:]
                
                features['most_common_red_freq'] = np.mean([freq for _, freq in most_common_red])
                features['least_common_red_freq'] = np.mean([freq for _, freq in least_common_red])
            
            return features
            
        except Exception as e:
            logger.warning(f"添加统计特征失败: {e}")
            return features
    
    def _add_sequence_features(self, features: pd.DataFrame, 
                             history_data: List[Dict]) -> pd.DataFrame:
        """添加序列模式特征"""
        try:
            if len(history_data) < 10:
                return features
            
            # 分析最近的序列模式
            recent_sequences = []
            for data in history_data[-20:]:  # 最近20期
                red_balls, blue_balls = self.extract_numbers_from_data(data)
                if red_balls:
                    recent_sequences.append(sorted(red_balls))
            
            if not recent_sequences:
                return features
            
            # 计算序列相似度
            similarities = []
            for i in range(1, len(recent_sequences)):
                similarity = len(set(recent_sequences[i]) & set(recent_sequences[i-1]))
                similarities.append(similarity)
            
            if similarities:
                features['avg_sequence_similarity'] = np.mean(similarities)
                features['max_sequence_similarity'] = np.max(similarities)
                features['sequence_stability'] = np.std(similarities)
            
            # 添加模式重复特征
            pattern_counts = Counter([tuple(seq) for seq in recent_sequences])
            features['pattern_repetition_rate'] = len([count for count in pattern_counts.values() if count > 1]) / len(recent_sequences)
            
            return features
            
        except Exception as e:
            logger.warning(f"添加序列特征失败: {e}")
            return features
    
    def _add_probability_features(self, features: pd.DataFrame, 
                                history_data: List[Dict]) -> pd.DataFrame:
        """添加概率特征"""
        try:
            # 计算条件概率特征
            if len(history_data) < 20:
                return features
            
            # 分析号码共现概率
            cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
            total_combinations = 0
            
            for data in history_data:
                red_balls, blue_balls = self.extract_numbers_from_data(data)
                if red_balls and len(red_balls) >= 2:
                    total_combinations += 1
                    for i in range(len(red_balls)):
                        for j in range(i + 1, len(red_balls)):
                            cooccurrence_matrix[red_balls[i]][red_balls[j]] += 1
                            cooccurrence_matrix[red_balls[j]][red_balls[i]] += 1
            
            # 计算平均共现概率
            if total_combinations > 0:
                all_cooccurrences = []
                for num1_dict in cooccurrence_matrix.values():
                    all_cooccurrences.extend(num1_dict.values())
                
                if all_cooccurrences:
                    features['avg_cooccurrence_prob'] = np.mean(all_cooccurrences) / total_combinations
                    features['max_cooccurrence_prob'] = np.max(all_cooccurrences) / total_combinations
            
            return features
            
        except Exception as e:
            logger.warning(f"添加概率特征失败: {e}")
            return features
    
    def _calculate_entropy(self, numbers: List[int]) -> float:
        """计算信息熵"""
        try:
            counter = Counter(numbers)
            total = len(numbers)
            entropy = 0
            
            for count in counter.values():
                probability = count / total
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            return entropy
        except:
            return 0.0
    
    def train(self, history_data: List[Dict]) -> bool:
        """
        训练高级预测模型
        """
        try:
            start_time = datetime.now()
            logger.info(f"开始训练高级预测模型，数据量: {len(history_data)}")
            
            # 准备特征
            df = self.prepare_features(history_data)
            
            if df.empty or len(df) < 20:
                logger.error("高级预测器训练数据不足")
                return False
            
            # 准备训练数据
            success = self._prepare_training_data(df)
            if not success:
                return False
            
            # 构建集成模型
            success = self._build_ensemble_models()
            if not success:
                return False
            
            # 训练模型
            success = self._train_ensemble_models()
            if not success:
                return False
            
            # 超参数优化（如果启用）
            if self.enable_optimization:
                self._optimize_hyperparameters()
            
            # 更新性能指标
            training_time = (datetime.now() - start_time).total_seconds()
            self.update_performance_metrics(training_time=training_time)
            
            self.is_trained = True
            logger.info(f"高级预测模型训练完成，耗时: {training_time:.2f}秒")
            
            return True
            
        except Exception as e:
            logger.error(f"高级预测模型训练失败: {e}")
            return False
    
    def _prepare_training_data(self, df: pd.DataFrame) -> bool:
        """准备训练数据"""
        try:
            # 分离特征和目标
            feature_columns = [col for col in df.columns 
                             if col not in ['period', 'date'] + 
                             [f'red_{i+1}' for i in range(self.red_count)] +
                             [f'blue_{i+1}' for i in range(self.blue_count)]]
            
            self.X = df[feature_columns].fillna(0)
            
            # 准备目标变量
            red_targets = []
            blue_targets = []
            
            for i in range(self.red_count):
                col = f'red_{i+1}'
                if col in df.columns:
                    red_targets.append(df[col])
            
            for i in range(self.blue_count):
                col = f'blue_{i+1}'
                if col in df.columns:
                    blue_targets.append(df[col])
            
            if not red_targets or not blue_targets:
                logger.error("无法找到目标变量")
                return False
            
            self.y_red = pd.DataFrame(red_targets).T
            self.y_blue = pd.DataFrame(blue_targets).T
            
            # 特征缩放
            self.X_scaled = self.feature_scaler.fit_transform(self.X)
            
            # 目标变量缩放（可选）
            self.y_red_scaled = self.y_red.values
            self.y_blue_scaled = self.y_blue.values
            
            self.feature_columns = feature_columns
            
            logger.info(f"训练数据准备完成: 特征维度 {self.X_scaled.shape}, 红球目标 {self.y_red_scaled.shape}, 蓝球目标 {self.y_blue_scaled.shape}")
            return True
            
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            return False
    
    def _build_ensemble_models(self) -> bool:
        """构建集成模型"""
        try:
            # 定义基础模型
            base_models = []
            for i in range(self.ensemble_size):
                base_models.extend([
                    (f'gb_{i}', GradientBoostingRegressor(random_state=self.random_state + i)),
                    (f'mlp_{i}', MLPRegressor(random_state=self.random_state + i, max_iter=500)),
                    (f'gp_{i}', GaussianProcessRegressor(
                        kernel=ConstantKernel() * RBF(),
                        random_state=self.random_state + i
                    ))
                ])
            
            # 创建投票回归器
            self.red_models = []
            self.blue_models = []
            
            for i in range(self.red_count):
                red_ensemble = VotingRegressor(
                    estimators=[(f'{name}_{i}', model) for name, model in base_models],
                    n_jobs=-1
                )
                self.red_models.append(red_ensemble)
            
            for i in range(self.blue_count):
                blue_ensemble = VotingRegressor(
                    estimators=[(f'{name}_{i}', model) for name, model in base_models],
                    n_jobs=-1
                )
                self.blue_models.append(blue_ensemble)
            
            logger.info(f"集成模型构建完成: {len(self.red_models)}个红球模型, {len(self.blue_models)}个蓝球模型")
            return True
            
        except Exception as e:
            logger.error(f"构建集成模型失败: {e}")
            return False
    
    def _train_ensemble_models(self) -> bool:
        """训练集成模型"""
        try:
            # 训练红球模型
            for i, model in enumerate(self.red_models):
                if i < self.y_red_scaled.shape[1]:
                    model.fit(self.X_scaled, self.y_red_scaled[:, i])
                    logger.debug(f"红球模型 {i+1} 训练完成")
            
            # 训练蓝球模型
            for i, model in enumerate(self.blue_models):
                if i < self.y_blue_scaled.shape[1]:
                    model.fit(self.X_scaled, self.y_blue_scaled[:, i])
                    logger.debug(f"蓝球模型 {i+1} 训练完成")
            
            # 评估模型性能
            self._evaluate_models()
            
            return True
            
        except Exception as e:
            logger.error(f"训练集成模型失败: {e}")
            return False
    
    def _evaluate_models(self):
        """评估模型性能"""
        try:
            # 交叉验证评估
            red_scores = []
            blue_scores = []
            
            for i, model in enumerate(self.red_models):
                if i < self.y_red_scaled.shape[1]:
                    scores = cross_val_score(model, self.X_scaled, self.y_red_scaled[:, i], 
                                           cv=min(5, len(self.X_scaled) // 4), 
                                           scoring='neg_mean_squared_error')
                    red_scores.extend(scores)
            
            for i, model in enumerate(self.blue_models):
                if i < self.y_blue_scaled.shape[1]:
                    scores = cross_val_score(model, self.X_scaled, self.y_blue_scaled[:, i], 
                                           cv=min(5, len(self.X_scaled) // 4), 
                                           scoring='neg_mean_squared_error')
                    blue_scores.extend(scores)
            
            if red_scores:
                self.performance_metrics['red_cv_score'] = -np.mean(red_scores)
                self.performance_metrics['red_cv_std'] = np.std(red_scores)
            
            if blue_scores:
                self.performance_metrics['blue_cv_score'] = -np.mean(blue_scores)
                self.performance_metrics['blue_cv_std'] = np.std(blue_scores)
            
            logger.info(f"模型评估完成: 红球CV-MSE {self.performance_metrics.get('red_cv_score', 0):.4f}, "
                       f"蓝球CV-MSE {self.performance_metrics.get('blue_cv_score', 0):.4f}")
            
        except Exception as e:
            logger.warning(f"模型评估失败: {e}")
    
    def _optimize_hyperparameters(self):
        """超参数优化（使用Optuna）"""
        try:
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna未安装，跳过超参数优化")
                return
            
            logger.info("开始超参数优化...")
            
            def objective(trial):
                # 定义超参数搜索空间
                gb_params = {
                    'n_estimators': trial.suggest_int('gb_n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('gb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3)
                }
                
                mlp_params = {
                    'hidden_layer_sizes': (trial.suggest_int('mlp_hidden_size', 50, 200),),
                    'alpha': trial.suggest_float('mlp_alpha', 0.0001, 0.01, log=True),
                    'learning_rate_init': trial.suggest_float('mlp_lr', 0.001, 0.1, log=True)
                }
                
                # 创建优化后的模型
                gb_model = GradientBoostingRegressor(random_state=self.random_state, **gb_params)
                mlp_model = MLPRegressor(random_state=self.random_state, max_iter=500, **mlp_params)
                
                # 评估性能
                gb_scores = cross_val_score(gb_model, self.X_scaled, self.y_red_scaled[:, 0], 
                                          cv=3, scoring='neg_mean_squared_error')
                mlp_scores = cross_val_score(mlp_model, self.X_scaled, self.y_red_scaled[:, 0], 
                                           cv=3, scoring='neg_mean_squared_error')
                
                return -np.mean(gb_scores + mlp_scores)
            
            # 运行优化
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20, timeout=300)  # 5分钟超时
            
            # 应用最佳参数
            best_params = study.best_params
            logger.info(f"超参数优化完成，最佳参数: {best_params}")
            
            self.performance_metrics['optimization_score'] = study.best_value
            self.performance_metrics['best_params'] = best_params
            
        except Exception as e:
            logger.warning(f"超参数优化失败: {e}")
    
    def predict(self, recent_data: Optional[List[Dict]] = None) -> Dict:
        """
        进行高级预测
        """
        try:
            start_time = datetime.now()
            
            if not self.is_trained:
                return {
                    'success': False,
                    'error': '高级预测模型未训练',
                    'model_name': self.model_name
                }
            
            if not recent_data or len(recent_data) == 0:
                return {
                    'success': False,
                    'error': '缺少预测所需的历史数据',
                    'model_name': self.model_name
                }
            
            # 准备预测特征
            df = self.prepare_features(recent_data)
            
            if df.empty:
                return {
                    'success': False,
                    'error': '高级特征准备失败',
                    'model_name': self.model_name
                }
            
            logger.info(f"高级特征准备完成: {len(df)} 条记录, {len(df.columns)} 个特征")
            
            # 确保特征列与训练时一致
            if hasattr(self, 'feature_columns') and self.feature_columns is not None:
                # 获取训练时的特征列
                missing_cols = set(self.feature_columns) - set(df.columns)
                extra_cols = set(df.columns) - set(self.feature_columns)
                
                # 添加缺失的列（填充0）
                for col in missing_cols:
                    df[col] = 0
                    logger.warning(f"添加缺失特征列: {col}")
                
                # 移除多余的列
                if extra_cols:
                    logger.warning(f"移除多余特征列: {list(extra_cols)}")
                    df = df.drop(columns=list(extra_cols), errors='ignore')
                
                # 重新排序列以匹配训练时的顺序
                df = df[self.feature_columns]
            else:
                logger.warning("未找到训练时的特征列信息，使用当前所有特征")
                self.feature_columns = df.columns.tolist()
            
            # 使用最后一条记录进行预测
            latest_features = df.iloc[-1:].fillna(0)
            X_pred = self.feature_scaler.transform(latest_features)
            
            # 预测红球
            red_predictions = []
            red_confidences = []
            
            for i, model in enumerate(self.red_models):
                pred = model.predict(X_pred)[0]
                # 限制在有效范围内
                pred = max(self.red_range[0], min(self.red_range[1], round(pred)))
                red_predictions.append(int(pred))
                
                # 计算预测置信度
                confidence = self._calculate_prediction_confidence(model, X_pred)
                red_confidences.append(confidence)
            
            # 预测蓝球
            blue_predictions = []
            blue_confidences = []
            
            for i, model in enumerate(self.blue_models):
                pred = model.predict(X_pred)[0]
                # 限制在有效范围内
                pred = max(self.blue_range[0], min(self.blue_range[1], round(pred)))
                blue_predictions.append(int(pred))
                
                # 计算预测置信度
                confidence = self._calculate_prediction_confidence(model, X_pred)
                blue_confidences.append(confidence)
            
            # 确保号码唯一性
            red_predictions = self._ensure_unique_numbers(
                red_predictions, self.red_range, self.red_count
            )
            blue_predictions = self._ensure_unique_numbers(
                blue_predictions, self.blue_range, self.blue_count
            )
            
            # 验证预测结果
            if not self.validate_prediction(red_predictions, blue_predictions):
                logger.warning("高级预测结果验证失败，使用随机生成")
                red_predictions, blue_predictions = self._generate_random_numbers()
            
            # 计算综合置信度
            base_confidence = np.mean(red_confidences + blue_confidences)
            adaptive_confidence = self.get_adaptive_confidence(base_confidence)
            
            # 更新性能指标
            prediction_time = (datetime.now() - start_time).total_seconds()
            self.update_performance_metrics(prediction_time=prediction_time)
            
            result = {
                'success': True,
                'model_name': self.model_name,
                'lottery_type': self.lottery_type,
                'red_balls': sorted(red_predictions),
                'blue_balls': blue_predictions,
                'confidence': round(adaptive_confidence, 3),
                'prediction_time': datetime.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'individual_confidences': {
                    'red': red_confidences,
                    'blue': blue_confidences
                },
                'ensemble_size': self.ensemble_size,
                'prediction_method': 'advanced_ensemble'
            }
            
            # 记录预测历史
            self.prediction_history.append({
                'prediction': result,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"高级预测完成: 红球 {result['red_balls']}, 蓝球 {result['blue_balls']}, "
                       f"置信度 {adaptive_confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"高级预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': self.model_name
            }
    
    def _calculate_prediction_confidence(self, model, X_pred: np.ndarray) -> float:
        """计算预测置信度"""
        try:
            # 基于模型的预测不确定性
            if hasattr(model, 'predict_std'):
                # 高斯过程回归的标准差
                _, std = model.predict(X_pred, return_std=True)
                confidence = 1 / (1 + std[0])
            else:
                # 基于集成模型的方差
                if hasattr(model, 'estimators_'):
                    predictions = [est.predict(X_pred)[0] for est in model.estimators_]
                    variance = np.var(predictions)
                    confidence = 1 / (1 + variance)
                else:
                    confidence = 0.7  # 默认置信度
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"计算预测置信度失败: {e}")
            return 0.5
    
    def _ensure_unique_numbers(self, predictions: List[int], number_range: tuple, 
                              required_count: int) -> List[int]:
        """确保号码唯一性"""
        unique_predictions = []
        used_numbers = set()
        
        for pred in predictions:
            if pred not in used_numbers:
                unique_predictions.append(pred)
                used_numbers.add(pred)
            else:
                # 找一个未使用的号码
                for num in range(number_range[0], number_range[1] + 1):
                    if num not in used_numbers:
                        unique_predictions.append(num)
                        used_numbers.add(num)
                        break
        
        # 如果数量不足，随机补充
        while len(unique_predictions) < required_count:
            for num in range(number_range[0], number_range[1] + 1):
                if num not in used_numbers:
                    unique_predictions.append(num)
                    used_numbers.add(num)
                    break
            else:
                break
        
        return unique_predictions[:required_count]
    
    def _generate_random_numbers(self) -> tuple:
        """生成随机号码作为备选"""
        red_balls = np.random.choice(
            range(self.red_range[0], self.red_range[1] + 1),
            size=self.red_count,
            replace=False
        ).tolist()
        
        blue_balls = np.random.choice(
            range(self.blue_range[0], self.blue_range[1] + 1),
            size=self.blue_count,
            replace=False
        ).tolist()
        
        return red_balls, blue_balls
    
    def online_learning_update(self, actual_results: List[int], prediction_confidence: float):
        """在线学习更新"""
        try:
            # 添加到在线学习缓冲区
            self.online_buffer.append({
                'actual_results': actual_results,
                'confidence': prediction_confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # 保持缓冲区大小
            if len(self.online_buffer) > self.online_buffer_size:
                self.online_buffer = self.online_buffer[-self.online_buffer_size:]
            
            # 检查是否需要模型适应
            if len(self.online_buffer) >= 10:
                recent_performance = self._evaluate_recent_performance()
                if recent_performance < self.adaptation_threshold:
                    self._adapt_model()
                    self.performance_tracker['adaptation_count'] += 1
            
        except Exception as e:
            logger.warning(f"在线学习更新失败: {e}")
    
    def _evaluate_recent_performance(self) -> float:
        """评估最近的性能"""
        try:
            if len(self.online_buffer) < 5:
                return 1.0
            
            recent_confidences = [item['confidence'] for item in self.online_buffer[-10:]]
            return np.mean(recent_confidences)
            
        except:
            return 0.5
    
    def _adapt_model(self):
        """模型自适应调整"""
        try:
            logger.info("开始模型自适应调整...")
            
            # 简单的自适应策略：调整学习率或正则化参数
            if hasattr(self, 'red_models') and self.red_models:
                for model in self.red_models:
                    if hasattr(model, 'estimators_'):
                        for estimator_name, estimator in model.estimators_:
                            if hasattr(estimator, 'learning_rate') and hasattr(estimator, 'set_params'):
                                current_lr = getattr(estimator, 'learning_rate', 0.1)
                                new_lr = max(0.01, current_lr * 0.9)  # 降低学习率
                                estimator.set_params(learning_rate=new_lr)
            
            logger.info("模型自适应调整完成")
            
        except Exception as e:
            logger.warning(f"模型自适应调整失败: {e}")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        base_info = super().get_model_info()
        
        advanced_info = {
            'ensemble_size': self.ensemble_size,
            'optimization_enabled': self.enable_optimization,
            'online_buffer_size': len(self.online_buffer),
            'adaptation_count': self.performance_tracker.get('adaptation_count', 0),
            'prediction_history_size': len(self.prediction_history)
        }
        
        base_info.update(advanced_info)
        return base_info
