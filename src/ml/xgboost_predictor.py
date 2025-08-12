"""
XGBoost预测器 - 基于XGBoost算法的彩票预测
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
import joblib
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost未安装，将使用LightGBM作为替代")

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class XGBoostPredictor(BasePredictor):
    """
    XGBoost预测器
    使用XGBoost梯度提升算法进行彩票号码预测
    """
    
    def __init__(self, lottery_type: str, max_depth: int = 6, learning_rate: float = 0.1,
                 n_estimators: int = 100, random_state: int = 42):
        """
        初始化XGBoost预测器
        
        Args:
            lottery_type: 彩票类型
            max_depth: 树的最大深度
            learning_rate: 学习率
            n_estimators: 树的数量
            random_state: 随机种子
        """
        super().__init__(lottery_type, f"XGBoost_{lottery_type}")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装，请先安装: pip install xgboost")
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # 为每个号码位置创建独立的模型
        self.red_models = []
        self.blue_models = []
        
        # 模型性能指标
        self.performance_metrics = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        # 红球模型
        for i in range(self.red_count):
            model = xgb.XGBRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                random_state=self.random_state + i,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                objective='reg:squarederror'
            )
            self.red_models.append(model)
        
        # 蓝球模型
        for i in range(self.blue_count):
            model = xgb.XGBRegressor(
                max_depth=self.max_depth - 1,  # 蓝球范围较小，使用较浅的树
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                random_state=self.random_state + 100 + i,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                objective='reg:squarederror'
            )
            self.blue_models.append(model)
    
    def prepare_features(self, history_data: List[Dict]) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            特征DataFrame
        """
        try:
            # 使用基类的特征计算方法
            df = self.calculate_features(history_data)
            
            if df.empty:
                return df
            
            # 添加XGBoost特定的特征
            df = self._add_xgboost_features(df)
            
            # 添加滞后特征
            df = self._add_lag_features(df)
            
            # 添加差分特征
            df = self._add_diff_features(df)
            
            logger.info(f"XGBoost特征准备完成，共 {len(df)} 条记录，{len(df.columns)} 个特征")
            return df
            
        except Exception as e:
            logger.error(f"XGBoost特征准备失败: {e}")
            return pd.DataFrame()
    
    def _add_xgboost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加XGBoost特定特征"""
        if len(df) < 2:
            return df
        
        # 指数移动平均
        for alpha in [0.1, 0.3, 0.5]:
            df[f'red_sum_ema_{alpha}'] = df['red_sum'].ewm(alpha=alpha).mean()
            df[f'red_mean_ema_{alpha}'] = df['red_mean'].ewm(alpha=alpha).mean()
            df[f'blue_sum_ema_{alpha}'] = df['blue_sum'].ewm(alpha=alpha).mean()
        
        # 相对强弱指标 (RSI)
        for window in [5, 10, 14]:
            if len(df) >= window:
                df[f'red_sum_rsi_{window}'] = self._calculate_rsi(df['red_sum'], window)
                df[f'red_mean_rsi_{window}'] = self._calculate_rsi(df['red_mean'], window)
        
        # 布林带
        for window in [10, 20]:
            if len(df) >= window:
                bollinger_features = self._calculate_bollinger_bands(df['red_sum'], window)
                for key, value in bollinger_features.items():
                    df[f'red_sum_bollinger_{key}_{window}'] = value
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滞后特征"""
        lag_features = ['red_sum', 'red_mean', 'blue_sum', 'red_odd_count', 'red_big_count']
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加差分特征"""
        diff_features = ['red_sum', 'red_mean', 'blue_sum']
        
        for feature in diff_features:
            if feature in df.columns:
                df[f'{feature}_diff_1'] = df[feature].diff(1)
                df[f'{feature}_diff_2'] = df[feature].diff(2)
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """计算相对强弱指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, series: pd.Series, window: int) -> Dict:
        """计算布林带"""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        return {
            'upper': sma + (std * 2),
            'lower': sma - (std * 2),
            'middle': sma,
            'width': (std * 4) / sma,
            'position': (series - sma) / (std * 2)
        }
    
    def train(self, history_data: List[Dict]) -> bool:
        """
        训练模型
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            训练是否成功
        """
        try:
            logger.info(f"开始训练XGBoost模型，数据量: {len(history_data)}")
            
            # 准备特征
            df = self.prepare_features(history_data)
            
            if df.empty or len(df) < 15:
                logger.error("XGBoost训练数据不足（至少需要15条记录）")
                return False
            
            # 删除包含NaN的行
            df = df.dropna()
            
            if len(df) < 10:
                logger.error("清理后的XGBoost训练数据不足")
                return False
            
            # 准备特征和目标变量
            feature_columns = [col for col in df.columns 
                             if col not in ['period', 'date'] + 
                             [f'red_{i+1}' for i in range(self.red_count)] +
                             [f'blue_{i+1}' for i in range(self.blue_count)]]
            
            X = df[feature_columns].values
            self.feature_columns = feature_columns
            
            # 训练红球模型
            red_scores = []
            for i in range(self.red_count):
                target_col = f'red_{i+1}'
                if target_col in df.columns:
                    y = df[target_col].values
                    
                    # 创建验证集
                    val_size = max(1, len(X) // 5)  # 20%作为验证集
                    X_train, X_val = X[:-val_size], X[-val_size:]
                    y_train, y_val = y[:-val_size], y[-val_size:]
                    
                    # 训练模型
                    self.red_models[i].fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    
                    # 评估
                    val_pred = self.red_models[i].predict(X_val)
                    val_score = np.sqrt(np.mean((y_val - val_pred) ** 2))
                    red_scores.append(val_score)
                    
                    logger.debug(f"红球位置 {i+1} 训练完成，验证RMSE: {val_score:.4f}")
            
            # 训练蓝球模型
            blue_scores = []
            for i in range(self.blue_count):
                target_col = f'blue_{i+1}'
                if target_col in df.columns:
                    y = df[target_col].values
                    
                    # 创建验证集
                    val_size = max(1, len(X) // 5)
                    X_train, X_val = X[:-val_size], X[-val_size:]
                    y_train, y_val = y[:-val_size], y[-val_size:]
                    
                    # 训练模型
                    self.blue_models[i].fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    
                    # 评估
                    val_pred = self.blue_models[i].predict(X_val)
                    val_score = np.sqrt(np.mean((y_val - val_pred) ** 2))
                    blue_scores.append(val_score)
                    
                    logger.debug(f"蓝球位置 {i+1} 训练完成，验证RMSE: {val_score:.4f}")
            
            # 保存性能指标
            self.performance_metrics = {
                'red_avg_rmse': np.mean(red_scores) if red_scores else 0,
                'blue_avg_rmse': np.mean(blue_scores) if blue_scores else 0,
                'training_samples': len(df),
                'feature_count': len(self.feature_columns),
                'feature_importance': self._get_feature_importance()
            }
            
            self.is_trained = True
            logger.info(f"XGBoost模型训练完成，红球平均RMSE: {self.performance_metrics['red_avg_rmse']:.4f}, "
                       f"蓝球平均RMSE: {self.performance_metrics['blue_avg_rmse']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"XGBoost模型训练失败: {e}")
            return False
    
    def _get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        try:
            importance_dict = {}
            
            # 红球模型的特征重要性
            if self.red_models and self.feature_columns:
                red_importance = np.mean([
                    model.feature_importances_ for model in self.red_models
                ], axis=0)
                
                for i, feature in enumerate(self.feature_columns):
                    importance_dict[f'red_{feature}'] = float(red_importance[i])
            
            # 蓝球模型的特征重要性
            if self.blue_models and self.feature_columns:
                blue_importance = np.mean([
                    model.feature_importances_ for model in self.blue_models
                ], axis=0)
                
                for i, feature in enumerate(self.feature_columns):
                    importance_dict[f'blue_{feature}'] = float(blue_importance[i])
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            return {}
    
    def predict(self, recent_data: Optional[List[Dict]] = None) -> Dict:
        """
        进行预测
        
        Args:
            recent_data: 最近的历史数据
            
        Returns:
            预测结果字典
        """
        try:
            if not self.is_trained:
                return {
                    'success': False,
                    'error': 'XGBoost模型未训练',
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
                    'error': 'XGBoost特征准备失败',
                    'model_name': self.model_name
                }
            
            # 使用最后一条记录进行预测
            latest_features = df[self.feature_columns].iloc[-1:].fillna(0).values
            
            # 预测红球
            red_predictions = []
            red_confidences = []
            for i, model in enumerate(self.red_models):
                pred = model.predict(latest_features)[0]
                # 限制在有效范围内
                pred = max(self.red_range[0], min(self.red_range[1], round(pred)))
                red_predictions.append(int(pred))
                
                # 计算单个预测的置信度
                confidence = self._calculate_single_confidence(model, latest_features)
                red_confidences.append(confidence)
            
            # 预测蓝球
            blue_predictions = []
            blue_confidences = []
            for i, model in enumerate(self.blue_models):
                pred = model.predict(latest_features)[0]
                # 限制在有效范围内
                pred = max(self.blue_range[0], min(self.blue_range[1], round(pred)))
                blue_predictions.append(int(pred))
                
                # 计算单个预测的置信度
                confidence = self._calculate_single_confidence(model, latest_features)
                blue_confidences.append(confidence)
            
            # 确保红球无重复
            red_predictions = self._ensure_unique_numbers(
                red_predictions, self.red_range, self.red_count
            )
            
            # 确保蓝球无重复
            blue_predictions = self._ensure_unique_numbers(
                blue_predictions, self.blue_range, self.blue_count
            )
            
            # 验证预测结果
            if not self.validate_prediction(red_predictions, blue_predictions):
                logger.warning("XGBoost预测结果验证失败，使用随机生成")
                red_predictions, blue_predictions = self._generate_random_numbers()
            
            # 计算综合置信度
            overall_confidence = np.mean(red_confidences + blue_confidences)
            
            result = {
                'success': True,
                'model_name': self.model_name,
                'lottery_type': self.lottery_type,
                'red_balls': sorted(red_predictions),
                'blue_balls': blue_predictions,
                'confidence': round(overall_confidence, 3),
                'prediction_time': pd.Timestamp.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'red_confidences': red_confidences,
                'blue_confidences': blue_confidences
            }
            
            logger.info(f"XGBoost预测完成: 红球 {result['red_balls']}, 蓝球 {result['blue_balls']}, "
                       f"置信度 {overall_confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"XGBoost预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': self.model_name
            }
    
    def _calculate_single_confidence(self, model, features: np.ndarray) -> float:
        """计算单个模型的预测置信度"""
        try:
            # 基于树的预测方差计算置信度
            if hasattr(model, 'predict'):
                # 简单的置信度计算：基于特征重要性
                prediction = model.predict(features)[0]
                
                # 基于预测值与范围的关系计算置信度
                if hasattr(self, 'red_range'):
                    range_size = self.red_range[1] - self.red_range[0] + 1
                    normalized_pred = (prediction - self.red_range[0]) / range_size
                    # 越接近中间值置信度越高
                    confidence = 1 - abs(normalized_pred - 0.5) * 2
                    return max(0.1, min(1.0, confidence))
                
            return 0.5
            
        except Exception as e:
            logger.warning(f"单个置信度计算失败: {e}")
            return 0.5
    
    def _ensure_unique_numbers(self, predictions: List[int], number_range: tuple, 
                              required_count: int) -> List[int]:
        """确保号码唯一性（与RandomForest相同的实现）"""
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
    
    def save_model(self, filepath: str) -> bool:
        """保存模型"""
        try:
            model_data = {
                'red_models': self.red_models,
                'blue_models': self.blue_models,
                'feature_columns': self.feature_columns,
                'performance_metrics': self.performance_metrics,
                'lottery_type': self.lottery_type,
                'model_name': self.model_name,
                'is_trained': self.is_trained,
                'hyperparameters': {
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'n_estimators': self.n_estimators,
                    'random_state': self.random_state
                }
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"XGBoost模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"XGBoost模型保存失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"XGBoost模型文件不存在: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.red_models = model_data['red_models']
            self.blue_models = model_data['blue_models']
            self.feature_columns = model_data['feature_columns']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = model_data['is_trained']
            
            # 加载超参数
            if 'hyperparameters' in model_data:
                hyperparams = model_data['hyperparameters']
                self.max_depth = hyperparams.get('max_depth', self.max_depth)
                self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
                self.n_estimators = hyperparams.get('n_estimators', self.n_estimators)
                self.random_state = hyperparams.get('random_state', self.random_state)
            
            logger.info(f"XGBoost模型已加载: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"XGBoost模型加载失败: {e}")
            return False
