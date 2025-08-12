"""
随机森林预测器 - 基于随机森林算法的彩票预测
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import logging
import joblib
import os

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class RandomForestPredictor(BasePredictor):
    """
    随机森林预测器
    使用随机森林算法进行彩票号码预测
    """
    
    def __init__(self, lottery_type: str, n_estimators: int = 100, random_state: int = 42):
        """
        初始化随机森林预测器
        
        Args:
            lottery_type: 彩票类型
            n_estimators: 树的数量
            random_state: 随机种子
        """
        super().__init__(lottery_type, f"RandomForest_{lottery_type}")
        
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # 为每个号码位置创建独立的模型
        self.red_models = []
        self.blue_models = []
        
        # 特征缩放器
        self.scaler = StandardScaler()
        
        # 模型性能指标
        self.performance_metrics = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        # 红球模型 (回归模型预测具体数值)
        for i in range(self.red_count):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state + i,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.red_models.append(model)
        
        # 蓝球模型
        for i in range(self.blue_count):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state + 100 + i,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2
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
            
            # 添加额外的技术特征
            df = self._add_technical_features(df)
            
            # 添加时间特征
            df = self._add_time_features(df)
            
            # 添加趋势特征
            df = self._add_trend_features(df)
            
            logger.info(f"特征准备完成，共 {len(df)} 条记录，{len(df.columns)} 个特征")
            return df
            
        except Exception as e:
            logger.error(f"特征准备失败: {e}")
            return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        if len(df) < 2:
            return df
        
        # 移动平均
        for window in [3, 5, 10]:
            if len(df) >= window:
                df[f'red_sum_ma{window}'] = df['red_sum'].rolling(window=window).mean()
                df[f'red_mean_ma{window}'] = df['red_mean'].rolling(window=window).mean()
                df[f'blue_sum_ma{window}'] = df['blue_sum'].rolling(window=window).mean()
        
        # 变化率
        df['red_sum_change'] = df['red_sum'].pct_change()
        df['red_mean_change'] = df['red_mean'].pct_change()
        df['blue_sum_change'] = df['blue_sum'].pct_change()
        
        # 波动率 (标准差的滚动计算)
        for window in [5, 10]:
            if len(df) >= window:
                df[f'red_sum_volatility{window}'] = df['red_sum'].rolling(window=window).std()
                df[f'red_mean_volatility{window}'] = df['red_mean'].rolling(window=window).std()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间相关特征"""
        try:
            # 期号序列特征
            df['period_sequence'] = range(len(df))
            
            # 如果有日期信息
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['weekday'] = df['date'].dt.dayofweek
                df['month'] = df['date'].dt.month
                df['quarter'] = df['date'].dt.quarter
                df['is_weekend'] = (df['weekday'] >= 5).astype(int)
            
            return df
            
        except Exception as e:
            logger.warning(f"添加时间特征失败: {e}")
            return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势特征"""
        if len(df) < 3:
            return df
        
        # 连续上升/下降趋势
        df['red_sum_trend'] = (df['red_sum'] > df['red_sum'].shift(1)).astype(int)
        df['red_mean_trend'] = (df['red_mean'] > df['red_mean'].shift(1)).astype(int)
        
        # 震荡指标
        for window in [3, 5]:
            if len(df) >= window:
                df[f'red_sum_oscillator{window}'] = (
                    df['red_sum'] - df['red_sum'].rolling(window=window).mean()
                ) / df['red_sum'].rolling(window=window).std()
        
        return df
    
    def train(self, history_data: List[Dict]) -> bool:
        """
        训练模型
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            训练是否成功
        """
        try:
            logger.info(f"开始训练随机森林模型，数据量: {len(history_data)}")
            
            # 准备特征
            df = self.prepare_features(history_data)
            
            if df.empty or len(df) < 10:
                logger.error("训练数据不足")
                return False
            
            # 删除包含NaN的行
            df = df.dropna()
            
            if len(df) < 5:
                logger.error("清理后的训练数据不足")
                return False
            
            # 准备特征和目标变量
            feature_columns = [col for col in df.columns 
                             if col not in ['period', 'date'] + 
                             [f'red_{i+1}' for i in range(self.red_count)] +
                             [f'blue_{i+1}' for i in range(self.blue_count)]]
            
            X = df[feature_columns]
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = feature_columns
            
            # 训练红球模型
            red_scores = []
            for i in range(self.red_count):
                target_col = f'red_{i+1}'
                if target_col in df.columns:
                    y = df[target_col]
                    
                    # 分割训练集和验证集
                    if len(X_scaled) > 20:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=self.random_state + i
                        )
                    else:
                        X_train, X_val, y_train, y_val = X_scaled, X_scaled, y, y
                    
                    # 训练模型
                    self.red_models[i].fit(X_train, y_train)
                    
                    # 评估
                    val_pred = self.red_models[i].predict(X_val)
                    val_score = mean_squared_error(y_val, val_pred)
                    red_scores.append(val_score)
                    
                    logger.debug(f"红球位置 {i+1} 训练完成，验证MSE: {val_score:.4f}")
            
            # 训练蓝球模型
            blue_scores = []
            for i in range(self.blue_count):
                target_col = f'blue_{i+1}'
                if target_col in df.columns:
                    y = df[target_col]
                    
                    # 分割训练集和验证集
                    if len(X_scaled) > 20:
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_scaled, y, test_size=0.2, random_state=self.random_state + 200 + i
                        )
                    else:
                        X_train, X_val, y_train, y_val = X_scaled, X_scaled, y, y
                    
                    # 训练模型
                    self.blue_models[i].fit(X_train, y_train)
                    
                    # 评估
                    val_pred = self.blue_models[i].predict(X_val)
                    val_score = mean_squared_error(y_val, val_pred)
                    blue_scores.append(val_score)
                    
                    logger.debug(f"蓝球位置 {i+1} 训练完成，验证MSE: {val_score:.4f}")
            
            # 保存性能指标
            self.performance_metrics = {
                'red_avg_mse': np.mean(red_scores) if red_scores else 0,
                'blue_avg_mse': np.mean(blue_scores) if blue_scores else 0,
                'training_samples': len(df),
                'feature_count': len(self.feature_columns)
            }
            
            self.is_trained = True
            logger.info(f"模型训练完成，红球平均MSE: {self.performance_metrics['red_avg_mse']:.4f}, "
                       f"蓝球平均MSE: {self.performance_metrics['blue_avg_mse']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return False
    
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
                    'error': '模型未训练',
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
                    'error': '特征准备失败',
                    'model_name': self.model_name
                }
            
            # 使用最后一条记录进行预测
            latest_features = df[self.feature_columns].iloc[-1:].fillna(0)
            X_scaled = self.scaler.transform(latest_features)
            
            # 预测红球
            red_predictions = []
            for i, model in enumerate(self.red_models):
                pred = model.predict(X_scaled)[0]
                # 限制在有效范围内
                pred = max(self.red_range[0], min(self.red_range[1], round(pred)))
                red_predictions.append(int(pred))
            
            # 预测蓝球
            blue_predictions = []
            for i, model in enumerate(self.blue_models):
                pred = model.predict(X_scaled)[0]
                # 限制在有效范围内
                pred = max(self.blue_range[0], min(self.blue_range[1], round(pred)))
                blue_predictions.append(int(pred))
            
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
                logger.warning("预测结果验证失败，使用随机生成")
                red_predictions, blue_predictions = self._generate_random_numbers()
            
            # 计算预测置信度
            confidence = self._calculate_confidence(X_scaled)
            
            result = {
                'success': True,
                'model_name': self.model_name,
                'lottery_type': self.lottery_type,
                'red_balls': sorted(red_predictions),
                'blue_balls': blue_predictions,
                'confidence': confidence,
                'prediction_time': pd.Timestamp.now().isoformat(),
                'performance_metrics': self.performance_metrics
            }
            
            logger.info(f"预测完成: 红球 {result['red_balls']}, 蓝球 {result['blue_balls']}, "
                       f"置信度 {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': self.model_name
            }
    
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
    
    def _calculate_confidence(self, X_scaled: np.ndarray) -> float:
        """计算预测置信度"""
        try:
            # 基于模型性能计算置信度
            red_confidence = 1 / (1 + self.performance_metrics.get('red_avg_mse', 1))
            blue_confidence = 1 / (1 + self.performance_metrics.get('blue_avg_mse', 1))
            
            # 综合置信度
            overall_confidence = (red_confidence + blue_confidence) / 2
            
            # 标准化到0-1范围
            confidence = min(1.0, max(0.1, overall_confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.warning(f"置信度计算失败: {e}")
            return 0.5
    
    def save_model(self, filepath: str) -> bool:
        """保存模型"""
        try:
            model_data = {
                'red_models': self.red_models,
                'blue_models': self.blue_models,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'performance_metrics': self.performance_metrics,
                'lottery_type': self.lottery_type,
                'model_name': self.model_name,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"模型文件不存在: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.red_models = model_data['red_models']
            self.blue_models = model_data['blue_models']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"模型已加载: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
