"""
LSTM预测器 - 基于LSTM神经网络的时间序列彩票预测
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import joblib
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow未安装，LSTM预测器不可用")

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class LSTMPredictor(BasePredictor):
    """
    LSTM预测器
    使用长短期记忆网络进行时间序列彩票预测
    """
    
    def __init__(self, lottery_type: str, sequence_length: int = 20, 
                 hidden_units: int = 50, num_layers: int = 2,
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32, random_state: int = 42):
        """
        初始化LSTM预测器
        
        Args:
            lottery_type: 彩票类型
            sequence_length: 序列长度（用多少期历史数据预测下一期）
            hidden_units: LSTM隐藏单元数
            num_layers: LSTM层数
            dropout_rate: Dropout比例
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            random_state: 随机种子
        """
        super().__init__(lottery_type, f"LSTM_{lottery_type}")
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow未安装，请先安装: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # 设置随机种子
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # 模型和数据缩放器
        self.red_model = None
        self.blue_model = None
        self.red_scaler = MinMaxScaler()
        self.blue_scaler = MinMaxScaler()
        
        # 模型性能指标
        self.performance_metrics = {}
        
        # 训练历史
        self.training_history = {}
    
    def prepare_features(self, history_data: List[Dict]) -> pd.DataFrame:
        """
        准备LSTM特征数据
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            特征DataFrame
        """
        try:
            features = []
            
            for data in history_data:
                red_balls, blue_balls = self.extract_numbers_from_data(data)
                
                if not red_balls or not blue_balls:
                    continue
                
                feature_dict = {
                    'period': data.get('period', ''),
                    'date': data.get('date', ''),
                }
                
                # 红球特征
                for i, num in enumerate(red_balls):
                    feature_dict[f'red_{i+1}'] = num
                
                # 蓝球特征
                for i, num in enumerate(blue_balls):
                    feature_dict[f'blue_{i+1}'] = num
                
                # 统计特征
                feature_dict['red_sum'] = sum(red_balls)
                feature_dict['red_mean'] = np.mean(red_balls)
                feature_dict['red_std'] = np.std(red_balls)
                feature_dict['red_min'] = min(red_balls)
                feature_dict['red_max'] = max(red_balls)
                feature_dict['red_range'] = max(red_balls) - min(red_balls)
                
                feature_dict['blue_sum'] = sum(blue_balls)
                feature_dict['blue_mean'] = np.mean(blue_balls)
                if len(blue_balls) > 1:
                    feature_dict['blue_std'] = np.std(blue_balls)
                    feature_dict['blue_range'] = max(blue_balls) - min(blue_balls)
                else:
                    feature_dict['blue_std'] = 0
                    feature_dict['blue_range'] = 0
                
                features.append(feature_dict)
            
            df = pd.DataFrame(features)
            logger.info(f"LSTM特征准备完成，共 {len(df)} 条记录，{len(df.columns)} 个特征")
            return df
            
        except Exception as e:
            logger.error(f"LSTM特征准备失败: {e}")
            return pd.DataFrame()
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据
        
        Args:
            data: 原始数据
            sequence_length: 序列长度
            
        Returns:
            (X, y) 序列数据和标签
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: tuple, output_dim: int) -> Sequential:
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入形状 (sequence_length, features)
            output_dim: 输出维度
            
        Returns:
            编译后的模型
        """
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(
            units=self.hidden_units,
            return_sequences=True if self.num_layers > 1 else False,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # 中间LSTM层
        for i in range(1, self.num_layers - 1):
            model.add(LSTM(
                units=self.hidden_units,
                return_sequences=True
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # 最后一层LSTM（如果有多层）
        if self.num_layers > 1:
            model.add(LSTM(units=self.hidden_units))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # 输出层
        model.add(Dense(units=self.hidden_units // 2, activation='relu'))
        model.add(Dropout(self.dropout_rate / 2))
        model.add(Dense(units=output_dim, activation='linear'))
        
        # 编译模型
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, history_data: List[Dict]) -> bool:
        """
        训练LSTM模型
        
        Args:
            history_data: 历史开奖数据
            
        Returns:
            训练是否成功
        """
        try:
            logger.info(f"开始训练LSTM模型，数据量: {len(history_data)}")
            
            # 准备特征
            df = self.prepare_features(history_data)
            
            if df.empty or len(df) < self.sequence_length + 10:
                logger.error(f"LSTM训练数据不足（至少需要{self.sequence_length + 10}条记录）")
                return False
            
            # 分离红球和蓝球数据
            red_columns = [f'red_{i+1}' for i in range(self.red_count)]
            blue_columns = [f'blue_{i+1}' for i in range(self.blue_count)]
            
            red_data = df[red_columns].values
            blue_data = df[blue_columns].values
            
            # 数据标准化
            red_data_scaled = self.red_scaler.fit_transform(red_data)
            blue_data_scaled = self.blue_scaler.fit_transform(blue_data)
            
            # 创建序列数据
            X_red, y_red = self._create_sequences(red_data_scaled, self.sequence_length)
            X_blue, y_blue = self._create_sequences(blue_data_scaled, self.sequence_length)
            
            if len(X_red) < 5 or len(X_blue) < 5:
                logger.error("序列数据不足")
                return False
            
            # 训练红球模型
            logger.info("训练红球LSTM模型...")
            self.red_model = self._build_model(
                input_shape=(self.sequence_length, self.red_count),
                output_dim=self.red_count
            )
            
            # 回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # 分割训练集和验证集
            val_size = max(1, len(X_red) // 5)
            X_red_train, X_red_val = X_red[:-val_size], X_red[-val_size:]
            y_red_train, y_red_val = y_red[:-val_size], y_red[-val_size:]
            
            red_history = self.red_model.fit(
                X_red_train, y_red_train,
                validation_data=(X_red_val, y_red_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # 训练蓝球模型
            logger.info("训练蓝球LSTM模型...")
            self.blue_model = self._build_model(
                input_shape=(self.sequence_length, self.blue_count),
                output_dim=self.blue_count
            )
            
            # 分割训练集和验证集
            val_size = max(1, len(X_blue) // 5)
            X_blue_train, X_blue_val = X_blue[:-val_size], X_blue[-val_size:]
            y_blue_train, y_blue_val = y_blue[:-val_size], y_blue[-val_size:]
            
            blue_history = self.blue_model.fit(
                X_blue_train, y_blue_train,
                validation_data=(X_blue_val, y_blue_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # 保存训练历史
            self.training_history = {
                'red_loss': red_history.history['loss'],
                'red_val_loss': red_history.history['val_loss'],
                'blue_loss': blue_history.history['loss'],
                'blue_val_loss': blue_history.history['val_loss']
            }
            
            # 评估模型
            red_val_loss = min(red_history.history['val_loss'])
            blue_val_loss = min(blue_history.history['val_loss'])
            
            self.performance_metrics = {
                'red_val_loss': red_val_loss,
                'blue_val_loss': blue_val_loss,
                'red_epochs_trained': len(red_history.history['loss']),
                'blue_epochs_trained': len(blue_history.history['loss']),
                'training_samples': len(X_red),
                'sequence_length': self.sequence_length
            }
            
            self.is_trained = True
            logger.info(f"LSTM模型训练完成，红球验证损失: {red_val_loss:.4f}, "
                       f"蓝球验证损失: {blue_val_loss:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"LSTM模型训练失败: {e}")
            return False
    
    def predict(self, recent_data: Optional[List[Dict]] = None) -> Dict:
        """
        进行LSTM预测
        
        Args:
            recent_data: 最近的历史数据
            
        Returns:
            预测结果字典
        """
        try:
            if not self.is_trained or self.red_model is None or self.blue_model is None:
                return {
                    'success': False,
                    'error': 'LSTM模型未训练',
                    'model_name': self.model_name
                }
            
            if not recent_data or len(recent_data) < self.sequence_length:
                return {
                    'success': False,
                    'error': f'LSTM预测需要至少{self.sequence_length}期历史数据',
                    'model_name': self.model_name
                }
            
            # 准备预测特征
            df = self.prepare_features(recent_data)
            
            if df.empty or len(df) < self.sequence_length:
                return {
                    'success': False,
                    'error': 'LSTM特征准备失败或数据不足',
                    'model_name': self.model_name
                }
            
            # 获取最后sequence_length期的数据
            red_columns = [f'red_{i+1}' for i in range(self.red_count)]
            blue_columns = [f'blue_{i+1}' for i in range(self.blue_count)]
            
            red_sequence = df[red_columns].iloc[-self.sequence_length:].values
            blue_sequence = df[blue_columns].iloc[-self.sequence_length:].values
            
            # 数据标准化
            red_sequence_scaled = self.red_scaler.transform(red_sequence)
            blue_sequence_scaled = self.blue_scaler.transform(blue_sequence)
            
            # 重塑为LSTM输入格式
            X_red = red_sequence_scaled.reshape(1, self.sequence_length, self.red_count)
            X_blue = blue_sequence_scaled.reshape(1, self.sequence_length, self.blue_count)
            
            # 进行预测
            red_pred_scaled = self.red_model.predict(X_red, verbose=0)
            blue_pred_scaled = self.blue_model.predict(X_blue, verbose=0)
            
            # 反标准化
            red_pred = self.red_scaler.inverse_transform(red_pred_scaled)[0]
            blue_pred = self.blue_scaler.inverse_transform(blue_pred_scaled)[0]
            
            # 处理预测结果
            red_predictions = []
            for pred in red_pred:
                pred = max(self.red_range[0], min(self.red_range[1], round(pred)))
                red_predictions.append(int(pred))
            
            blue_predictions = []
            for pred in blue_pred:
                pred = max(self.blue_range[0], min(self.blue_range[1], round(pred)))
                blue_predictions.append(int(pred))
            
            # 确保号码唯一性
            red_predictions = self._ensure_unique_numbers(
                red_predictions, self.red_range, self.red_count
            )
            blue_predictions = self._ensure_unique_numbers(
                blue_predictions, self.blue_range, self.blue_count
            )
            
            # 验证预测结果
            if not self.validate_prediction(red_predictions, blue_predictions):
                logger.warning("LSTM预测结果验证失败，使用随机生成")
                red_predictions, blue_predictions = self._generate_random_numbers()
            
            # 计算置信度
            confidence = self._calculate_confidence(red_pred_scaled, blue_pred_scaled)
            
            result = {
                'success': True,
                'model_name': self.model_name,
                'lottery_type': self.lottery_type,
                'red_balls': sorted(red_predictions),
                'blue_balls': blue_predictions,
                'confidence': confidence,
                'prediction_time': pd.Timestamp.now().isoformat(),
                'performance_metrics': self.performance_metrics,
                'sequence_used': self.sequence_length
            }
            
            logger.info(f"LSTM预测完成: 红球 {result['red_balls']}, 蓝球 {result['blue_balls']}, "
                       f"置信度 {confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"LSTM预测失败: {e}")
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
    
    def _calculate_confidence(self, red_pred_scaled: np.ndarray, 
                            blue_pred_scaled: np.ndarray) -> float:
        """计算LSTM预测置信度"""
        try:
            # 基于预测结果的方差计算置信度
            red_variance = np.var(red_pred_scaled)
            blue_variance = np.var(blue_pred_scaled)
            
            # 方差越小，置信度越高
            red_confidence = 1 / (1 + red_variance)
            blue_confidence = 1 / (1 + blue_variance)
            
            # 综合置信度
            overall_confidence = (red_confidence + blue_confidence) / 2
            
            # 基于训练损失调整置信度
            if 'red_val_loss' in self.performance_metrics:
                loss_factor = 1 / (1 + self.performance_metrics['red_val_loss'] + 
                                 self.performance_metrics['blue_val_loss'])
                overall_confidence = (overall_confidence + loss_factor) / 2
            
            return round(min(1.0, max(0.1, overall_confidence)), 3)
            
        except Exception as e:
            logger.warning(f"LSTM置信度计算失败: {e}")
            return 0.5
    
    def save_model(self, filepath: str) -> bool:
        """保存LSTM模型"""
        try:
            # 创建保存目录
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存Keras模型
            red_model_path = filepath.replace('.pkl', '_red_model.h5')
            blue_model_path = filepath.replace('.pkl', '_blue_model.h5')
            
            if self.red_model:
                self.red_model.save(red_model_path)
            if self.blue_model:
                self.blue_model.save(blue_model_path)
            
            # 保存其他数据
            model_data = {
                'red_scaler': self.red_scaler,
                'blue_scaler': self.blue_scaler,
                'performance_metrics': self.performance_metrics,
                'training_history': self.training_history,
                'lottery_type': self.lottery_type,
                'model_name': self.model_name,
                'is_trained': self.is_trained,
                'hyperparameters': {
                    'sequence_length': self.sequence_length,
                    'hidden_units': self.hidden_units,
                    'num_layers': self.num_layers,
                    'dropout_rate': self.dropout_rate,
                    'learning_rate': self.learning_rate,
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'random_state': self.random_state
                },
                'model_paths': {
                    'red_model': red_model_path,
                    'blue_model': blue_model_path
                }
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"LSTM模型已保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"LSTM模型保存失败: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """加载LSTM模型"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"LSTM模型文件不存在: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            # 加载Keras模型
            red_model_path = model_data['model_paths']['red_model']
            blue_model_path = model_data['model_paths']['blue_model']
            
            if os.path.exists(red_model_path):
                self.red_model = load_model(red_model_path)
            if os.path.exists(blue_model_path):
                self.blue_model = load_model(blue_model_path)
            
            # 加载其他数据
            self.red_scaler = model_data['red_scaler']
            self.blue_scaler = model_data['blue_scaler']
            self.performance_metrics = model_data['performance_metrics']
            self.training_history = model_data['training_history']
            self.is_trained = model_data['is_trained']
            
            # 加载超参数
            if 'hyperparameters' in model_data:
                hyperparams = model_data['hyperparameters']
                self.sequence_length = hyperparams.get('sequence_length', self.sequence_length)
                self.hidden_units = hyperparams.get('hidden_units', self.hidden_units)
                self.num_layers = hyperparams.get('num_layers', self.num_layers)
                self.dropout_rate = hyperparams.get('dropout_rate', self.dropout_rate)
                self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
                self.epochs = hyperparams.get('epochs', self.epochs)
                self.batch_size = hyperparams.get('batch_size', self.batch_size)
                self.random_state = hyperparams.get('random_state', self.random_state)
            
            logger.info(f"LSTM模型已加载: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"LSTM模型加载失败: {e}")
            return False
