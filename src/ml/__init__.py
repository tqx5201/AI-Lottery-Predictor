"""
机器学习模块 - 预测算法和模型
"""

from .base_predictor import BasePredictor
from .random_forest_predictor import RandomForestPredictor

# 条件导入可选依赖
try:
    from .xgboost_predictor import XGBoostPredictor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from .lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except (ImportError, NameError):
    LSTM_AVAILABLE = False

# 模型管理器和推荐引擎
from .model_manager import ModelManager
from .recommendation_engine import RecommendationEngine

# 导出列表
__all__ = ['BasePredictor', 'RandomForestPredictor', 'ModelManager', 'RecommendationEngine']

if XGBOOST_AVAILABLE:
    __all__.append('XGBoostPredictor')
    
if LSTM_AVAILABLE:
    __all__.append('LSTMPredictor')
