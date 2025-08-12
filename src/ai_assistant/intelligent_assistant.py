"""
AI智能助手系统
提供自然语言交互、智能分析、决策支持等功能
"""

import json
import time
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# NLP和AI库导入
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False
    pipeline = AutoTokenizer = AutoModel = torch = None
    logger.warning(f"Transformers/PyTorch导入失败: {e}")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = word_tokenize = sent_tokenize = stopwords = WordNetLemmatizer = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)


class ConversationRole(Enum):
    """对话角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class IntentType(Enum):
    """意图类型"""
    ANALYSIS_REQUEST = "analysis_request"
    PREDICTION_REQUEST = "prediction_request"
    DATA_QUERY = "data_query"
    HELP_REQUEST = "help_request"
    CONFIGURATION = "configuration"
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


class ResponseType(Enum):
    """响应类型"""
    TEXT = "text"
    DATA = "data"
    CHART = "chart"
    ACTION = "action"
    MIXED = "mixed"


@dataclass
class ConversationMessage:
    """对话消息"""
    role: ConversationRole
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['role'] = self.role.value
        return result


@dataclass
class UserIntent:
    """用户意图"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    parameters: Dict[str, Any]
    context: Dict[str, Any]


@dataclass
class AssistantResponse:
    """助手响应"""
    response_type: ResponseType
    content: str
    data: Optional[Dict[str, Any]] = None
    actions: Optional[List[str]] = None
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class NLPProcessor:
    """自然语言处理器"""
    
    def __init__(self):
        self.tokenizer = None
        self.lemmatizer = None
        self.nlp_model = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """初始化NLP工具"""
        try:
            if NLTK_AVAILABLE:
                # 下载必要的NLTK数据
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
                
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords')
                
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet')
                
                self.lemmatizer = WordNetLemmatizer()
                logger.info("NLTK初始化完成")
            
            if SPACY_AVAILABLE:
                self.nlp_model = None
                # 尝试加载不同的SpaCy模型，按优先级排序
                models_to_try = [
                    ("zh_core_web_sm", "中文"),
                    ("en_core_web_sm", "英文"),
                    ("en_core_web_md", "英文-中等"),
                    ("en_core_web_lg", "英文-大型")
                ]
                
                for model_name, model_desc in models_to_try:
                    try:
                        self.nlp_model = spacy.load(model_name)
                        logger.info(f"SpaCy{model_desc}模型({model_name})加载完成")
                        break
                    except OSError:
                        continue
                
                if self.nlp_model is None:
                    logger.warning("SpaCy模型加载失败，请运行 'python setup_models.py' 安装模型")
            else:
                logger.info("SpaCy不可用，跳过模型加载")
            
            if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                try:
                    # 检查PyTorch版本
                    if hasattr(torch, '__version__'):
                        torch_version = torch.__version__
                        logger.info(f"PyTorch版本: {torch_version}")
                    
                    # 使用更轻量级的模型，避免网络下载问题
                    self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                                      model="distilbert-base-uncased-finetuned-sst-2-english",
                                                      return_all_scores=True)
                    logger.info("Transformers情感分析模型加载完成")
                except Exception as e:
                    logger.warning(f"Transformers模型加载失败: {e}")
                    self.sentiment_pipeline = None
            else:
                logger.info("Transformers或PyTorch不可用，跳过模型加载")
                self.sentiment_pipeline = None
            
        except Exception as e:
            logger.error(f"NLP初始化失败: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 清理文本
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 保留中文、英文、数字
        text = text.strip().lower()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.nlp_model:
            doc = self.nlp_model(text)
            return [token.text for token in doc if not token.is_punct and not token.is_space]
        elif NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            return text.split()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体"""
        entities = {
            'numbers': [],
            'dates': [],
            'models': [],
            'actions': [],
            'periods': []
        }
        
        # 数字提取
        numbers = re.findall(r'\d+', text)
        entities['numbers'] = numbers
        
        # 日期提取
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}年\d{1,2}月\d{1,2}日'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            entities['dates'].extend(dates)
        
        # 模型名称提取
        model_keywords = ['随机森林', 'xgboost', 'lstm', '神经网络', '机器学习', '深度学习']
        for keyword in model_keywords:
            if keyword in text:
                entities['models'].append(keyword)
        
        # 动作提取
        action_keywords = ['预测', '分析', '查询', '比较', '推荐', '解释', '配置']
        for keyword in action_keywords:
            if keyword in text:
                entities['actions'].append(keyword)
        
        # 期号提取
        period_pattern = r'\d{4}\d{3}'
        periods = re.findall(period_pattern, text)
        entities['periods'] = periods
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感分析"""
        if hasattr(self, 'sentiment_pipeline') and self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text)
                return {
                    'sentiment': result[0]['label'],
                    'confidence': result[0]['score']
                }
            except Exception as e:
                logger.warning(f"情感分析失败: {e}")
        
        # 简单的规则基础情感分析
        positive_words = ['好', '棒', '优秀', '满意', '喜欢', '赞', '不错']
        negative_words = ['差', '糟', '失望', '不满', '讨厌', '烂', '问题']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {'sentiment': 'POSITIVE', 'confidence': 0.7}
        elif negative_count > positive_count:
            return {'sentiment': 'NEGATIVE', 'confidence': 0.7}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.6}


class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
    
    def _build_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """构建意图模式"""
        return {
            IntentType.ANALYSIS_REQUEST: [
                '分析', '统计', '趋势', '频率', '规律', '模式', '特征'
            ],
            IntentType.PREDICTION_REQUEST: [
                '预测', '预报', '推测', '估计', '预估', '下期', '下一期'
            ],
            IntentType.DATA_QUERY: [
                '查询', '查看', '显示', '数据', '历史', '记录', '结果'
            ],
            IntentType.HELP_REQUEST: [
                '帮助', '如何', '怎么', '什么', '为什么', '说明', '教程'
            ],
            IntentType.CONFIGURATION: [
                '设置', '配置', '调整', '修改', '参数', '选项', '偏好'
            ],
            IntentType.EXPLANATION: [
                '解释', '说明', '原理', '为什么', '如何工作', '机制'
            ],
            IntentType.RECOMMENDATION: [
                '推荐', '建议', '推荐', '最好', '最佳', '优化', '改进'
            ],
            IntentType.COMPARISON: [
                '比较', '对比', '差异', '区别', '哪个好', '优劣'
            ]
        }
    
    def classify_intent(self, text: str, entities: Dict[str, List[str]]) -> UserIntent:
        """分类用户意图"""
        text_lower = text.lower()
        
        # 计算每个意图的匹配分数
        intent_scores = {}
        
        for intent_type, keywords in self.intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            # 根据实体信息调整分数
            if intent_type == IntentType.PREDICTION_REQUEST and entities['actions']:
                if '预测' in entities['actions']:
                    score += 2
            
            if intent_type == IntentType.DATA_QUERY and entities['numbers']:
                score += 1
            
            intent_scores[intent_type] = score
        
        # 找到最高分的意图
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if best_intent[1] == 0:
            intent_type = IntentType.UNKNOWN
            confidence = 0.1
        else:
            intent_type = best_intent[0]
            confidence = min(0.9, best_intent[1] / len(self.intent_patterns[intent_type]))
        
        # 提取参数
        parameters = self._extract_parameters(text_lower, entities, intent_type)
        
        return UserIntent(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            parameters=parameters,
            context={'original_text': text}
        )
    
    def _extract_parameters(self, text: str, entities: Dict[str, List[str]], 
                          intent_type: IntentType) -> Dict[str, Any]:
        """提取参数"""
        parameters = {}
        
        if intent_type == IntentType.PREDICTION_REQUEST:
            # 提取预测相关参数
            if '下期' in text or '下一期' in text:
                parameters['target'] = 'next_period'
            
            if entities['numbers']:
                parameters['numbers'] = entities['numbers']
            
            if '双色球' in text:
                parameters['lottery_type'] = '双色球'
            elif '大乐透' in text:
                parameters['lottery_type'] = '大乐透'
        
        elif intent_type == IntentType.ANALYSIS_REQUEST:
            # 提取分析相关参数
            if '频率' in text:
                parameters['analysis_type'] = 'frequency'
            elif '趋势' in text:
                parameters['analysis_type'] = 'trend'
            elif '相关' in text:
                parameters['analysis_type'] = 'correlation'
            
            if entities['numbers']:
                parameters['period_count'] = int(entities['numbers'][0]) if entities['numbers'] else 30
        
        elif intent_type == IntentType.DATA_QUERY:
            # 提取查询相关参数
            if entities['periods']:
                parameters['periods'] = entities['periods']
            
            if entities['dates']:
                parameters['dates'] = entities['dates']
        
        return parameters


class KnowledgeBase:
    """知识库"""
    
    def __init__(self):
        self.knowledge = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """构建知识库"""
        return {
            'lottery_info': {
                '双色球': {
                    'description': '双色球是中国福利彩票的一种，由红球和蓝球组成',
                    'red_balls': {'range': '1-33', 'count': 6},
                    'blue_balls': {'range': '1-16', 'count': 1},
                    'draw_frequency': '每周二、四、日开奖'
                },
                '大乐透': {
                    'description': '大乐透是中国体育彩票的一种',
                    'front_area': {'range': '1-35', 'count': 5},
                    'back_area': {'range': '1-12', 'count': 2},
                    'draw_frequency': '每周一、三、六开奖'
                }
            },
            'analysis_methods': {
                '频率分析': '统计每个号码在历史开奖中出现的次数',
                '趋势分析': '分析号码出现的时间趋势和周期性',
                '相关性分析': '分析不同号码之间的关联关系',
                '模式识别': '识别开奖号码中的特定模式'
            },
            'prediction_models': {
                '随机森林': '基于决策树的集成学习算法',
                'XGBoost': '梯度提升决策树算法',
                'LSTM': '长短期记忆神经网络，适合时序数据',
                '量子算法': '使用量子计算进行优化和预测'
            },
            'common_questions': {
                '如何提高预测准确率': [
                    '增加历史数据量',
                    '使用多种模型集成',
                    '考虑更多特征维度',
                    '定期更新模型参数'
                ],
                '哪个模型最好': [
                    '没有绝对最好的模型',
                    '建议使用集成方法',
                    '根据数据特点选择',
                    '定期评估模型性能'
                ]
            }
        }
    
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """搜索知识库"""
        results = []
        query_lower = query.lower()
        
        # 搜索彩票信息
        for lottery_type, info in self.knowledge['lottery_info'].items():
            if lottery_type in query_lower or info['description'] in query_lower:
                results.append({
                    'type': 'lottery_info',
                    'title': lottery_type,
                    'content': info,
                    'relevance': 0.9
                })
        
        # 搜索分析方法
        for method, description in self.knowledge['analysis_methods'].items():
            if any(keyword in query_lower for keyword in method.lower().split()):
                results.append({
                    'type': 'analysis_method',
                    'title': method,
                    'content': description,
                    'relevance': 0.8
                })
        
        # 搜索预测模型
        for model, description in self.knowledge['prediction_models'].items():
            if model.lower() in query_lower:
                results.append({
                    'type': 'prediction_model',
                    'title': model,
                    'content': description,
                    'relevance': 0.8
                })
        
        # 搜索常见问题
        for question, answers in self.knowledge['common_questions'].items():
            if any(keyword in query_lower for keyword in question.lower().split()):
                results.append({
                    'type': 'faq',
                    'title': question,
                    'content': answers,
                    'relevance': 0.7
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results


class ResponseGenerator:
    """响应生成器"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.response_templates = self._build_response_templates()
    
    def _build_response_templates(self) -> Dict[IntentType, Dict[str, str]]:
        """构建响应模板"""
        return {
            IntentType.ANALYSIS_REQUEST: {
                'greeting': '我来为您进行数据分析。',
                'processing': '正在分析历史数据...',
                'result': '分析结果如下：',
                'suggestion': '基于分析结果，我建议：'
            },
            IntentType.PREDICTION_REQUEST: {
                'greeting': '我来为您进行预测分析。',
                'processing': '正在运行预测模型...',
                'result': '预测结果如下：',
                'confidence': '预测置信度：',
                'warning': '请注意：彩票预测仅供参考，不保证准确性。'
            },
            IntentType.DATA_QUERY: {
                'greeting': '我来帮您查询数据。',
                'processing': '正在检索数据...',
                'result': '查询结果：',
                'empty': '抱歉，没有找到相关数据。'
            },
            IntentType.HELP_REQUEST: {
                'greeting': '我很乐意为您提供帮助。',
                'explanation': '让我来解释一下：',
                'guidance': '您可以尝试：'
            },
            IntentType.EXPLANATION: {
                'greeting': '让我来解释这个概念。',
                'details': '详细说明：',
                'example': '举个例子：'
            }
        }
    
    def generate_response(self, intent: UserIntent, 
                         context: Optional[Dict[str, Any]] = None) -> AssistantResponse:
        """生成响应"""
        intent_type = intent.intent_type
        
        if intent_type == IntentType.PREDICTION_REQUEST:
            return self._generate_prediction_response(intent, context)
        elif intent_type == IntentType.ANALYSIS_REQUEST:
            return self._generate_analysis_response(intent, context)
        elif intent_type == IntentType.DATA_QUERY:
            return self._generate_query_response(intent, context)
        elif intent_type == IntentType.HELP_REQUEST:
            return self._generate_help_response(intent, context)
        elif intent_type == IntentType.EXPLANATION:
            return self._generate_explanation_response(intent, context)
        else:
            return self._generate_default_response(intent, context)
    
    def _generate_prediction_response(self, intent: UserIntent, 
                                    context: Optional[Dict[str, Any]]) -> AssistantResponse:
        """生成预测响应"""
        templates = self.response_templates[IntentType.PREDICTION_REQUEST]
        
        # 提取参数
        lottery_type = intent.parameters.get('lottery_type', '双色球')
        target = intent.parameters.get('target', 'next_period')
        
        # 模拟预测结果
        if lottery_type == '双色球':
            predicted_red = sorted(np.random.choice(range(1, 34), 6, replace=False).tolist())
            predicted_blue = [np.random.randint(1, 17)]
            confidence = np.random.uniform(0.6, 0.9)
        else:
            predicted_red = sorted(np.random.choice(range(1, 36), 5, replace=False).tolist())
            predicted_blue = sorted(np.random.choice(range(1, 13), 2, replace=False).tolist())
            confidence = np.random.uniform(0.6, 0.9)
        
        content = f"{templates['greeting']}\n\n"
        content += f"{templates['result']}\n"
        content += f"彩票类型：{lottery_type}\n"
        
        if lottery_type == '双色球':
            content += f"预测红球：{' '.join(map(str, predicted_red))}\n"
            content += f"预测蓝球：{predicted_blue[0]}\n"
        else:
            content += f"前区号码：{' '.join(map(str, predicted_red))}\n"
            content += f"后区号码：{' '.join(map(str, predicted_blue))}\n"
        
        content += f"\n{templates['confidence']}{confidence:.1%}\n"
        content += f"\n⚠️ {templates['warning']}"
        
        return AssistantResponse(
            response_type=ResponseType.MIXED,
            content=content,
            data={
                'lottery_type': lottery_type,
                'predicted_numbers': {
                    'red': predicted_red,
                    'blue': predicted_blue
                },
                'confidence': confidence
            },
            confidence=confidence,
            metadata={'intent_type': intent.intent_type.value}
        )
    
    def _generate_analysis_response(self, intent: UserIntent, 
                                  context: Optional[Dict[str, Any]]) -> AssistantResponse:
        """生成分析响应"""
        templates = self.response_templates[IntentType.ANALYSIS_REQUEST]
        
        analysis_type = intent.parameters.get('analysis_type', 'frequency')
        period_count = intent.parameters.get('period_count', 30)
        
        content = f"{templates['greeting']}\n\n"
        content += f"分析类型：{analysis_type}\n"
        content += f"分析期数：最近{period_count}期\n\n"
        
        # 模拟分析结果
        if analysis_type == 'frequency':
            content += "📊 号码频率分析结果：\n"
            content += "热门号码：1, 7, 12, 18, 25, 33\n"
            content += "冷门号码：3, 9, 15, 21, 27, 31\n"
            content += "平均出现频率：16.7%\n"
        elif analysis_type == 'trend':
            content += "📈 趋势分析结果：\n"
            content += "整体趋势：号码分布趋于均匀\n"
            content += "周期性：存在7期的小周期\n"
            content += "波动性：中等波动\n"
        else:
            content += "🔍 综合分析结果：\n"
            content += "数据质量：良好\n"
            content += "模式识别：发现3种常见模式\n"
            content += "预测难度：中等\n"
        
        content += f"\n{templates['suggestion']}\n"
        content += "1. 结合多种分析方法\n"
        content += "2. 关注长期趋势变化\n"
        content += "3. 定期更新分析数据"
        
        return AssistantResponse(
            response_type=ResponseType.MIXED,
            content=content,
            data={
                'analysis_type': analysis_type,
                'period_count': period_count,
                'results': {
                    'hot_numbers': [1, 7, 12, 18, 25, 33],
                    'cold_numbers': [3, 9, 15, 21, 27, 31]
                }
            },
            confidence=0.8,
            metadata={'intent_type': intent.intent_type.value}
        )
    
    def _generate_query_response(self, intent: UserIntent, 
                               context: Optional[Dict[str, Any]]) -> AssistantResponse:
        """生成查询响应"""
        templates = self.response_templates[IntentType.DATA_QUERY]
        
        content = f"{templates['greeting']}\n\n"
        
        if intent.entities['periods']:
            periods = intent.entities['periods']
            content += f"查询期号：{', '.join(periods)}\n"
            content += "查询结果：\n"
            for period in periods[:3]:  # 最多显示3个
                content += f"期号 {period}：红球 01 07 12 18 25 33，蓝球 08\n"
        elif intent.entities['dates']:
            dates = intent.entities['dates']
            content += f"查询日期：{', '.join(dates)}\n"
            content += "在此日期范围内共找到 5 期开奖记录\n"
        else:
            content += "最近开奖记录：\n"
            content += "2024001期：红球 03 08 15 22 29 32，蓝球 12\n"
            content += "2024002期：红球 05 11 17 24 28 31，蓝球 06\n"
            content += "2024003期：红球 02 09 16 21 26 33，蓝球 14\n"
        
        return AssistantResponse(
            response_type=ResponseType.DATA,
            content=content,
            data={
                'query_type': 'historical_data',
                'results': [
                    {'period': '2024001', 'red': [3, 8, 15, 22, 29, 32], 'blue': [12]},
                    {'period': '2024002', 'red': [5, 11, 17, 24, 28, 31], 'blue': [6]},
                    {'period': '2024003', 'red': [2, 9, 16, 21, 26, 33], 'blue': [14]}
                ]
            },
            confidence=0.9,
            metadata={'intent_type': intent.intent_type.value}
        )
    
    def _generate_help_response(self, intent: UserIntent, 
                              context: Optional[Dict[str, Any]]) -> AssistantResponse:
        """生成帮助响应"""
        templates = self.response_templates[IntentType.HELP_REQUEST]
        
        # 搜索知识库
        query = intent.context.get('original_text', '')
        knowledge_results = self.knowledge_base.search_knowledge(query)
        
        content = f"{templates['greeting']}\n\n"
        
        if knowledge_results:
            content += f"{templates['explanation']}\n"
            for result in knowledge_results[:3]:  # 显示前3个结果
                content += f"\n📌 {result['title']}：\n"
                if isinstance(result['content'], dict):
                    for key, value in result['content'].items():
                        content += f"  • {key}：{value}\n"
                elif isinstance(result['content'], list):
                    for item in result['content']:
                        content += f"  • {item}\n"
                else:
                    content += f"  {result['content']}\n"
        else:
            content += "我可以帮您：\n"
            content += "• 🎯 预测下期号码\n"
            content += "• 📊 分析历史数据\n"
            content += "• 🔍 查询开奖记录\n"
            content += "• ⚙️ 配置系统参数\n"
            content += "• 📈 生成统计图表\n"
            content += "• 💡 提供专业建议\n"
        
        content += f"\n{templates['guidance']}\n"
        content += "• 说\"预测双色球下期号码\"\n"
        content += "• 说\"分析最近30期的频率\"\n"
        content += "• 说\"查询2024001期开奖结果\"\n"
        
        return AssistantResponse(
            response_type=ResponseType.TEXT,
            content=content,
            confidence=0.9,
            metadata={'intent_type': intent.intent_type.value}
        )
    
    def _generate_explanation_response(self, intent: UserIntent, 
                                     context: Optional[Dict[str, Any]]) -> AssistantResponse:
        """生成解释响应"""
        templates = self.response_templates[IntentType.EXPLANATION]
        
        content = f"{templates['greeting']}\n\n"
        
        # 根据实体提供解释
        if '随机森林' in intent.context.get('original_text', ''):
            content += f"{templates['details']}\n"
            content += "随机森林是一种集成学习算法，它的工作原理是：\n"
            content += "1. 🌳 构建多个决策树\n"
            content += "2. 🎲 每个树使用不同的数据子集\n"
            content += "3. 🗳️ 通过投票决定最终结果\n"
            content += "4. 📊 提供特征重要性分析\n\n"
            content += f"{templates['example']}\n"
            content += "在彩票预测中，随机森林可以：\n"
            content += "• 分析号码出现的历史模式\n"
            content += "• 识别重要的预测特征\n"
            content += "• 提供稳定的预测结果\n"
        elif '量子' in intent.context.get('original_text', ''):
            content += f"{templates['details']}\n"
            content += "量子计算在彩票分析中的应用：\n"
            content += "1. ⚛️ 量子叠加：同时处理多种可能性\n"
            content += "2. 🔗 量子纠缠：发现复杂的关联关系\n"
            content += "3. 🚀 量子加速：解决大规模优化问题\n"
            content += "4. 🎯 量子算法：QAOA、VQE等专门算法\n"
        else:
            content += "请告诉我您想了解什么概念，我会详细解释。\n"
            content += "我可以解释的内容包括：\n"
            content += "• 🤖 机器学习算法\n"
            content += "• 📊 数据分析方法\n"
            content += "• ⚛️ 量子计算原理\n"
            content += "• 📈 统计学概念\n"
        
        return AssistantResponse(
            response_type=ResponseType.TEXT,
            content=content,
            confidence=0.8,
            metadata={'intent_type': intent.intent_type.value}
        )
    
    def _generate_default_response(self, intent: UserIntent, 
                                 context: Optional[Dict[str, Any]]) -> AssistantResponse:
        """生成默认响应"""
        content = "抱歉，我没有完全理解您的问题。\n\n"
        content += "您可以尝试这样问我：\n"
        content += "• \"预测双色球下期号码\"\n"
        content += "• \"分析最近50期的号码频率\"\n"
        content += "• \"查询2024001期的开奖结果\"\n"
        content += "• \"解释什么是随机森林算法\"\n"
        content += "• \"如何提高预测准确率\"\n\n"
        content += "如果您需要更多帮助，请说\"帮助\"。"
        
        return AssistantResponse(
            response_type=ResponseType.TEXT,
            content=content,
            confidence=0.5,
            metadata={'intent_type': intent.intent_type.value}
        )


class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.conversations = {}  # session_id -> messages
        self.max_history = 50    # 最大历史记录数
    
    def add_message(self, session_id: str, message: ConversationMessage):
        """添加消息"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append(message)
        
        # 限制历史记录长度
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[ConversationMessage]:
        """获取对话历史"""
        if session_id not in self.conversations:
            return []
        
        return self.conversations[session_id][-limit:]
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        history = self.get_conversation_history(session_id, limit=5)
        
        context = {
            'previous_intents': [],
            'mentioned_entities': set(),
            'conversation_length': len(history),
            'last_response_time': None
        }
        
        for message in history:
            if message.metadata:
                if 'intent_type' in message.metadata:
                    context['previous_intents'].append(message.metadata['intent_type'])
                
                if 'entities' in message.metadata:
                    for entity_list in message.metadata['entities'].values():
                        context['mentioned_entities'].update(entity_list)
            
            context['last_response_time'] = message.timestamp
        
        context['mentioned_entities'] = list(context['mentioned_entities'])
        
        return context


class IntelligentAssistant:
    """智能助手主类"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.intent_classifier = IntentClassifier()
        self.knowledge_base = KnowledgeBase()
        self.response_generator = ResponseGenerator(self.knowledge_base)
        self.conversation_manager = ConversationManager()
        
        logger.info("AI智能助手初始化完成")
    
    def process_message(self, user_message: str, session_id: str = "default") -> AssistantResponse:
        """处理用户消息"""
        start_time = time.time()
        
        # 1. 预处理文本
        processed_text = self.nlp_processor.preprocess_text(user_message)
        
        # 2. 提取实体
        entities = self.nlp_processor.extract_entities(user_message)
        
        # 3. 分类意图
        intent = self.intent_classifier.classify_intent(user_message, entities)
        
        # 4. 获取对话上下文
        context = self.conversation_manager.get_context(session_id)
        
        # 5. 生成响应
        response = self.response_generator.generate_response(intent, context)
        
        # 6. 记录对话
        user_msg = ConversationMessage(
            role=ConversationRole.USER,
            content=user_message,
            timestamp=time.time(),
            metadata={
                'entities': entities,
                'intent_type': intent.intent_type.value,
                'intent_confidence': intent.confidence
            }
        )
        
        assistant_msg = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=response.content,
            timestamp=time.time(),
            metadata={
                'response_type': response.response_type.value,
                'confidence': response.confidence,
                'processing_time': time.time() - start_time
            }
        )
        
        self.conversation_manager.add_message(session_id, user_msg)
        self.conversation_manager.add_message(session_id, assistant_msg)
        
        return response
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """获取对话历史"""
        messages = self.conversation_manager.get_conversation_history(session_id)
        return [msg.to_dict() for msg in messages]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """分析情感"""
        return self.nlp_processor.analyze_sentiment(text)
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """获取系统能力"""
        return {
            'nlp_capabilities': {
                'text_preprocessing': True,
                'entity_extraction': True,
                'sentiment_analysis': True,
                'intent_classification': True
            },
            'supported_intents': [intent.value for intent in IntentType],
            'knowledge_domains': list(self.knowledge_base.knowledge.keys()),
            'conversation_management': True,
            'multilingual_support': False,  # 目前主要支持中文
            'available_models': {
                'nltk': NLTK_AVAILABLE,
                'spacy': SPACY_AVAILABLE,
                'transformers': TRANSFORMERS_AVAILABLE,
                'openai': OPENAI_AVAILABLE
            }
        }


# 全局助手实例
_intelligent_assistant = None

def get_intelligent_assistant() -> IntelligentAssistant:
    """获取智能助手实例（单例模式）"""
    global _intelligent_assistant
    if _intelligent_assistant is None:
        _intelligent_assistant = IntelligentAssistant()
    return _intelligent_assistant


def main():
    """测试主函数"""
    print("🤖 测试AI智能助手系统...")
    
    # 创建助手实例
    assistant = get_intelligent_assistant()
    
    # 获取系统能力
    capabilities = assistant.get_system_capabilities()
    print(f"系统能力: {json.dumps(capabilities, indent=2, ensure_ascii=False)}")
    
    # 测试对话
    test_messages = [
        "你好，我想预测双色球下期号码",
        "分析最近30期的号码频率",
        "查询2024001期的开奖结果",
        "什么是随机森林算法？",
        "如何提高预测准确率？",
        "帮助我配置系统参数"
    ]
    
    session_id = "test_session"
    
    print(f"\n开始对话测试（会话ID：{session_id}）：")
    print("=" * 60)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n👤 用户: {message}")
        
        try:
            response = assistant.process_message(message, session_id)
            print(f"🤖 助手: {response.content}")
            
            if response.data:
                print(f"📊 数据: {json.dumps(response.data, indent=2, ensure_ascii=False)}")
            
            print(f"🎯 置信度: {response.confidence:.2%}")
            print(f"📝 响应类型: {response.response_type.value}")
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
        
        print("-" * 40)
    
    # 测试情感分析
    print(f"\n情感分析测试:")
    test_texts = [
        "这个预测结果很好，我很满意",
        "预测不准确，很失望",
        "系统运行正常"
    ]
    
    for text in test_texts:
        sentiment = assistant.analyze_sentiment(text)
        print(f"文本: {text}")
        print(f"情感: {sentiment['sentiment']}, 置信度: {sentiment['confidence']:.2%}")
    
    # 获取对话历史
    print(f"\n对话历史:")
    history = assistant.get_conversation_history(session_id)
    for msg in history[-4:]:  # 显示最后4条消息
        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
        print(f"{role_emoji} {msg['role']}: {msg['content'][:50]}...")
    
    print(f"\n✅ AI智能助手系统测试完成！")
    print(f"📊 处理了 {len(test_messages)} 条消息")
    print(f"💬 对话历史包含 {len(history)} 条记录")


if __name__ == "__main__":
    main()
