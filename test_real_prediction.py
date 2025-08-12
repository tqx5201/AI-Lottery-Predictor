#!/usr/bin/env python3
"""
真实数据预测测试脚本
测试数据获取、模型训练和预测功能
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_fetcher():
    """测试数据获取功能"""
    logger.info("=" * 60)
    logger.info("测试数据获取功能")
    logger.info("=" * 60)
    
    try:
        from src.realtime.data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # 测试双色球数据获取
        logger.info("测试双色球数据获取...")
        ssq_data = fetcher.get_latest_data('双色球', 10)
        
        if ssq_data:
            logger.info(f"✅ 成功获取双色球数据: {len(ssq_data)}期")
            logger.info(f"最新期号: {ssq_data[0]['period']}")
            logger.info(f"开奖号码: {ssq_data[0]['numbers']}")
        else:
            logger.warning("❌ 未获取到双色球数据")
        
        # 测试大乐透数据获取
        logger.info("测试大乐透数据获取...")
        dlt_data = fetcher.get_latest_data('大乐透', 10)
        
        if dlt_data:
            logger.info(f"✅ 成功获取大乐透数据: {len(dlt_data)}期")
            logger.info(f"最新期号: {dlt_data[0]['period']}")
            logger.info(f"开奖号码: {dlt_data[0]['numbers']}")
        else:
            logger.warning("❌ 未获取到大乐透数据")
        
        return ssq_data, dlt_data
        
    except Exception as e:
        logger.error(f"❌ 数据获取测试失败: {e}")
        return None, None

def test_model_training_and_prediction(history_data, lottery_type):
    """测试模型训练和预测"""
    logger.info("=" * 60)
    logger.info(f"测试{lottery_type}模型训练和预测")
    logger.info("=" * 60)
    
    if not history_data:
        logger.warning(f"❌ {lottery_type}没有历史数据，跳过测试")
        return
    
    try:
        from src.ml.model_manager import ModelManager
        
        manager = ModelManager()
        
        # 自动选择模型
        recommended_model = manager.auto_select_model(lottery_type, history_data)
        logger.info(f"推荐模型: {recommended_model}")
        
        # 创建模型
        model = manager.create_model(recommended_model, lottery_type)
        if not model:
            logger.error(f"❌ {lottery_type}模型创建失败")
            return
        
        logger.info(f"✅ {lottery_type}模型创建成功")
        
        # 训练模型
        logger.info(f"开始训练{lottery_type}模型...")
        model_key = f"{recommended_model}_{lottery_type}"
        
        success = manager.train_model(model_key, history_data)
        if not success:
            logger.error(f"❌ {lottery_type}模型训练失败")
            return
        
        logger.info(f"✅ {lottery_type}模型训练成功")
        
        # 进行预测
        logger.info(f"使用{lottery_type}模型进行预测...")
        recent_data = history_data[:30]  # 使用最近30期数据
        
        prediction = manager.predict_with_model(model_key, recent_data)
        if prediction and 'numbers' in prediction:
            logger.info(f"✅ {lottery_type}预测成功")
            logger.info(f"预测号码: {prediction['numbers']}")
            logger.info(f"预测置信度: {prediction.get('confidence', 'N/A')}")
        else:
            logger.error(f"❌ {lottery_type}预测失败")
        
        # 获取模型性能指标
        performance = model.get_performance_metrics()
        if performance:
            logger.info(f"模型性能指标: {performance}")
        
    except Exception as e:
        logger.error(f"❌ {lottery_type}模型测试失败: {e}")

def test_database_functionality():
    """测试数据库功能"""
    logger.info("=" * 60)
    logger.info("测试数据库功能")
    logger.info("=" * 60)
    
    try:
        from src.core.database_manager import DatabaseManager
        
        db = DatabaseManager()
        
        # 测试数据库连接
        logger.info("测试数据库连接...")
        if db.test_connection():
            logger.info("✅ 数据库连接成功")
        else:
            logger.error("❌ 数据库连接失败")
            return False
        
        # 测试表创建
        logger.info("测试表结构...")
        db.create_tables()
        logger.info("✅ 表结构创建/验证成功")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据库测试失败: {e}")
        return False

def test_analysis_functionality(history_data, lottery_type):
    """测试分析功能"""
    logger.info("=" * 60)
    logger.info(f"测试{lottery_type}分析功能")
    logger.info("=" * 60)
    
    if not history_data:
        logger.warning(f"❌ {lottery_type}没有历史数据，跳过分析测试")
        return
    
    try:
        from src.analysis.lottery_analysis import LotteryAnalysis
        
        analyzer = LotteryAnalysis()
        
        # 测试基础分析
        logger.info(f"进行{lottery_type}基础分析...")
        analysis_result = analyzer.analyze_lottery_data(lottery_type, history_data)
        
        if analysis_result:
            logger.info(f"✅ {lottery_type}基础分析成功")
            
            # 显示部分分析结果
            if 'frequency_analysis' in analysis_result:
                freq_data = analysis_result['frequency_analysis']
                logger.info(f"频率分析 - 热门号码: {freq_data.get('hot_numbers', [])[:5]}")
                logger.info(f"频率分析 - 冷门号码: {freq_data.get('cold_numbers', [])[:5]}")
            
            if 'trend_analysis' in analysis_result:
                trend_data = analysis_result['trend_analysis']
                logger.info(f"趋势分析 - 趋势方向: {trend_data.get('trend_direction', 'N/A')}")
                logger.info(f"趋势分析 - 波动性: {trend_data.get('volatility', 'N/A')}")
        else:
            logger.error(f"❌ {lottery_type}基础分析失败")
        
    except Exception as e:
        logger.error(f"❌ {lottery_type}分析功能测试失败: {e}")

def test_ai_assistant():
    """测试AI助手功能"""
    logger.info("=" * 60)
    logger.info("测试AI助手功能")
    logger.info("=" * 60)
    
    try:
        from src.ai_assistant.intelligent_assistant import get_intelligent_assistant
        
        assistant = get_intelligent_assistant()
        
        # 测试基本对话
        test_messages = [
            "你好",
            "预测双色球下期号码",
            "分析最近30期的频率",
            "什么是机器学习？"
        ]
        
        for msg in test_messages:
            logger.info(f"用户输入: {msg}")
            response = assistant.process_message(msg)
            logger.info(f"助手回复: {response.content[:100]}...")
            logger.info(f"置信度: {response.confidence}")
        
        logger.info("✅ AI助手功能测试完成")
        
    except Exception as e:
        logger.error(f"❌ AI助手测试失败: {e}")

def test_streaming_functionality():
    """测试流处理功能"""
    logger.info("=" * 60)
    logger.info("测试实时流处理功能")
    logger.info("=" * 60)
    
    try:
        import asyncio
        from src.streaming.realtime_processor import get_stream_engine, StreamEvent, StreamEventType
        
        async def run_streaming_test():
            # 获取流处理引擎
            engine = await get_stream_engine()
            
            # 创建测试事件
            test_event = StreamEvent(
                event_id="test_001",
                event_type=StreamEventType.DATA_ARRIVAL,
                timestamp=datetime.now().timestamp(),
                data={"test": "streaming_data"},
                source="test"
            )
            
            # 发送事件
            await engine.emit_event(test_event)
            
            # 等待处理
            await asyncio.sleep(0.1)
            
            # 获取统计信息
            stats = engine.get_system_stats()
            logger.info(f"✅ 流处理测试成功")
            logger.info(f"处理统计: {stats}")
            
            # 停止引擎
            await engine.stop()
        
        # 运行异步测试
        asyncio.run(run_streaming_test())
        
    except Exception as e:
        logger.error(f"❌ 流处理测试失败: {e}")

def main():
    """主测试函数"""
    logger.info("🎊 开始AI彩票预测系统真实数据测试")
    logger.info("测试时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 1. 测试数据获取
    ssq_data, dlt_data = test_data_fetcher()
    
    # 2. 测试数据库功能
    db_ok = test_database_functionality()
    
    # 3. 测试分析功能
    if ssq_data:
        test_analysis_functionality(ssq_data, '双色球')
    if dlt_data:
        test_analysis_functionality(dlt_data, '大乐透')
    
    # 4. 测试模型训练和预测
    if ssq_data:
        test_model_training_and_prediction(ssq_data, '双色球')
    if dlt_data:
        test_model_training_and_prediction(dlt_data, '大乐透')
    
    # 5. 测试AI助手
    test_ai_assistant()
    
    # 6. 测试流处理功能
    test_streaming_functionality()
    
    logger.info("=" * 60)
    logger.info("🎉 所有测试完成！")
    logger.info("=" * 60)
    
    # 总结
    summary = []
    if ssq_data:
        summary.append(f"✅ 双色球数据获取: {len(ssq_data)}期")
    else:
        summary.append("❌ 双色球数据获取失败")
    
    if dlt_data:
        summary.append(f"✅ 大乐透数据获取: {len(dlt_data)}期")
    else:
        summary.append("❌ 大乐透数据获取失败")
    
    if db_ok:
        summary.append("✅ 数据库功能正常")
    else:
        summary.append("❌ 数据库功能异常")
    
    logger.info("测试总结:")
    for item in summary:
        logger.info(f"  {item}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试运行异常: {e}")
        sys.exit(1)
