"""
实时流处理系统
支持实时数据摄取、处理和预测
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import websockets
import aiohttp
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """流事件类型"""
    DATA_ARRIVAL = "data_arrival"
    PREDICTION_REQUEST = "prediction_request"
    MODEL_UPDATE = "model_update"
    ALERT = "alert"
    SYSTEM_STATUS = "system_status"
    USER_ACTION = "user_action"


class StreamProcessingMode(Enum):
    """流处理模式"""
    BATCH = "batch"           # 批处理模式
    STREAMING = "streaming"   # 流处理模式
    MICRO_BATCH = "micro_batch"  # 微批处理模式


@dataclass
class StreamEvent:
    """流事件"""
    event_id: str
    event_type: StreamEventType
    timestamp: float
    data: Dict[str, Any]
    source: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['event_type'] = self.event_type.value
        return result


class StreamBuffer:
    """流缓冲区"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.buffer = deque(maxlen=max_size)
        self.index = {}  # 用于快速查找
        self.lock = threading.RLock()
    
    def add(self, event: StreamEvent):
        """添加事件"""
        with self.lock:
            # 清理过期事件
            self._cleanup_expired()
            
            # 添加新事件
            self.buffer.append(event)
            self.index[event.event_id] = event
    
    def get_events(self, event_type: Optional[StreamEventType] = None,
                  since: Optional[float] = None,
                  limit: Optional[int] = None) -> List[StreamEvent]:
        """获取事件"""
        with self.lock:
            events = []
            
            for event in reversed(self.buffer):  # 最新的在前
                # 类型过滤
                if event_type and event.event_type != event_type:
                    continue
                
                # 时间过滤
                if since and event.timestamp < since:
                    continue
                
                events.append(event)
                
                # 数量限制
                if limit and len(events) >= limit:
                    break
            
            return events
    
    def _cleanup_expired(self):
        """清理过期事件"""
        current_time = time.time()
        cutoff_time = current_time - self.ttl_seconds
        
        # 从左侧移除过期事件
        while self.buffer and self.buffer[0].timestamp < cutoff_time:
            expired_event = self.buffer.popleft()
            self.index.pop(expired_event.event_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        with self.lock:
            event_types = defaultdict(int)
            for event in self.buffer:
                event_types[event.event_type.value] += 1
            
            return {
                'total_events': len(self.buffer),
                'buffer_utilization': len(self.buffer) / self.max_size,
                'event_types': dict(event_types),
                'oldest_event_age': time.time() - self.buffer[0].timestamp if self.buffer else 0,
                'newest_event_age': time.time() - self.buffer[-1].timestamp if self.buffer else 0
            }


class StreamProcessor:
    """流处理器"""
    
    def __init__(self, processor_id: str, process_func: Callable[[StreamEvent], Any]):
        self.processor_id = processor_id
        self.process_func = process_func
        self.processed_count = 0
        self.error_count = 0
        self.last_processed_time = None
        self.processing_times = deque(maxlen=1000)  # 记录最近1000次处理时间
    
    async def process(self, event: StreamEvent) -> Any:
        """处理事件"""
        start_time = time.time()
        
        try:
            # 执行处理函数
            if asyncio.iscoroutinefunction(self.process_func):
                result = await self.process_func(event)
            else:
                result = self.process_func(event)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.processed_count += 1
            self.last_processed_time = time.time()
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"流处理器 {self.processor_id} 处理失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计"""
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            min_time = min(self.processing_times)
        else:
            avg_time = max_time = min_time = 0
        
        return {
            'processor_id': self.processor_id,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.processed_count + self.error_count, 1),
            'avg_processing_time': avg_time,
            'max_processing_time': max_time,
            'min_processing_time': min_time,
            'last_processed_time': self.last_processed_time
        }


class RealtimeStreamEngine:
    """实时流处理引擎"""
    
    def __init__(self, buffer_size: int = 10000, worker_count: int = 4):
        self.buffer = StreamBuffer(buffer_size)
        self.processors = {}
        self.running = False
        self.worker_count = worker_count
        self.event_queue = asyncio.Queue(maxsize=1000)
        self.subscribers = defaultdict(list)  # 事件订阅者
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'start_time': None,
            'processing_rate': 0.0
        }
        
        # WebSocket连接管理
        self.websocket_clients = set()
        self.websocket_server = None
        
        logger.info("实时流处理引擎初始化完成")
    
    def register_processor(self, event_type: StreamEventType, 
                          processor_id: str, 
                          process_func: Callable[[StreamEvent], Any]):
        """注册流处理器"""
        if event_type not in self.processors:
            self.processors[event_type] = {}
        
        self.processors[event_type][processor_id] = StreamProcessor(processor_id, process_func)
        logger.info(f"注册流处理器: {event_type.value} -> {processor_id}")
    
    def subscribe(self, event_type: StreamEventType, callback: Callable[[StreamEvent], None]):
        """订阅事件"""
        self.subscribers[event_type].append(callback)
        logger.info(f"订阅事件: {event_type.value}")
    
    async def emit_event(self, event: StreamEvent):
        """发送事件"""
        # 添加到缓冲区
        self.buffer.add(event)
        
        # 添加到处理队列
        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            logger.warning("事件队列已满，丢弃事件")
        
        # 通知订阅者
        for callback in self.subscribers.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"事件回调执行失败: {e}")
        
        # 广播到WebSocket客户端
        await self._broadcast_to_websockets(event)
    
    async def _broadcast_to_websockets(self, event: StreamEvent):
        """广播事件到WebSocket客户端"""
        if not self.websocket_clients:
            return
        
        message = json.dumps(event.to_dict(), ensure_ascii=False)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"WebSocket广播失败: {e}")
                disconnected_clients.add(client)
        
        # 清理断开的连接
        self.websocket_clients -= disconnected_clients
    
    async def _process_events(self):
        """处理事件的工作协程"""
        while self.running:
            try:
                # 从队列获取事件
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # 处理事件
                await self._handle_event(event)
                
                # 更新指标
                self.metrics['events_processed'] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"事件处理异常: {e}")
                self.metrics['events_failed'] += 1
    
    async def _handle_event(self, event: StreamEvent):
        """处理单个事件"""
        processors = self.processors.get(event.event_type, {})
        
        if not processors:
            logger.debug(f"没有找到事件类型 {event.event_type.value} 的处理器")
            return
        
        # 并发处理所有相关处理器
        tasks = []
        for processor in processors.values():
            task = asyncio.create_task(processor.process(event))
            tasks.append(task)
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"事件处理失败: {e}")
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """启动WebSocket服务器"""
        async def handle_client(websocket, path):
            """处理WebSocket客户端连接"""
            self.websocket_clients.add(websocket)
            logger.info(f"WebSocket客户端连接: {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.discard(websocket)
                logger.info(f"WebSocket客户端断开: {websocket.remote_address}")
        
        self.websocket_server = await websockets.serve(handle_client, host, port)
        logger.info(f"WebSocket服务器启动: ws://{host}:{port}")
    
    async def start(self):
        """启动流处理引擎"""
        if self.running:
            return
        
        self.running = True
        self.metrics['start_time'] = time.time()
        
        # 启动工作协程
        self.worker_tasks = []
        for i in range(self.worker_count):
            task = asyncio.create_task(self._process_events())
            self.worker_tasks.append(task)
        
        # 启动指标更新任务
        self.metrics_task = asyncio.create_task(self._update_metrics())
        
        logger.info(f"实时流处理引擎启动，工作协程数: {self.worker_count}")
    
    async def stop(self):
        """停止流处理引擎"""
        self.running = False
        
        # 停止工作协程
        if hasattr(self, 'worker_tasks'):
            for task in self.worker_tasks:
                task.cancel()
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # 停止指标任务
        if hasattr(self, 'metrics_task'):
            self.metrics_task.cancel()
        
        # 停止WebSocket服务器
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        logger.info("实时流处理引擎已停止")
    
    async def _update_metrics(self):
        """更新性能指标"""
        last_processed = 0
        last_time = time.time()
        
        while self.running:
            await asyncio.sleep(5)  # 每5秒更新一次
            
            current_time = time.time()
            current_processed = self.metrics['events_processed']
            
            # 计算处理速率
            time_delta = current_time - last_time
            events_delta = current_processed - last_processed
            
            if time_delta > 0:
                self.metrics['processing_rate'] = events_delta / time_delta
            
            last_processed = current_processed
            last_time = current_time
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        # 缓冲区统计
        buffer_stats = self.buffer.get_stats()
        
        # 处理器统计
        processor_stats = {}
        for event_type, processors in self.processors.items():
            processor_stats[event_type.value] = {
                pid: processor.get_stats() 
                for pid, processor in processors.items()
            }
        
        # 系统指标
        uptime = time.time() - self.metrics['start_time'] if self.metrics['start_time'] else 0
        
        return {
            'system': {
                'running': self.running,
                'uptime_seconds': uptime,
                'worker_count': self.worker_count,
                'websocket_clients': len(self.websocket_clients),
                'queue_size': self.event_queue.qsize(),
                'processing_rate': self.metrics['processing_rate'],
                'total_processed': self.metrics['events_processed'],
                'total_failed': self.metrics['events_failed']
            },
            'buffer': buffer_stats,
            'processors': processor_stats
        }


class LotteryStreamAnalyzer:
    """彩票流数据分析器"""
    
    def __init__(self, stream_engine: RealtimeStreamEngine):
        self.stream_engine = stream_engine
        self.analysis_cache = {}
        self.trend_window = deque(maxlen=100)  # 保留最近100个数据点
        
        # 注册分析处理器
        self._register_processors()
    
    def _register_processors(self):
        """注册分析处理器"""
        self.stream_engine.register_processor(
            StreamEventType.DATA_ARRIVAL,
            "trend_analyzer",
            self._analyze_trends
        )
        
        self.stream_engine.register_processor(
            StreamEventType.DATA_ARRIVAL,
            "pattern_detector",
            self._detect_patterns
        )
        
        self.stream_engine.register_processor(
            StreamEventType.PREDICTION_REQUEST,
            "realtime_predictor",
            self._make_prediction
        )
    
    async def _analyze_trends(self, event: StreamEvent) -> Dict[str, Any]:
        """分析数据趋势"""
        data = event.data
        
        if 'numbers' in data:
            numbers = data['numbers']
            
            # 添加到趋势窗口
            self.trend_window.append({
                'timestamp': event.timestamp,
                'numbers': numbers,
                'sum': sum(numbers.get('red', [])) + sum(numbers.get('blue', []))
            })
            
            # 分析趋势
            if len(self.trend_window) >= 10:
                trend_analysis = self._calculate_trends()
                
                # 发送趋势分析事件
                trend_event = StreamEvent(
                    event_id=f"trend_{int(time.time())}",
                    event_type=StreamEventType.SYSTEM_STATUS,
                    timestamp=time.time(),
                    data={
                        'type': 'trend_analysis',
                        'analysis': trend_analysis
                    },
                    source='trend_analyzer'
                )
                
                await self.stream_engine.emit_event(trend_event)
                
                return trend_analysis
        
        return {}
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """计算趋势指标"""
        recent_data = list(self.trend_window)[-10:]  # 最近10期
        
        # 和值趋势
        sums = [item['sum'] for item in recent_data]
        sum_trend = 'rising' if sums[-1] > sums[0] else 'falling'
        sum_volatility = max(sums) - min(sums)
        
        # 号码频率分析
        red_freq = defaultdict(int)
        blue_freq = defaultdict(int)
        
        for item in recent_data:
            numbers = item['numbers']
            for red in numbers.get('red', []):
                red_freq[red] += 1
            for blue in numbers.get('blue', []):
                blue_freq[blue] += 1
        
        # 热门号码
        hot_red = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        hot_blue = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return {
            'sum_trend': sum_trend,
            'sum_volatility': sum_volatility,
            'average_sum': sum(sums) / len(sums),
            'hot_red_numbers': hot_red,
            'hot_blue_numbers': hot_blue,
            'data_points': len(recent_data),
            'analysis_time': time.time()
        }
    
    async def _detect_patterns(self, event: StreamEvent) -> Dict[str, Any]:
        """检测数据模式"""
        # 简化的模式检测
        patterns = {
            'consecutive_numbers': self._check_consecutive_pattern(event.data),
            'odd_even_ratio': self._check_odd_even_pattern(event.data),
            'zone_distribution': self._check_zone_pattern(event.data)
        }
        
        # 如果发现异常模式，发送警报
        if any(patterns.values()):
            alert_event = StreamEvent(
                event_id=f"alert_{int(time.time())}",
                event_type=StreamEventType.ALERT,
                timestamp=time.time(),
                data={
                    'type': 'pattern_alert',
                    'patterns': patterns,
                    'original_data': event.data
                },
                source='pattern_detector',
                priority=2
            )
            
            await self.stream_engine.emit_event(alert_event)
        
        return patterns
    
    def _check_consecutive_pattern(self, data: Dict[str, Any]) -> bool:
        """检查连续号码模式"""
        if 'numbers' in data and 'red' in data['numbers']:
            red_numbers = sorted(data['numbers']['red'])
            consecutive_count = 0
            
            for i in range(1, len(red_numbers)):
                if red_numbers[i] - red_numbers[i-1] == 1:
                    consecutive_count += 1
            
            return consecutive_count >= 3  # 3个或更多连续号码
        return False
    
    def _check_odd_even_pattern(self, data: Dict[str, Any]) -> bool:
        """检查奇偶比例模式"""
        if 'numbers' in data and 'red' in data['numbers']:
            red_numbers = data['numbers']['red']
            odd_count = sum(1 for n in red_numbers if n % 2 == 1)
            even_count = len(red_numbers) - odd_count
            
            # 检查极端比例（全奇数或全偶数）
            return odd_count == 0 or even_count == 0
        return False
    
    def _check_zone_pattern(self, data: Dict[str, Any]) -> bool:
        """检查区间分布模式"""
        if 'numbers' in data and 'red' in data['numbers']:
            red_numbers = data['numbers']['red']
            
            # 分为三个区间: 1-11, 12-22, 23-33
            zone1 = sum(1 for n in red_numbers if 1 <= n <= 11)
            zone2 = sum(1 for n in red_numbers if 12 <= n <= 22)
            zone3 = sum(1 for n in red_numbers if 23 <= n <= 33)
            
            # 检查是否有区间为空
            return zone1 == 0 or zone2 == 0 or zone3 == 0
        return False
    
    async def _make_prediction(self, event: StreamEvent) -> Dict[str, Any]:
        """实时预测"""
        # 获取最近的趋势数据
        recent_trends = list(self.trend_window)[-20:] if len(self.trend_window) >= 20 else list(self.trend_window)
        
        if not recent_trends:
            return {'error': 'insufficient_data'}
        
        # 简化的预测逻辑
        prediction = self._generate_realtime_prediction(recent_trends)
        
        # 发送预测结果事件
        prediction_event = StreamEvent(
            event_id=f"prediction_{int(time.time())}",
            event_type=StreamEventType.SYSTEM_STATUS,
            timestamp=time.time(),
            data={
                'type': 'prediction_result',
                'prediction': prediction,
                'confidence': prediction.get('confidence', 0.5),
                'method': 'realtime_stream_analysis'
            },
            source='realtime_predictor'
        )
        
        await self.stream_engine.emit_event(prediction_event)
        
        return prediction
    
    def _generate_realtime_prediction(self, recent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成实时预测"""
        # 分析最近数据的统计特征
        all_red = []
        all_blue = []
        
        for item in recent_data:
            numbers = item['numbers']
            all_red.extend(numbers.get('red', []))
            all_blue.extend(numbers.get('blue', []))
        
        # 计算频率
        red_freq = defaultdict(int)
        blue_freq = defaultdict(int)
        
        for red in all_red:
            red_freq[red] += 1
        for blue in all_blue:
            blue_freq[blue] += 1
        
        # 基于频率和趋势的简单预测
        # 选择中等频率的号码（避免过热和过冷）
        red_candidates = sorted(red_freq.items(), key=lambda x: x[1])
        blue_candidates = sorted(blue_freq.items(), key=lambda x: x[1])
        
        # 选择中间频率的号码
        mid_start = len(red_candidates) // 3
        mid_end = 2 * len(red_candidates) // 3
        selected_red = [num for num, _ in red_candidates[mid_start:mid_end]][:6]
        
        mid_start_blue = len(blue_candidates) // 3
        mid_end_blue = 2 * len(blue_candidates) // 3
        selected_blue = [num for num, _ in blue_candidates[mid_start_blue:mid_end_blue]][:1]
        
        # 如果选择的号码不够，随机补充
        import random
        if len(selected_red) < 6:
            all_possible_red = set(range(1, 34))
            remaining = all_possible_red - set(selected_red)
            selected_red.extend(random.sample(list(remaining), 6 - len(selected_red)))
        
        if len(selected_blue) < 1:
            all_possible_blue = set(range(1, 17))
            remaining_blue = all_possible_blue - set(selected_blue)
            selected_blue.extend(random.sample(list(remaining_blue), 1 - len(selected_blue)))
        
        # 计算置信度（基于数据量和趋势稳定性）
        confidence = min(0.8, len(recent_data) / 50.0)  # 数据越多置信度越高
        
        return {
            'red_numbers': sorted(selected_red[:6]),
            'blue_numbers': sorted(selected_blue[:1]),
            'confidence': confidence,
            'data_points_used': len(recent_data),
            'prediction_time': time.time(),
            'method': 'frequency_trend_analysis'
        }


# 全局流处理引擎实例
_stream_engine = None

async def get_stream_engine() -> RealtimeStreamEngine:
    """获取流处理引擎实例（单例模式）"""
    global _stream_engine
    if _stream_engine is None:
        _stream_engine = RealtimeStreamEngine()
        await _stream_engine.start()
    return _stream_engine


async def main():
    """测试主函数"""
    # 创建流处理引擎
    engine = RealtimeStreamEngine()
    
    # 创建分析器
    analyzer = LotteryStreamAnalyzer(engine)
    
    # 启动引擎
    await engine.start()
    
    # 启动WebSocket服务器
    await engine.start_websocket_server()
    
    # 模拟数据流
    async def simulate_data_stream():
        import random
        
        for i in range(10):
            # 模拟新数据到达
            event = StreamEvent(
                event_id=f"data_{i}",
                event_type=StreamEventType.DATA_ARRIVAL,
                timestamp=time.time(),
                data={
                    'period': f"2024{i+1:03d}",
                    'numbers': {
                        'red': sorted(random.sample(range(1, 34), 6)),
                        'blue': [random.randint(1, 16)]
                    }
                },
                source='data_simulator'
            )
            
            await engine.emit_event(event)
            await asyncio.sleep(2)
        
        # 模拟预测请求
        prediction_event = StreamEvent(
            event_id="prediction_request_1",
            event_type=StreamEventType.PREDICTION_REQUEST,
            timestamp=time.time(),
            data={'request_type': 'next_period'},
            source='user'
        )
        
        await engine.emit_event(prediction_event)
    
    # 启动数据模拟
    data_task = asyncio.create_task(simulate_data_stream())
    
    # 运行一段时间
    await asyncio.sleep(30)
    
    # 输出统计信息
    stats = engine.get_system_stats()
    print("系统统计信息:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 停止引擎
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
