"""
增强可视化系统
支持3D图表、交互式可视化、动态图表等高级功能
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time
from datetime import datetime, timedelta
import logging

# 图表库导入
try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = px = ff = make_subplots = pyo = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = animation = FigureCanvas = Figure = sns = None

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QLabel, QPushButton
    from PyQt5.QtCore import QTimer, pyqtSignal
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartType:
    """图表类型常量"""
    LINE_3D = "line_3d"
    SCATTER_3D = "scatter_3d"
    SURFACE_3D = "surface_3d"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    SANKEY = "sankey"
    WATERFALL = "waterfall"
    VIOLIN = "violin"
    BOX_3D = "box_3d"
    NETWORK = "network"
    GANTT = "gantt"
    CANDLESTICK = "candlestick"
    FUNNEL = "funnel"
    RADAR = "radar"


class InteractiveChart:
    """交互式图表基类"""
    
    def __init__(self, chart_id: str, title: str = ""):
        self.chart_id = chart_id
        self.title = title
        self.data = {}
        self.config = {}
        self.figure = None
        self.callbacks = {}
    
    def set_data(self, data: Dict[str, Any]):
        """设置图表数据"""
        self.data = data
    
    def set_config(self, config: Dict[str, Any]):
        """设置图表配置"""
        self.config.update(config)
    
    def add_callback(self, event: str, callback):
        """添加交互回调"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def create_figure(self) -> Any:
        """创建图表（子类实现）"""
        raise NotImplementedError
    
    def update_data(self, new_data: Dict[str, Any]):
        """更新图表数据"""
        self.data.update(new_data)
        if self.figure:
            self.refresh()
    
    def refresh(self):
        """刷新图表"""
        self.figure = self.create_figure()
    
    def to_html(self) -> str:
        """导出为HTML"""
        if not self.figure:
            self.create_figure()
        
        if PLOTLY_AVAILABLE and hasattr(self.figure, 'to_html'):
            return self.figure.to_html()
        return ""
    
    def save_image(self, filename: str, format: str = "png"):
        """保存为图片"""
        if not self.figure:
            self.create_figure()
        
        if PLOTLY_AVAILABLE and hasattr(self.figure, 'write_image'):
            self.figure.write_image(filename, format=format)
        elif MATPLOTLIB_AVAILABLE and hasattr(self.figure, 'savefig'):
            self.figure.savefig(filename, format=format, dpi=300, bbox_inches='tight')


class LotteryTrend3D(InteractiveChart):
    """3D趋势图表"""
    
    def create_figure(self):
        """创建3D趋势图"""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly不可用，无法创建3D图表")
            return None
        
        history_data = self.data.get('history_data', [])
        if not history_data:
            return None
        
        # 准备数据
        periods = []
        red_sums = []
        blue_sums = []
        dates = []
        
        for item in history_data:
            periods.append(item.get('period', ''))
            red_numbers = item.get('numbers', {}).get('red', [])
            blue_numbers = item.get('numbers', {}).get('blue', [])
            red_sums.append(sum(red_numbers))
            blue_sums.append(sum(blue_numbers))
            dates.append(item.get('date', ''))
        
        # 创建3D散点图
        fig = go.Figure()
        
        # 添加红球趋势
        fig.add_trace(go.Scatter3d(
            x=list(range(len(periods))),
            y=red_sums,
            z=[1] * len(red_sums),  # Z轴固定为1
            mode='markers+lines',
            marker=dict(
                size=8,
                color=red_sums,
                colorscale='Reds',
                colorbar=dict(title="红球和值", x=1.1)
            ),
            line=dict(color='red', width=4),
            name='红球趋势',
            hovertemplate='期号: %{text}<br>红球和值: %{y}<extra></extra>',
            text=periods
        ))
        
        # 添加蓝球趋势
        fig.add_trace(go.Scatter3d(
            x=list(range(len(periods))),
            y=blue_sums,
            z=[0] * len(blue_sums),  # Z轴固定为0
            mode='markers+lines',
            marker=dict(
                size=6,
                color=blue_sums,
                colorscale='Blues',
                colorbar=dict(title="蓝球值", x=1.2)
            ),
            line=dict(color='blue', width=3),
            name='蓝球趋势',
            hovertemplate='期号: %{text}<br>蓝球值: %{y}<extra></extra>',
            text=periods
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{self.title} - 3D趋势分析",
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title="期号序列",
                yaxis_title="号码和值",
                zaxis_title="球类型",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1, z=0.5)
            ),
            width=1000,
            height=700,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        self.figure = fig
        return fig


class NumberHeatmap(InteractiveChart):
    """号码热力图"""
    
    def create_figure(self):
        """创建号码热力图"""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly不可用，无法创建热力图")
            return None
        
        history_data = self.data.get('history_data', [])
        if not history_data:
            return None
        
        # 统计号码频率
        red_freq = {}
        blue_freq = {}
        
        for item in history_data:
            numbers = item.get('numbers', {})
            for red in numbers.get('red', []):
                red_freq[red] = red_freq.get(red, 0) + 1
            for blue in numbers.get('blue', []):
                blue_freq[blue] = blue_freq.get(blue, 0) + 1
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('红球出现频率热力图', '蓝球出现频率热力图'),
            vertical_spacing=0.1
        )
        
        # 红球热力图数据
        red_matrix = []
        red_numbers = []
        for i in range(1, 34):  # 红球1-33
            red_numbers.append(str(i))
        
        # 创建5行7列的矩阵（33个红球 + 2个空位）
        red_data = []
        for i in range(5):
            row = []
            for j in range(7):
                num = i * 7 + j + 1
                if num <= 33:
                    row.append(red_freq.get(num, 0))
                else:
                    row.append(0)
            red_data.append(row)
        
        # 添加红球热力图
        fig.add_trace(
            go.Heatmap(
                z=red_data,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="出现次数", len=0.4, y=0.8),
                hovertemplate='号码: %{text}<br>出现次数: %{z}<extra></extra>',
                text=[[str(i*7+j+1) if i*7+j+1 <= 33 else '' 
                      for j in range(7)] for i in range(5)]
            ),
            row=1, col=1
        )
        
        # 蓝球热力图数据
        blue_data = [[blue_freq.get(i, 0) for i in range(1, 17)]]  # 蓝球1-16
        
        # 添加蓝球热力图
        fig.add_trace(
            go.Heatmap(
                z=blue_data,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="出现次数", len=0.4, y=0.2),
                hovertemplate='号码: %{text}<br>出现次数: %{z}<extra></extra>',
                text=[[str(i) for i in range(1, 17)]]
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{self.title} - 号码出现频率热力图",
                x=0.5,
                font=dict(size=18)
            ),
            height=800,
            width=1000
        )
        
        # 隐藏坐标轴
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        self.figure = fig
        return fig


class NumberNetworkGraph(InteractiveChart):
    """号码关联网络图"""
    
    def create_figure(self):
        """创建号码关联网络图"""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly不可用，无法创建网络图")
            return None
        
        history_data = self.data.get('history_data', [])
        if not history_data:
            return None
        
        # 计算号码共现矩阵
        cooccurrence = {}
        
        for item in history_data:
            red_numbers = item.get('numbers', {}).get('red', [])
            
            # 计算红球之间的共现关系
            for i, num1 in enumerate(red_numbers):
                for j, num2 in enumerate(red_numbers):
                    if i != j:
                        key = tuple(sorted([num1, num2]))
                        cooccurrence[key] = cooccurrence.get(key, 0) + 1
        
        # 构建网络图数据
        edges = []
        weights = []
        nodes = set()
        
        # 只显示共现次数大于阈值的关系
        threshold = max(1, len(history_data) // 20)  # 动态阈值
        
        for (num1, num2), count in cooccurrence.items():
            if count >= threshold:
                edges.append((num1, num2))
                weights.append(count)
                nodes.add(num1)
                nodes.add(num2)
        
        if not edges:
            logger.warning("没有足够的共现数据创建网络图")
            return None
        
        # 使用简单的圆形布局
        import math
        node_list = sorted(nodes)
        n = len(node_list)
        node_positions = {}
        
        for i, node in enumerate(node_list):
            angle = 2 * math.pi * i / n
            x = math.cos(angle)
            y = math.sin(angle)
            node_positions[node] = (x, y)
        
        # 创建边的轨迹
        edge_trace = []
        for i, (num1, num2) in enumerate(edges):
            x0, y0 = node_positions[num1]
            x1, y1 = node_positions[num2]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weights[i]/max(weights)*5, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # 创建节点轨迹
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        # 计算节点大小（基于度数）
        node_degrees = {}
        for num1, num2 in edges:
            node_degrees[num1] = node_degrees.get(num1, 0) + 1
            node_degrees[num2] = node_degrees.get(num2, 0) + 1
        
        for node in node_list:
            x, y = node_positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            node_size.append(10 + node_degrees.get(node, 0) * 2)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=12, color="white"),
            marker=dict(
                size=node_size,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='号码: %{text}<br>连接数: %{marker.size}<extra></extra>',
            name='红球号码'
        )
        
        # 创建图表
        fig = go.Figure(data=[node_trace] + edge_trace)
        
        fig.update_layout(
            title=dict(
                text=f"{self.title} - 号码关联网络图",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text=f"显示共现次数 ≥ {threshold} 的号码关系",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        self.figure = fig
        return fig


class PredictionConfidenceSurface(InteractiveChart):
    """预测置信度3D表面图"""
    
    def create_figure(self):
        """创建3D表面图"""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly不可用，无法创建3D表面图")
            return None
        
        prediction_data = self.data.get('prediction_data', [])
        if not prediction_data:
            return None
        
        # 准备数据
        models = []
        periods = []
        confidences = []
        
        for item in prediction_data:
            models.append(item.get('model', 'unknown'))
            periods.append(item.get('period', ''))
            confidences.append(item.get('confidence', 0))
        
        # 创建网格数据
        unique_models = list(set(models))
        unique_periods = sorted(list(set(periods)))
        
        if len(unique_models) < 2 or len(unique_periods) < 2:
            logger.warning("数据不足以创建3D表面图")
            return None
        
        # 构建置信度矩阵
        confidence_matrix = np.zeros((len(unique_models), len(unique_periods)))
        
        for i, model in enumerate(unique_models):
            for j, period in enumerate(unique_periods):
                # 查找对应的置信度
                for item in prediction_data:
                    if item.get('model') == model and item.get('period') == period:
                        confidence_matrix[i, j] = item.get('confidence', 0)
                        break
        
        # 创建3D表面图
        fig = go.Figure(data=[
            go.Surface(
                z=confidence_matrix,
                x=unique_periods,
                y=unique_models,
                colorscale='Viridis',
                colorbar=dict(title="置信度", titleside="right"),
                hovertemplate='模型: %{y}<br>期号: %{x}<br>置信度: %{z:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f"{self.title} - 预测置信度3D表面图",
                x=0.5,
                font=dict(size=18)
            ),
            scene=dict(
                xaxis_title="期号",
                yaxis_title="预测模型",
                zaxis_title="置信度",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            width=1000,
            height=700
        )
        
        self.figure = fig
        return fig


class AnimatedChart:
    """动画图表类"""
    
    def __init__(self, chart_id: str, title: str = ""):
        self.chart_id = chart_id
        self.title = title
        self.frames_data = []
        self.current_frame = 0
        self.animation_speed = 500  # 毫秒
    
    def add_frame(self, frame_data: Dict[str, Any]):
        """添加动画帧"""
        self.frames_data.append(frame_data)
    
    def create_animated_trend(self, history_data: List[Dict[str, Any]]) -> Any:
        """创建动画趋势图"""
        if not PLOTLY_AVAILABLE:
            return None
        
        # 准备动画帧数据
        frames = []
        
        for i in range(1, len(history_data) + 1):
            current_data = history_data[:i]
            
            periods = [item.get('period', '') for item in current_data]
            red_sums = [sum(item.get('numbers', {}).get('red', [])) for item in current_data]
            blue_sums = [sum(item.get('numbers', {}).get('blue', [])) for item in current_data]
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=periods,
                        y=red_sums,
                        mode='lines+markers',
                        name='红球和值',
                        line=dict(color='red'),
                        marker=dict(size=8)
                    ),
                    go.Scatter(
                        x=periods,
                        y=blue_sums,
                        mode='lines+markers',
                        name='蓝球值',
                        line=dict(color='blue'),
                        marker=dict(size=6)
                    )
                ],
                name=f"frame_{i}"
            )
            frames.append(frame)
        
        # 创建初始图表
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # 添加播放控制按钮
        fig.update_layout(
            title=f"{self.title} - 动画趋势图",
            xaxis_title="期号",
            yaxis_title="号码和值",
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": self.animation_speed, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 300,
                                                                         "easing": "quadratic-in-out"}}],
                        "label": "播放",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}],
                        "label": "暂停",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "期数:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"frame_{i}"],
                                {"frame": {"duration": 300, "redraw": True},
                                 "mode": "immediate",
                                 "transition": {"duration": 300}}],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i in range(1, len(frames) + 1)
                ]
            }]
        )
        
        return fig


class EnhancedVisualizationEngine:
    """增强可视化引擎"""
    
    def __init__(self):
        self.charts = {}
        self.themes = {
            'default': self._get_default_theme(),
            'dark': self._get_dark_theme(),
            'colorful': self._get_colorful_theme()
        }
        self.current_theme = 'default'
        
        logger.info("增强可视化引擎初始化完成")
    
    def _get_default_theme(self) -> Dict[str, Any]:
        """默认主题"""
        return {
            'background_color': 'white',
            'grid_color': '#E5E5E5',
            'text_color': '#2E2E2E',
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    
    def _get_dark_theme(self) -> Dict[str, Any]:
        """深色主题"""
        return {
            'background_color': '#2E2E2E',
            'grid_color': '#404040',
            'text_color': '#FFFFFF',
            'color_palette': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']
        }
    
    def _get_colorful_theme(self) -> Dict[str, Any]:
        """彩色主题"""
        return {
            'background_color': '#F8F9FA',
            'grid_color': '#DEE2E6',
            'text_color': '#495057',
            'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        }
    
    def set_theme(self, theme_name: str):
        """设置主题"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            logger.info(f"切换到主题: {theme_name}")
    
    def create_chart(self, chart_type: str, chart_id: str, data: Dict[str, Any], 
                    config: Optional[Dict[str, Any]] = None) -> InteractiveChart:
        """创建图表"""
        config = config or {}
        
        # 应用当前主题
        theme = self.themes[self.current_theme]
        config.update(theme)
        
        if chart_type == ChartType.LINE_3D or chart_type == ChartType.SCATTER_3D:
            chart = LotteryTrend3D(chart_id, config.get('title', '3D趋势图'))
        elif chart_type == ChartType.HEATMAP:
            chart = NumberHeatmap(chart_id, config.get('title', '热力图'))
        elif chart_type == ChartType.NETWORK:
            chart = NumberNetworkGraph(chart_id, config.get('title', '网络图'))
        elif chart_type == ChartType.SURFACE_3D:
            chart = PredictionConfidenceSurface(chart_id, config.get('title', '3D表面图'))
        else:
            logger.warning(f"不支持的图表类型: {chart_type}")
            return None
        
        chart.set_data(data)
        chart.set_config(config)
        chart.create_figure()
        
        self.charts[chart_id] = chart
        return chart
    
    def create_animated_chart(self, chart_id: str, data: List[Dict[str, Any]], 
                            config: Optional[Dict[str, Any]] = None) -> AnimatedChart:
        """创建动画图表"""
        config = config or {}
        
        animated_chart = AnimatedChart(chart_id, config.get('title', '动画图表'))
        
        # 创建动画趋势图
        figure = animated_chart.create_animated_trend(data)
        
        if figure:
            self.charts[chart_id] = animated_chart
        
        return animated_chart
    
    def get_chart(self, chart_id: str) -> Optional[InteractiveChart]:
        """获取图表"""
        return self.charts.get(chart_id)
    
    def update_chart_data(self, chart_id: str, new_data: Dict[str, Any]):
        """更新图表数据"""
        chart = self.charts.get(chart_id)
        if chart:
            chart.update_data(new_data)
    
    def export_chart(self, chart_id: str, filename: str, format: str = "html"):
        """导出图表"""
        chart = self.charts.get(chart_id)
        if not chart:
            logger.error(f"图表 {chart_id} 不存在")
            return
        
        if format == "html":
            html_content = chart.to_html()
            if html_content:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"图表已导出为HTML: {filename}")
        else:
            chart.save_image(filename, format)
            logger.info(f"图表已导出为{format.upper()}: {filename}")
    
    def create_dashboard(self, charts: List[str], layout: str = "grid") -> str:
        """创建仪表板"""
        if not PLOTLY_AVAILABLE:
            return ""
        
        dashboard_charts = [self.charts.get(cid) for cid in charts if cid in self.charts]
        if not dashboard_charts:
            return ""
        
        if layout == "grid":
            # 网格布局
            rows = int(np.ceil(np.sqrt(len(dashboard_charts))))
            cols = int(np.ceil(len(dashboard_charts) / rows))
            
            subplot_titles = [chart.title for chart in dashboard_charts]
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
            )
            
            for i, chart in enumerate(dashboard_charts):
                row = i // cols + 1
                col = i % cols + 1
                
                if chart.figure and hasattr(chart.figure, 'data'):
                    for trace in chart.figure.data:
                        fig.add_trace(trace, row=row, col=col)
            
            fig.update_layout(
                title="彩票分析仪表板",
                height=300 * rows,
                showlegend=False
            )
            
            return fig.to_html()
        
        return ""
    
    def get_chart_statistics(self) -> Dict[str, Any]:
        """获取图表统计信息"""
        stats = {
            'total_charts': len(self.charts),
            'chart_types': {},
            'themes_available': list(self.themes.keys()),
            'current_theme': self.current_theme
        }
        
        for chart in self.charts.values():
            chart_type = type(chart).__name__
            stats['chart_types'][chart_type] = stats['chart_types'].get(chart_type, 0) + 1
        
        return stats


# 全局可视化引擎实例
_visualization_engine = None

def get_visualization_engine() -> EnhancedVisualizationEngine:
    """获取可视化引擎实例（单例模式）"""
    global _visualization_engine
    if _visualization_engine is None:
        _visualization_engine = EnhancedVisualizationEngine()
    return _visualization_engine


def main():
    """测试主函数"""
    # 创建可视化引擎
    engine = get_visualization_engine()
    
    # 模拟数据
    import random
    history_data = []
    for i in range(50):
        history_data.append({
            'period': f"2024{i+1:03d}",
            'date': f"2024-01-{i+1:02d}",
            'numbers': {
                'red': sorted(random.sample(range(1, 34), 6)),
                'blue': [random.randint(1, 16)]
            }
        })
    
    # 创建各种图表
    print("创建3D趋势图...")
    chart_3d = engine.create_chart(
        ChartType.SCATTER_3D,
        "trend_3d",
        {'history_data': history_data},
        {'title': '双色球3D趋势分析'}
    )
    
    print("创建热力图...")
    heatmap = engine.create_chart(
        ChartType.HEATMAP,
        "number_heatmap",
        {'history_data': history_data},
        {'title': '双色球号码频率分析'}
    )
    
    print("创建网络图...")
    network = engine.create_chart(
        ChartType.NETWORK,
        "number_network",
        {'history_data': history_data},
        {'title': '双色球号码关联分析'}
    )
    
    # 创建动画图表
    print("创建动画图表...")
    animated = engine.create_animated_chart(
        "trend_animated",
        history_data,
        {'title': '双色球趋势动画'}
    )
    
    # 导出图表
    print("导出图表...")
    if chart_3d:
        engine.export_chart("trend_3d", "charts_output/trend_3d.html")
    if heatmap:
        engine.export_chart("number_heatmap", "charts_output/heatmap.html")
    if network:
        engine.export_chart("number_network", "charts_output/network.html")
    
    # 创建仪表板
    print("创建仪表板...")
    dashboard_html = engine.create_dashboard(["trend_3d", "number_heatmap", "number_network"])
    if dashboard_html:
        with open("charts_output/dashboard.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)
    
    # 输出统计信息
    stats = engine.get_chart_statistics()
    print("可视化引擎统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
