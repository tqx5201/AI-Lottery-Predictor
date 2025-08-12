"""
AI彩票预测系统 - PyQt5图表组件
提供可嵌入PyQt5界面的matplotlib图表组件
"""

import sys
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSplitter, QTabWidget, QFrame, QScrollArea, QSizePolicy, QGroupBox,
    QProgressBar, QMessageBox, QFileDialog, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# 添加路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from analysis.lottery_visualization import LotteryVisualization
    from analysis.prediction_statistics import PredictionStatistics
    from analysis.lottery_analysis import LotteryAnalysis
    from core.database_adapter import DatabaseAdapter
except ImportError:
    # 回退到相对导入方式
    sys.path.append('..')  # 添加上级目录到路径
    from analysis.lottery_visualization import LotteryVisualization
    from analysis.prediction_statistics import PredictionStatistics
    from analysis.lottery_analysis import LotteryAnalysis
    from core.database_adapter import DatabaseAdapter

# 设置matplotlib的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class MatplotlibCanvas(FigureCanvas):
    """matplotlib画布基类"""
    
    def __init__(self, figure=None, parent=None):
        if figure is None:
            self.figure = Figure(figsize=(12, 8), dpi=100)
        else:
            self.figure = figure
        
        super().__init__(self.figure)
        self.setParent(parent)
        
        # 设置画布属性
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        # 设置背景色
        self.figure.patch.set_facecolor('white')
    
    def clear_figure(self):
        """清空图表"""
        self.figure.clear()
        self.draw()


class ChartUpdateThread(QThread):
    """图表更新线程"""
    finished = pyqtSignal(object)  # 完成信号，传递结果
    error = pyqtSignal(str)       # 错误信号
    progress = pyqtSignal(int)    # 进度信号
    
    def __init__(self, chart_type, data_params, parent=None):
        super().__init__(parent)
        self.chart_type = chart_type
        self.data_params = data_params
        self.visualization = LotteryVisualization()
        self.statistics = PredictionStatistics()
        self.analysis = LotteryAnalysis()
    
    def run(self):
        """运行线程"""
        try:
            self.progress.emit(10)
            
            if self.chart_type == 'frequency':
                result = self._create_frequency_chart()
            elif self.chart_type == 'trend':
                result = self._create_trend_chart()
            elif self.chart_type == 'accuracy':
                result = self._create_accuracy_chart()
            elif self.chart_type == 'analysis':
                result = self._create_analysis_chart()
            else:
                raise ValueError(f"不支持的图表类型: {self.chart_type}")
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _create_frequency_chart(self):
        """创建频率图表"""
        self.progress.emit(30)
        
        lottery_type = self.data_params.get('lottery_type', '双色球')
        period_range = self.data_params.get('period_range', '最近100期')
        chart_type = self.data_params.get('chart_type', 'bar')
        
        # 模拟频率数据（实际应用中从数据库获取）
        if lottery_type == "双色球":
            frequency_data = {
                'red_balls': {str(i): np.random.randint(5, 25) for i in range(1, 34)},
                'blue_balls': {str(i): np.random.randint(3, 15) for i in range(1, 17)}
            }
        else:
            frequency_data = {
                'front_balls': {str(i): np.random.randint(4, 20) for i in range(1, 36)},
                'back_balls': {str(i): np.random.randint(3, 12) for i in range(1, 13)}
            }
        
        self.progress.emit(70)
        
        fig = self.visualization.create_number_frequency_chart(
            frequency_data, lottery_type, chart_type
        )
        
        return fig
    
    def _create_trend_chart(self):
        """创建趋势图表"""
        self.progress.emit(30)
        
        lottery_type = self.data_params.get('lottery_type', '双色球')
        trend_type = self.data_params.get('trend_type', 'frequency')
        
        # 模拟历史数据
        history_data = []
        for i in range(50):
            if lottery_type == "双色球":
                red_nums = sorted(np.random.choice(range(1, 34), 6, replace=False))
                blue_num = np.random.choice(range(1, 17))
                history_data.append({
                    'period': f'2024{i+1:03d}',
                    'numbers': {
                        'red': ','.join(map(str, red_nums)),
                        'blue': str(blue_num)
                    }
                })
            else:
                front_nums = sorted(np.random.choice(range(1, 36), 5, replace=False))
                back_nums = sorted(np.random.choice(range(1, 13), 2, replace=False))
                history_data.append({
                    'period': f'2024{i+1:03d}',
                    'numbers': {
                        'front': ','.join(map(str, front_nums)),
                        'back': ','.join(map(str, back_nums))
                    }
                })
        
        self.progress.emit(70)
        
        fig = self.visualization.create_trend_chart(
            history_data, lottery_type, trend_type
        )
        
        return fig
    
    def _create_accuracy_chart(self):
        """创建准确率图表"""
        self.progress.emit(30)
        
        # 模拟准确率数据
        accuracy_data = {
            'total_predictions': 50,
            'avg_hit_count': 2.5,
            'max_hit_count': 5,
            'avg_accuracy_score': 35.2,
            'by_lottery_type': {
                '双色球': {
                    'count': 30,
                    'avg_hits': 2.3,
                    'avg_accuracy': 32.1
                },
                '大乐透': {
                    'count': 20,
                    'avg_hits': 2.8,
                    'avg_accuracy': 39.5
                }
            }
        }
        
        self.progress.emit(70)
        
        fig = self.visualization.create_prediction_accuracy_chart(accuracy_data)
        
        return fig
    
    def _create_analysis_chart(self):
        """创建分析图表"""
        self.progress.emit(30)
        
        lottery_type = self.data_params.get('lottery_type', '双色球')
        
        # 模拟分析数据
        analysis_data = {
            'hot_cold': {
                'hot': [1, 5, 12, 23, 28],
                'cold': [9, 14, 19, 25, 31]
            },
            'missing': {
                'range_0_5': 15,
                'range_6_10': 8,
                'range_11_20': 6,
                'range_20_plus': 4
            },
            'odd_even': {
                'odd_count': 18,
                'even_count': 15
            },
            'big_small': {
                'big_count': 17,
                'small_count': 16
            },
            'sum_analysis': {
                'recent_sums': list(np.random.randint(80, 150, 20))
            },
            'span': {
                'span_0_10': 5,
                'span_11_20': 15,
                'span_21_30': 8,
                'span_30_plus': 2
            },
            'consecutive': {
                'consecutive_distribution': {0: 10, 2: 15, 3: 8, 4: 5}
            },
            'repeat': {
                'repeat_distribution': {0: 8, 1: 12, 2: 8, 3: 2}
            },
            'scores': {
                'regularity_score': 75.5,
                'randomness_score': 68.2,
                'hotness_score': 82.1,
                'stability_score': 71.8,
                'overall_score': 74.4
            }
        }
        
        self.progress.emit(70)
        
        fig = self.visualization.create_comprehensive_analysis_chart(analysis_data)
        
        return fig


class FrequencyChartWidget(QWidget):
    """频率图表组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_adapter = DatabaseAdapter()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 图表画布
        self.canvas = MatplotlibCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("频率分析设置")
        layout = QHBoxLayout()
        
        # 彩票类型选择
        layout.addWidget(QLabel("彩票类型:"))
        self.lottery_type_combo = QComboBox()
        self.lottery_type_combo.addItems(["双色球", "大乐透"])
        layout.addWidget(self.lottery_type_combo)
        
        # 期数选择
        layout.addWidget(QLabel("期数范围:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["最近50期", "最近100期", "最近200期", "最近500期"])
        self.period_combo.setCurrentText("最近100期")
        layout.addWidget(self.period_combo)
        
        # 图表类型选择
        layout.addWidget(QLabel("图表类型:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["条形图", "热力图"])
        layout.addWidget(self.chart_type_combo)
        
        # 更新按钮
        self.update_button = QPushButton("🔄 更新图表")
        self.update_button.clicked.connect(self.update_chart)
        layout.addWidget(self.update_button)
        
        # 导出按钮
        self.export_button = QPushButton("💾 导出图表")
        self.export_button.clicked.connect(self.export_chart)
        layout.addWidget(self.export_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def update_chart(self):
        """更新图表"""
        try:
            self.update_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            
            # 获取参数
            lottery_type = self.lottery_type_combo.currentText()
            period_range = self.period_combo.currentText()
            chart_type = 'bar' if self.chart_type_combo.currentText() == '条形图' else 'heatmap'
            
            # 启动更新线程
            self.update_thread = ChartUpdateThread(
                'frequency',
                {
                    'lottery_type': lottery_type,
                    'period_range': period_range,
                    'chart_type': chart_type
                }
            )
            
            self.update_thread.progress.connect(self.progress_bar.setValue)
            self.update_thread.finished.connect(self.on_chart_updated)
            self.update_thread.error.connect(self.on_update_error)
            self.update_thread.start()
            
        except Exception as e:
            self.on_update_error(str(e))
    
    def on_chart_updated(self, figure):
        """图表更新完成"""
        try:
            self.canvas.figure.clear()
            
            # 复制新图表到画布
            for i, ax in enumerate(figure.axes):
                new_ax = self.canvas.figure.add_subplot(len(figure.axes), 1, i+1)
                
                # 复制轴的内容
                for line in ax.lines:
                    new_ax.plot(line.get_xdata(), line.get_ydata(), 
                               color=line.get_color(), linewidth=line.get_linewidth(),
                               label=line.get_label())
                
                for patch in ax.patches:
                    new_ax.add_patch(patch)
                
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.grid(True, alpha=0.3)
                
                if ax.legend_:
                    new_ax.legend()
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.on_update_error(f"更新图表显示失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.update_button.setEnabled(True)
    
    def on_update_error(self, error_msg):
        """更新错误处理"""
        QMessageBox.warning(self, "更新失败", f"图表更新失败:\n{error_msg}")
        self.progress_bar.setVisible(False)
        self.update_button.setEnabled(True)
    
    def export_chart(self):
        """导出图表"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "导出图表", 
                f"frequency_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG文件 (*.png);;PDF文件 (*.pdf);;SVG文件 (*.svg)"
            )
            
            if filename:
                self.canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "导出成功", f"图表已导出到:\n{filename}")
                
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"图表导出失败:\n{str(e)}")


class TrendChartWidget(QWidget):
    """趋势图表组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 图表画布
        self.canvas = MatplotlibCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("走势分析设置")
        layout = QHBoxLayout()
        
        # 彩票类型选择
        layout.addWidget(QLabel("彩票类型:"))
        self.lottery_type_combo = QComboBox()
        self.lottery_type_combo.addItems(["双色球", "大乐透"])
        layout.addWidget(self.lottery_type_combo)
        
        # 趋势类型选择
        layout.addWidget(QLabel("趋势类型:"))
        self.trend_type_combo = QComboBox()
        self.trend_type_combo.addItems(["频率走势", "和值走势", "形态走势"])
        layout.addWidget(self.trend_type_combo)
        
        # 期数设置
        layout.addWidget(QLabel("显示期数:"))
        self.period_spin = QSpinBox()
        self.period_spin.setRange(20, 500)
        self.period_spin.setValue(100)
        layout.addWidget(self.period_spin)
        
        # 更新按钮
        self.update_button = QPushButton("🔄 更新图表")
        self.update_button.clicked.connect(self.update_chart)
        layout.addWidget(self.update_button)
        
        # 导出按钮
        self.export_button = QPushButton("💾 导出图表")
        self.export_button.clicked.connect(self.export_chart)
        layout.addWidget(self.export_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def update_chart(self):
        """更新图表"""
        try:
            self.update_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            
            # 获取参数
            lottery_type = self.lottery_type_combo.currentText()
            trend_type_map = {
                "频率走势": "frequency",
                "和值走势": "sum",
                "形态走势": "pattern"
            }
            trend_type = trend_type_map[self.trend_type_combo.currentText()]
            
            # 启动更新线程
            self.update_thread = ChartUpdateThread(
                'trend',
                {
                    'lottery_type': lottery_type,
                    'trend_type': trend_type,
                    'period_count': self.period_spin.value()
                }
            )
            
            self.update_thread.progress.connect(self.progress_bar.setValue)
            self.update_thread.finished.connect(self.on_chart_updated)
            self.update_thread.error.connect(self.on_update_error)
            self.update_thread.start()
            
        except Exception as e:
            self.on_update_error(str(e))
    
    def on_chart_updated(self, figure):
        """图表更新完成"""
        try:
            # 清空并更新画布
            self.canvas.figure.clear()
            
            # 复制图表内容
            ax = self.canvas.figure.add_subplot(111)
            source_ax = figure.axes[0]
            
            # 复制线条
            for line in source_ax.lines:
                ax.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), linewidth=line.get_linewidth(),
                       marker=line.get_marker(), label=line.get_label())
            
            ax.set_title(source_ax.get_title())
            ax.set_xlabel(source_ax.get_xlabel())
            ax.set_ylabel(source_ax.get_ylabel())
            ax.grid(True, alpha=0.3)
            
            if source_ax.legend_:
                ax.legend()
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.on_update_error(f"更新图表显示失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.update_button.setEnabled(True)
    
    def on_update_error(self, error_msg):
        """更新错误处理"""
        QMessageBox.warning(self, "更新失败", f"图表更新失败:\n{error_msg}")
        self.progress_bar.setVisible(False)
        self.update_button.setEnabled(True)
    
    def export_chart(self):
        """导出图表"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "导出图表",
                f"trend_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG文件 (*.png);;PDF文件 (*.pdf);;SVG文件 (*.svg)"
            )
            
            if filename:
                self.canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "导出成功", f"图表已导出到:\n{filename}")
                
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"图表导出失败:\n{str(e)}")


class AccuracyChartWidget(QWidget):
    """准确率图表组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 图表画布
        self.canvas = MatplotlibCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("预测准确率统计")
        layout = QHBoxLayout()
        
        # 统计天数
        layout.addWidget(QLabel("统计天数:"))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(7, 365)
        self.days_spin.setValue(30)
        layout.addWidget(self.days_spin)
        
        # 模型筛选
        layout.addWidget(QLabel("AI模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["全部模型", "deepseek-chat", "qwen-turbo", "qwen-plus"])
        layout.addWidget(self.model_combo)
        
        # 彩票类型筛选
        layout.addWidget(QLabel("彩票类型:"))
        self.lottery_filter_combo = QComboBox()
        self.lottery_filter_combo.addItems(["全部类型", "双色球", "大乐透"])
        layout.addWidget(self.lottery_filter_combo)
        
        # 更新按钮
        self.update_button = QPushButton("🔄 更新统计")
        self.update_button.clicked.connect(self.update_chart)
        layout.addWidget(self.update_button)
        
        # 导出按钮
        self.export_button = QPushButton("💾 导出报告")
        self.export_button.clicked.connect(self.export_report)
        layout.addWidget(self.export_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def update_chart(self):
        """更新图表"""
        try:
            self.update_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            
            # 启动更新线程
            self.update_thread = ChartUpdateThread(
                'accuracy',
                {
                    'days': self.days_spin.value(),
                    'model': self.model_combo.currentText(),
                    'lottery_type': self.lottery_filter_combo.currentText()
                }
            )
            
            self.update_thread.progress.connect(self.progress_bar.setValue)
            self.update_thread.finished.connect(self.on_chart_updated)
            self.update_thread.error.connect(self.on_update_error)
            self.update_thread.start()
            
        except Exception as e:
            self.on_update_error(str(e))
    
    def on_chart_updated(self, figure):
        """图表更新完成"""
        try:
            # 清空并重新绘制
            self.canvas.figure.clear()
            
            # 创建子图网格
            axes = figure.axes
            if len(axes) == 4:
                # 2x2网格
                for i, source_ax in enumerate(axes):
                    ax = self.canvas.figure.add_subplot(2, 2, i+1)
                    
                    # 复制不同类型的图表
                    if source_ax.patches:  # 条形图
                        for patch in source_ax.patches:
                            ax.add_patch(patch)
                    elif len(source_ax.texts) > 0:  # 饼图或文本
                        for text in source_ax.texts:
                            ax.text(text.get_position()[0], text.get_position()[1], 
                                   text.get_text(), ha=text.get_ha(), va=text.get_va())
                    
                    ax.set_title(source_ax.get_title())
                    ax.set_xlabel(source_ax.get_xlabel())
                    ax.set_ylabel(source_ax.get_ylabel())
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.on_update_error(f"更新图表显示失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.update_button.setEnabled(True)
    
    def on_update_error(self, error_msg):
        """更新错误处理"""
        QMessageBox.warning(self, "更新失败", f"统计更新失败:\n{error_msg}")
        self.progress_bar.setVisible(False)
        self.update_button.setEnabled(True)
    
    def export_report(self):
        """导出报告"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "导出准确率报告",
                f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF文件 (*.pdf);;Excel文件 (*.xlsx)"
            )
            
            if filename:
                if filename.endswith('.pdf'):
                    self.canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                elif filename.endswith('.xlsx'):
                    # 导出Excel格式的统计数据
                    # 这里可以调用统计模块的导出功能
                    pass
                
                QMessageBox.information(self, "导出成功", f"报告已导出到:\n{filename}")
                
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"报告导出失败:\n{str(e)}")


class AnalysisChartWidget(QWidget):
    """综合分析图表组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 图表画布
        self.canvas = MatplotlibCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("综合分析设置")
        layout = QHBoxLayout()
        
        # 彩票类型选择
        layout.addWidget(QLabel("彩票类型:"))
        self.lottery_type_combo = QComboBox()
        self.lottery_type_combo.addItems(["双色球", "大乐透"])
        layout.addWidget(self.lottery_type_combo)
        
        # 分析期数
        layout.addWidget(QLabel("分析期数:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["最近50期", "最近100期", "最近200期", "最近500期"])
        self.period_combo.setCurrentText("最近100期")
        layout.addWidget(self.period_combo)
        
        # 强制刷新
        self.force_refresh_check = QCheckBox("强制刷新")
        layout.addWidget(self.force_refresh_check)
        
        # 开始分析按钮
        self.analyze_button = QPushButton("🔍 开始分析")
        self.analyze_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_button)
        
        # 导出按钮
        self.export_button = QPushButton("💾 导出分析")
        self.export_button.clicked.connect(self.export_analysis)
        layout.addWidget(self.export_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def start_analysis(self):
        """开始综合分析"""
        try:
            self.analyze_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            
            # 启动分析线程
            self.update_thread = ChartUpdateThread(
                'analysis',
                {
                    'lottery_type': self.lottery_type_combo.currentText(),
                    'period_range': self.period_combo.currentText(),
                    'force_refresh': self.force_refresh_check.isChecked()
                }
            )
            
            self.update_thread.progress.connect(self.progress_bar.setValue)
            self.update_thread.finished.connect(self.on_analysis_completed)
            self.update_thread.error.connect(self.on_analysis_error)
            self.update_thread.start()
            
        except Exception as e:
            self.on_analysis_error(str(e))
    
    def on_analysis_completed(self, figure):
        """分析完成"""
        try:
            # 清空并绘制新图表
            self.canvas.figure.clear()
            
            # 复制复杂的综合分析图表
            source_axes = figure.axes
            
            # 创建3x3子图网格
            for i, source_ax in enumerate(source_axes):
                if i < 9:  # 最多显示9个子图
                    ax = self.canvas.figure.add_subplot(3, 3, i+1)
                    
                    # 复制图表内容
                    for patch in source_ax.patches:
                        ax.add_patch(patch)
                    
                    for line in source_ax.lines:
                        ax.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(), linewidth=line.get_linewidth())
                    
                    ax.set_title(source_ax.get_title(), fontsize=10)
                    ax.set_xlabel(source_ax.get_xlabel(), fontsize=8)
                    ax.set_ylabel(source_ax.get_ylabel(), fontsize=8)
                    ax.tick_params(labelsize=7)
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.on_analysis_error(f"显示分析结果失败: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.analyze_button.setEnabled(True)
    
    def on_analysis_error(self, error_msg):
        """分析错误处理"""
        QMessageBox.warning(self, "分析失败", f"综合分析失败:\n{error_msg}")
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
    
    def export_analysis(self):
        """导出分析结果"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "导出综合分析",
                f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG文件 (*.png);;PDF文件 (*.pdf);;Excel报告 (*.xlsx)"
            )
            
            if filename:
                if filename.endswith(('.png', '.pdf')):
                    self.canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                elif filename.endswith('.xlsx'):
                    # 导出Excel格式的分析报告
                    pass
                
                QMessageBox.information(self, "导出成功", f"分析结果已导出到:\n{filename}")
                
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"分析导出失败:\n{str(e)}")


class ChartsMainWidget(QWidget):
    """图表主组件（集成所有图表标签页）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 创建标题
        title_label = QLabel("📊 数据可视化分析中心")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                padding: 10px;
                background: #ecf0f1;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
            QTabBar::tab:hover {
                background: #74b9ff;
                color: white;
            }
        """)
        
        # 添加各种图表标签页
        self.frequency_widget = FrequencyChartWidget()
        self.trend_widget = TrendChartWidget()
        self.accuracy_widget = AccuracyChartWidget()
        self.analysis_widget = AnalysisChartWidget()
        
        self.tab_widget.addTab(self.frequency_widget, "📊 频率分析")
        self.tab_widget.addTab(self.trend_widget, "📈 走势分析")
        self.tab_widget.addTab(self.accuracy_widget, "🎯 准确率统计")
        self.tab_widget.addTab(self.analysis_widget, "🔍 综合分析")
        
        layout.addWidget(self.tab_widget)
        
        # 添加状态栏
        status_layout = QHBoxLayout()
        self.status_label = QLabel("就绪 - 选择标签页开始数据分析")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                padding: 5px;
                font-size: 12px;
            }
        """)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # 添加刷新所有按钮
        refresh_all_button = QPushButton("🔄 刷新所有图表")
        refresh_all_button.clicked.connect(self.refresh_all_charts)
        refresh_all_button.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2ecc71;
            }
        """)
        status_layout.addWidget(refresh_all_button)
        
        layout.addLayout(status_layout)
        self.setLayout(layout)
    
    def refresh_all_charts(self):
        """刷新所有图表"""
        try:
            current_index = self.tab_widget.currentIndex()
            
            # 更新当前标签页的图表
            if current_index == 0:  # 频率分析
                self.frequency_widget.update_chart()
            elif current_index == 1:  # 走势分析
                self.trend_widget.update_chart()
            elif current_index == 2:  # 准确率统计
                self.accuracy_widget.update_chart()
            elif current_index == 3:  # 综合分析
                self.analysis_widget.start_analysis()
            
            self.status_label.setText(f"正在刷新第{current_index+1}个标签页的图表...")
            
        except Exception as e:
            QMessageBox.warning(self, "刷新失败", f"刷新图表失败:\n{str(e)}")


# 使用示例和测试
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_widget = ChartsMainWidget()
    main_widget.setWindowTitle("AI彩票预测系统 - 数据可视化")
    main_widget.resize(1400, 900)
    main_widget.show()
    
    sys.exit(app.exec_())
