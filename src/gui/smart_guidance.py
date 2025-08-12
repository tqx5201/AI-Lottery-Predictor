"""
智能推荐引导系统
提供智能化的用户引导和推荐功能
"""

import sys
import json
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QFrame, QScrollArea, QTextEdit, QProgressBar, QApplication,
    QGraphicsDropShadowEffect, QSizePolicy, QSpacerItem, QButtonGroup
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize,
    QPoint, pyqtSignal, QThread, QObject, QEvent
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPainter, QBrush, QLinearGradient,
    QPen, QPixmap, QIcon, QFontMetrics, QPainterPath
)


class GuidanceType(Enum):
    """引导类型"""
    WELCOME = "welcome"           # 欢迎引导
    FEATURE_INTRO = "feature"     # 功能介绍
    QUICK_START = "quick_start"   # 快速开始
    TIPS = "tips"                 # 使用技巧
    RECOMMENDATION = "recommendation"  # 智能推荐
    TUTORIAL = "tutorial"         # 教程指导
    HELP = "help"                # 帮助信息


class GuidancePriority(Enum):
    """引导优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GuidanceItem:
    """引导项"""
    id: str
    title: str
    content: str
    guidance_type: GuidanceType
    priority: GuidancePriority
    target_widget: Optional[str] = None  # 目标控件名称
    action_text: Optional[str] = None    # 操作按钮文本
    action_callback: Optional[Callable] = None  # 操作回调
    conditions: Optional[Dict[str, Any]] = None  # 显示条件
    icon: Optional[str] = None
    auto_dismiss: bool = False
    dismiss_delay: int = 5000  # 自动消失延迟(毫秒)
    
    def should_show(self, context: Dict[str, Any]) -> bool:
        """检查是否应该显示"""
        if not self.conditions:
            return True
        
        for key, expected_value in self.conditions.items():
            if key not in context or context[key] != expected_value:
                return False
        
        return True


class GuidanceCard(QFrame):
    """引导卡片"""
    
    actionClicked = pyqtSignal(str)  # 发送引导项ID
    dismissed = pyqtSignal(str)      # 发送引导项ID
    
    def __init__(self, guidance_item: GuidanceItem):
        super().__init__()
        self.guidance_item = guidance_item
        self.setup_ui()
        self.setup_style()
        self.setup_auto_dismiss()
    
    def setup_ui(self):
        """设置界面"""
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 标题栏
        header_layout = QHBoxLayout()
        
        # 标题
        title_label = QLabel(self.guidance_item.title)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setFont(QFont("Arial", 14, QFont.Bold))
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #7f8c8d;
                border-radius: 12px;
            }
            QPushButton:hover {
                background: #ecf0f1;
                color: #2c3e50;
            }
        """)
        close_btn.clicked.connect(self.dismiss)
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # 内容
        content_label = QLabel(self.guidance_item.content)
        content_label.setFont(QFont("Segoe UI", 10))
        content_label.setStyleSheet("color: #34495e; line-height: 1.4;")
        content_label.setWordWrap(True)
        layout.addWidget(content_label)
        
        # 操作按钮
        if self.guidance_item.action_text:
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            action_btn = QPushButton(self.guidance_item.action_text)
            action_btn.setFont(QFont("Segoe UI", 9, QFont.Bold))
            action_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3498db, stop:1 #2980b9);
                    border: none;
                    border-radius: 6px;
                    color: white;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5dade2, stop:1 #3498db);
                }
                QPushButton:pressed {
                    background: #2471a3;
                }
            """)
            action_btn.clicked.connect(self.on_action_clicked)
            button_layout.addWidget(action_btn)
            
            layout.addLayout(button_layout)
    
    def setup_style(self):
        """设置样式"""
        # 根据类型设置不同的边框颜色
        type_colors = {
            GuidanceType.WELCOME: "#e74c3c",
            GuidanceType.FEATURE_INTRO: "#3498db",
            GuidanceType.QUICK_START: "#2ecc71",
            GuidanceType.TIPS: "#f39c12",
            GuidanceType.RECOMMENDATION: "#9b59b6",
            GuidanceType.TUTORIAL: "#1abc9c",
            GuidanceType.HELP: "#95a5a6"
        }
        
        border_color = type_colors.get(self.guidance_item.guidance_type, "#bdc3c7")
        
        self.setStyleSheet(f"""
            GuidanceCard {{
                background-color: white;
                border-left: 4px solid {border_color};
                border-radius: 8px;
            }}
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
    
    def setup_auto_dismiss(self):
        """设置自动消失"""
        if self.guidance_item.auto_dismiss:
            self.dismiss_timer = QTimer()
            self.dismiss_timer.timeout.connect(self.dismiss)
            self.dismiss_timer.start(self.guidance_item.dismiss_delay)
    
    def on_action_clicked(self):
        """操作按钮点击"""
        if self.guidance_item.action_callback:
            self.guidance_item.action_callback()
        
        self.actionClicked.emit(self.guidance_item.id)
        self.dismiss()
    
    def dismiss(self):
        """关闭引导"""
        if hasattr(self, 'dismiss_timer'):
            self.dismiss_timer.stop()
        
        self.dismissed.emit(self.guidance_item.id)
        
        # 淡出动画
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.deleteLater)
        self.fade_animation.start()


class SmartRecommendationWidget(QWidget):
    """智能推荐控件"""
    
    def __init__(self):
        super().__init__()
        self.recommendations = []
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # 标题
        title = QLabel("🤖 智能推荐")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # 推荐列表容器
        self.recommendations_container = QVBoxLayout()
        layout.addLayout(self.recommendations_container)
        
        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新推荐")
        refresh_btn.setFont(QFont("Segoe UI", 10))
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 8px 16px;
                color: #2c3e50;
            }
            QPushButton:hover {
                background: #d5dbdb;
                border-color: #95a5a6;
            }
        """)
        refresh_btn.clicked.connect(self.generate_recommendations)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        
        # 生成初始推荐
        self.generate_recommendations()
    
    def generate_recommendations(self):
        """生成智能推荐"""
        # 清除现有推荐
        self.clear_recommendations()
        
        # 模拟智能推荐生成
        sample_recommendations = [
            {
                "title": "📊 尝试高级分析功能",
                "content": "基于您的使用习惯，建议您尝试我们的高级数据分析功能，它可以帮助您发现更多数据规律。",
                "action": "立即体验",
                "priority": "high"
            },
            {
                "title": "🎯 优化预测模型",
                "content": "检测到您经常使用预测功能，建议启用GPU加速以获得更快的预测速度。",
                "action": "启用GPU",
                "priority": "medium"
            },
            {
                "title": "💡 使用技巧分享",
                "content": "您知道吗？使用Ctrl+R可以快速刷新数据，使用Ctrl+S可以快速保存当前分析结果。",
                "action": "了解更多",
                "priority": "low"
            },
            {
                "title": "🔧 个性化设置",
                "content": "根据您的使用频率，建议您自定义界面布局以提高工作效率。",
                "action": "去设置",
                "priority": "medium"
            },
            {
                "title": "📈 数据可视化升级",
                "content": "新版本增加了更多图表类型，建议您尝试使用热力图来展示数据关联性。",
                "action": "查看新功能",
                "priority": "high"
            }
        ]
        
        # 随机选择3-4个推荐
        selected = random.sample(sample_recommendations, random.randint(3, 4))
        
        for rec in selected:
            self.add_recommendation_item(rec)
    
    def add_recommendation_item(self, recommendation: Dict[str, str]):
        """添加推荐项"""
        item_widget = QFrame()
        item_widget.setFrameStyle(QFrame.StyledPanel)
        
        # 设置样式
        priority_colors = {
            "high": "#e74c3c",
            "medium": "#f39c12", 
            "low": "#95a5a6"
        }
        
        border_color = priority_colors.get(recommendation.get("priority", "low"), "#95a5a6")
        
        item_widget.setStyleSheet(f"""
            QFrame {{
                background: white;
                border-left: 3px solid {border_color};
                border-radius: 6px;
                margin: 2px 0px;
            }}
        """)
        
        layout = QVBoxLayout(item_widget)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        
        # 标题
        title_label = QLabel(recommendation["title"])
        title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title_label)
        
        # 内容
        content_label = QLabel(recommendation["content"])
        content_label.setFont(QFont("Segoe UI", 9))
        content_label.setStyleSheet("color: #7f8c8d; line-height: 1.3;")
        content_label.setWordWrap(True)
        layout.addWidget(content_label)
        
        # 操作按钮
        if recommendation.get("action"):
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            action_btn = QPushButton(recommendation["action"])
            action_btn.setFont(QFont("Segoe UI", 8))
            action_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {border_color};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 12px;
                }}
                QPushButton:hover {{
                    background: {self._darken_color(border_color)};
                }}
            """)
            button_layout.addWidget(action_btn)
            layout.addLayout(button_layout)
        
        self.recommendations_container.addWidget(item_widget)
        self.recommendations.append(item_widget)
    
    def _darken_color(self, color: str) -> str:
        """使颜色变暗"""
        color_map = {
            "#e74c3c": "#c0392b",
            "#f39c12": "#d68910",
            "#95a5a6": "#7f8c8d"
        }
        return color_map.get(color, color)
    
    def clear_recommendations(self):
        """清除所有推荐"""
        for widget in self.recommendations:
            widget.deleteLater()
        self.recommendations.clear()


class GuidancePanel(QScrollArea):
    """引导面板"""
    
    guidanceActionTriggered = pyqtSignal(str, str)  # guidance_id, action
    
    def __init__(self):
        super().__init__()
        self.guidance_items = []
        self.active_cards = {}
        self.user_context = {}
        self.setup_ui()
        self.load_default_guidance()
    
    def setup_ui(self):
        """设置界面"""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 主容器
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(12)
        
        self.setWidget(self.container)
        
        # 设置样式
        self.setStyleSheet("""
            QScrollArea {
                background: #f8f9fa;
                border: none;
            }
            QScrollBar:vertical {
                background: #ecf0f1;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
        """)
    
    def load_default_guidance(self):
        """加载默认引导"""
        default_guidance = [
            GuidanceItem(
                id="welcome",
                title="🎉 欢迎使用AI彩票预测系统",
                content="欢迎来到全新升级的AI彩票预测分析系统！我们为您准备了强大的分析工具和智能预测功能。",
                guidance_type=GuidanceType.WELCOME,
                priority=GuidancePriority.HIGH,
                action_text="开始探索",
                auto_dismiss=False
            ),
            GuidanceItem(
                id="quick_start",
                title="🚀 快速开始指南",
                content="点击左侧菜单开始数据分析，或使用顶部的快速预测功能立即获得智能推荐。",
                guidance_type=GuidanceType.QUICK_START,
                priority=GuidancePriority.HIGH,
                action_text="查看教程"
            ),
            GuidanceItem(
                id="new_features",
                title="✨ 新功能介绍",
                content="本次更新增加了GPU加速、智能缓存、响应式界面等功能，大幅提升了系统性能和用户体验。",
                guidance_type=GuidanceType.FEATURE_INTRO,
                priority=GuidancePriority.MEDIUM,
                action_text="了解详情"
            ),
            GuidanceItem(
                id="performance_tip",
                title="⚡ 性能优化建议",
                content="建议启用GPU加速和智能缓存功能以获得最佳性能体验。您可以在设置中进行配置。",
                guidance_type=GuidanceType.TIPS,
                priority=GuidancePriority.MEDIUM,
                action_text="去设置",
                conditions={"gpu_available": True}
            ),
            GuidanceItem(
                id="data_backup",
                title="💾 数据备份提醒",
                content="为了保护您的重要数据，建议定期备份分析结果和配置信息。",
                guidance_type=GuidanceType.TIPS,
                priority=GuidancePriority.LOW,
                action_text="立即备份",
                auto_dismiss=True,
                dismiss_delay=8000
            )
        ]
        
        for guidance in default_guidance:
            self.add_guidance_item(guidance)
        
        # 更新显示
        self.update_guidance_display()
    
    def add_guidance_item(self, guidance_item: GuidanceItem):
        """添加引导项"""
        self.guidance_items.append(guidance_item)
    
    def remove_guidance_item(self, guidance_id: str):
        """移除引导项"""
        self.guidance_items = [
            item for item in self.guidance_items 
            if item.id != guidance_id
        ]
        
        if guidance_id in self.active_cards:
            self.active_cards[guidance_id].deleteLater()
            del self.active_cards[guidance_id]
    
    def update_user_context(self, context: Dict[str, Any]):
        """更新用户上下文"""
        self.user_context.update(context)
        self.update_guidance_display()
    
    def update_guidance_display(self):
        """更新引导显示"""
        # 按优先级排序
        sorted_items = sorted(
            self.guidance_items,
            key=lambda x: (x.priority.value, x.guidance_type.value),
            reverse=True
        )
        
        # 显示符合条件的引导
        for guidance_item in sorted_items:
            if guidance_item.should_show(self.user_context):
                if guidance_item.id not in self.active_cards:
                    self.show_guidance_card(guidance_item)
    
    def show_guidance_card(self, guidance_item: GuidanceItem):
        """显示引导卡片"""
        card = GuidanceCard(guidance_item)
        card.actionClicked.connect(self.on_guidance_action)
        card.dismissed.connect(self.on_guidance_dismissed)
        
        self.container_layout.addWidget(card)
        self.active_cards[guidance_item.id] = card
        
        # 入场动画
        card.setWindowOpacity(0)
        card.show()
        
        fade_in = QPropertyAnimation(card, b"windowOpacity")
        fade_in.setDuration(400)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        fade_in.setEasingCurve(QEasingCurve.OutCubic)
        fade_in.start()
    
    def on_guidance_action(self, guidance_id: str):
        """引导操作处理"""
        self.guidanceActionTriggered.emit(guidance_id, "action")
        self.remove_guidance_item(guidance_id)
    
    def on_guidance_dismissed(self, guidance_id: str):
        """引导关闭处理"""
        self.guidanceActionTriggered.emit(guidance_id, "dismiss")
        self.remove_guidance_item(guidance_id)


class SmartGuidanceSystem(QObject):
    """智能引导系统"""
    
    def __init__(self):
        super().__init__()
        self.guidance_panel = GuidancePanel()
        self.recommendation_widget = SmartRecommendationWidget()
        self.user_behavior = {}
        self.setup_connections()
    
    def setup_connections(self):
        """设置连接"""
        self.guidance_panel.guidanceActionTriggered.connect(self.handle_guidance_action)
    
    def get_guidance_panel(self) -> GuidancePanel:
        """获取引导面板"""
        return self.guidance_panel
    
    def get_recommendation_widget(self) -> SmartRecommendationWidget:
        """获取推荐控件"""
        return self.recommendation_widget
    
    def track_user_behavior(self, action: str, context: Dict[str, Any] = None):
        """跟踪用户行为"""
        timestamp = time.time()
        
        if action not in self.user_behavior:
            self.user_behavior[action] = []
        
        self.user_behavior[action].append({
            'timestamp': timestamp,
            'context': context or {}
        })
        
        # 基于行为生成新的引导
        self.generate_contextual_guidance(action, context or {})
    
    def generate_contextual_guidance(self, action: str, context: Dict[str, Any]):
        """生成上下文相关的引导"""
        # 根据用户行为生成智能引导
        if action == "model_training_completed":
            guidance = GuidanceItem(
                id=f"training_success_{int(time.time())}",
                title="🎉 模型训练完成",
                content=f"您的{context.get('model_type', '模型')}训练已完成！准确率：{context.get('accuracy', 'N/A')}%",
                guidance_type=GuidanceType.RECOMMENDATION,
                priority=GuidancePriority.HIGH,
                action_text="查看结果",
                auto_dismiss=True,
                dismiss_delay=6000
            )
            self.guidance_panel.add_guidance_item(guidance)
        
        elif action == "prediction_low_confidence":
            guidance = GuidanceItem(
                id=f"low_confidence_{int(time.time())}",
                title="⚠️ 预测置信度较低",
                content="当前预测的置信度较低，建议您增加训练数据或尝试其他模型以获得更准确的预测。",
                guidance_type=GuidanceType.TIPS,
                priority=GuidancePriority.MEDIUM,
                action_text="优化模型"
            )
            self.guidance_panel.add_guidance_item(guidance)
        
        elif action == "first_time_user":
            guidance = GuidanceItem(
                id="first_time_tutorial",
                title="👋 新用户引导",
                content="看起来您是第一次使用我们的系统！让我们为您介绍主要功能和使用方法。",
                guidance_type=GuidanceType.TUTORIAL,
                priority=GuidancePriority.CRITICAL,
                action_text="开始教程"
            )
            self.guidance_panel.add_guidance_item(guidance)
        
        # 更新显示
        self.guidance_panel.update_guidance_display()
    
    def handle_guidance_action(self, guidance_id: str, action: str):
        """处理引导操作"""
        print(f"引导操作: {guidance_id} - {action}")
        
        # 根据不同的引导ID执行相应的操作
        if guidance_id == "welcome":
            self.show_welcome_tutorial()
        elif guidance_id == "quick_start":
            self.show_quick_start_guide()
        elif guidance_id == "performance_tip":
            self.open_settings_page()
        # 可以添加更多操作处理
    
    def show_welcome_tutorial(self):
        """显示欢迎教程"""
        print("显示欢迎教程")
        # 实现教程逻辑
    
    def show_quick_start_guide(self):
        """显示快速开始指南"""
        print("显示快速开始指南")
        # 实现指南逻辑
    
    def open_settings_page(self):
        """打开设置页面"""
        print("打开设置页面")
        # 实现设置页面打开逻辑
    
    def update_context(self, context: Dict[str, Any]):
        """更新上下文信息"""
        self.guidance_panel.update_user_context(context)


def main():
    """测试主函数"""
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = QWidget()
    window.setWindowTitle("智能引导系统测试")
    window.resize(800, 600)
    
    layout = QHBoxLayout(window)
    
    # 创建智能引导系统
    guidance_system = SmartGuidanceSystem()
    
    # 添加引导面板
    layout.addWidget(guidance_system.get_guidance_panel(), 2)
    
    # 添加推荐控件
    layout.addWidget(guidance_system.get_recommendation_widget(), 1)
    
    window.show()
    
    # 模拟用户行为
    def simulate_behavior():
        guidance_system.track_user_behavior("model_training_completed", {
            "model_type": "XGBoost",
            "accuracy": 85.6
        })
    
    def simulate_low_confidence():
        guidance_system.track_user_behavior("prediction_low_confidence", {
            "confidence": 0.3
        })
    
    # 延迟模拟
    QTimer.singleShot(3000, simulate_behavior)
    QTimer.singleShot(6000, simulate_low_confidence)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
