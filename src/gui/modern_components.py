"""
现代化界面组件
提供现代化的UI组件，包括卡片、按钮、输入框等
"""

import sys
import math
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QFrame,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QProgressBar, QSlider, QComboBox, QCheckBox, QRadioButton,
    QGroupBox, QTabWidget, QSplitter, QApplication, QGraphicsDropShadowEffect,
    QSizePolicy, QSpacerItem, QButtonGroup
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize,
    QPoint, pyqtSignal, QParallelAnimationGroup, QSequentialAnimationGroup,
    QAbstractAnimation, QVariantAnimation
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPainter, QBrush, QLinearGradient,
    QPen, QPixmap, QIcon, QFontMetrics, QPainterPath, QPolygonF
)


class ColorScheme:
    """颜色方案"""
    
    # 主色调
    PRIMARY = "#4CAF50"
    PRIMARY_LIGHT = "#81C784"
    PRIMARY_DARK = "#388E3C"
    
    # 次要色调
    SECONDARY = "#2196F3"
    SECONDARY_LIGHT = "#64B5F6"
    SECONDARY_DARK = "#1976D2"
    
    # 中性色
    BACKGROUND = "#FAFAFA"
    SURFACE = "#FFFFFF"
    ERROR = "#F44336"
    WARNING = "#FF9800"
    INFO = "#2196F3"
    SUCCESS = "#4CAF50"
    
    # 文本色
    TEXT_PRIMARY = "#212121"
    TEXT_SECONDARY = "#757575"
    TEXT_DISABLED = "#BDBDBD"
    
    # 边框色
    BORDER_LIGHT = "#E0E0E0"
    BORDER_MEDIUM = "#BDBDBD"
    BORDER_DARK = "#757575"


class AnimationType(Enum):
    """动画类型"""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_IN_LEFT = "slide_in_left"
    SLIDE_IN_RIGHT = "slide_in_right"
    SLIDE_IN_UP = "slide_in_up"
    SLIDE_IN_DOWN = "slide_in_down"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


class ModernCard(QFrame):
    """现代化卡片组件"""
    
    def __init__(self, title: str = "", subtitle: str = ""):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.setup_ui()
        self.setup_style()
    
    def setup_ui(self):
        """设置界面"""
        self.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # 标题区域
        if self.title or self.subtitle:
            header_layout = QVBoxLayout()
            
            if self.title:
                title_label = QLabel(self.title)
                title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
                title_label.setStyleSheet(f"color: {ColorScheme.TEXT_PRIMARY};")
                header_layout.addWidget(title_label)
            
            if self.subtitle:
                subtitle_label = QLabel(self.subtitle)
                subtitle_label.setFont(QFont("Segoe UI", 10))
                subtitle_label.setStyleSheet(f"color: {ColorScheme.TEXT_SECONDARY};")
                header_layout.addWidget(subtitle_label)
            
            layout.addLayout(header_layout)
        
        # 内容区域
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
    
    def setup_style(self):
        """设置样式"""
        self.setStyleSheet(f"""
            ModernCard {{
                background-color: {ColorScheme.SURFACE};
                border: 1px solid {ColorScheme.BORDER_LIGHT};
                border-radius: 8px;
            }}
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
    
    def add_widget(self, widget: QWidget):
        """添加控件到内容区域"""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """添加布局到内容区域"""
        self.content_layout.addLayout(layout)


class ModernButton(QPushButton):
    """现代化按钮"""
    
    def __init__(self, text: str, button_type: str = "default"):
        super().__init__(text)
        self.button_type = button_type
        self.setup_style()
        self.setup_animations()
    
    def setup_style(self):
        """设置样式"""
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        
        styles = {
            "primary": f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {ColorScheme.PRIMARY}, stop:1 {ColorScheme.PRIMARY_DARK});
                    border: none;
                    border-radius: 6px;
                    color: white;
                    font-weight: 500;
                    padding: 10px 20px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {ColorScheme.PRIMARY_LIGHT}, stop:1 {ColorScheme.PRIMARY});
                }}
                QPushButton:pressed {{
                    background: {ColorScheme.PRIMARY_DARK};
                }}
                QPushButton:disabled {{
                    background: {ColorScheme.TEXT_DISABLED};
                    color: white;
                }}
            """,
            "secondary": f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {ColorScheme.SECONDARY}, stop:1 {ColorScheme.SECONDARY_DARK});
                    border: none;
                    border-radius: 6px;
                    color: white;
                    font-weight: 500;
                    padding: 10px 20px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 {ColorScheme.SECONDARY_LIGHT}, stop:1 {ColorScheme.SECONDARY});
                }}
                QPushButton:pressed {{
                    background: {ColorScheme.SECONDARY_DARK};
                }}
            """,
            "outline": f"""
                QPushButton {{
                    background: transparent;
                    border: 2px solid {ColorScheme.PRIMARY};
                    border-radius: 6px;
                    color: {ColorScheme.PRIMARY};
                    font-weight: 500;
                    padding: 8px 18px;
                }}
                QPushButton:hover {{
                    background: {ColorScheme.PRIMARY};
                    color: white;
                }}
                QPushButton:pressed {{
                    background: {ColorScheme.PRIMARY_DARK};
                    border-color: {ColorScheme.PRIMARY_DARK};
                }}
            """,
            "text": f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    color: {ColorScheme.PRIMARY};
                    font-weight: 500;
                    padding: 10px 16px;
                }}
                QPushButton:hover {{
                    background: rgba(76, 175, 80, 0.1);
                    border-radius: 6px;
                }}
                QPushButton:pressed {{
                    background: rgba(76, 175, 80, 0.2);
                }}
            """,
            "default": f"""
                QPushButton {{
                    background: {ColorScheme.SURFACE};
                    border: 1px solid {ColorScheme.BORDER_MEDIUM};
                    border-radius: 6px;
                    color: {ColorScheme.TEXT_PRIMARY};
                    font-weight: 500;
                    padding: 10px 20px;
                }}
                QPushButton:hover {{
                    background: {ColorScheme.BACKGROUND};
                    border-color: {ColorScheme.BORDER_DARK};
                }}
                QPushButton:pressed {{
                    background: {ColorScheme.BORDER_LIGHT};
                }}
            """
        }
        
        self.setStyleSheet(styles.get(self.button_type, styles["default"]))
    
    def setup_animations(self):
        """设置动画"""
        self.press_animation = QPropertyAnimation(self, b"geometry")
        self.press_animation.setDuration(100)
        self.press_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        # 按下动画
        current_rect = self.geometry()
        pressed_rect = QRect(
            current_rect.x() + 1,
            current_rect.y() + 1,
            current_rect.width() - 2,
            current_rect.height() - 2
        )
        
        self.press_animation.setStartValue(current_rect)
        self.press_animation.setEndValue(pressed_rect)
        self.press_animation.start()
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        # 释放动画
        current_rect = self.geometry()
        released_rect = QRect(
            current_rect.x() - 1,
            current_rect.y() - 1,
            current_rect.width() + 2,
            current_rect.height() + 2
        )
        
        self.press_animation.setStartValue(current_rect)
        self.press_animation.setEndValue(released_rect)
        self.press_animation.start()
        
        super().mouseReleaseEvent(event)


class ModernLineEdit(QLineEdit):
    """现代化输入框"""
    
    def __init__(self, placeholder: str = ""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setup_style()
        self.setup_animations()
    
    def setup_style(self):
        """设置样式"""
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        
        self.setStyleSheet(f"""
            QLineEdit {{
                background: {ColorScheme.SURFACE};
                border: 2px solid {ColorScheme.BORDER_LIGHT};
                border-radius: 6px;
                color: {ColorScheme.TEXT_PRIMARY};
                padding: 8px 12px;
                selection-background-color: {ColorScheme.PRIMARY_LIGHT};
            }}
            QLineEdit:focus {{
                border-color: {ColorScheme.PRIMARY};
            }}
            QLineEdit:hover {{
                border-color: {ColorScheme.BORDER_MEDIUM};
            }}
            QLineEdit:disabled {{
                background: {ColorScheme.BACKGROUND};
                color: {ColorScheme.TEXT_DISABLED};
            }}
        """)
    
    def setup_animations(self):
        """设置动画"""
        self.focus_animation = QPropertyAnimation(self, b"styleSheet")
        self.focus_animation.setDuration(200)
    
    def focusInEvent(self, event):
        """获得焦点事件"""
        super().focusInEvent(event)
        # 可以添加获得焦点的动画效果
    
    def focusOutEvent(self, event):
        """失去焦点事件"""
        super().focusOutEvent(event)
        # 可以添加失去焦点的动画效果


class ModernProgressBar(QProgressBar):
    """现代化进度条"""
    
    def __init__(self):
        super().__init__()
        self.setup_style()
    
    def setup_style(self):
        """设置样式"""
        self.setMinimumHeight(8)
        self.setMaximumHeight(8)
        
        self.setStyleSheet(f"""
            QProgressBar {{
                background: {ColorScheme.BORDER_LIGHT};
                border: none;
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ColorScheme.PRIMARY}, stop:1 {ColorScheme.PRIMARY_LIGHT});
                border-radius: 4px;
            }}
        """)


class ModernSwitch(QCheckBox):
    """现代化开关"""
    
    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setup_style()
        self.setup_animations()
    
    def setup_style(self):
        """设置样式"""
        self.setStyleSheet(f"""
            QCheckBox {{
                font: 10pt "Segoe UI";
                color: {ColorScheme.TEXT_PRIMARY};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 44px;
                height: 24px;
                border-radius: 12px;
                background: {ColorScheme.BORDER_MEDIUM};
            }}
            QCheckBox::indicator:checked {{
                background: {ColorScheme.PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                background: {ColorScheme.BORDER_DARK};
            }}
            QCheckBox::indicator:checked:hover {{
                background: {ColorScheme.PRIMARY_LIGHT};
            }}
        """)
    
    def setup_animations(self):
        """设置动画"""
        self.toggle_animation = QPropertyAnimation(self, b"styleSheet")
        self.toggle_animation.setDuration(200)
        self.toggle_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def paintEvent(self, event):
        """自定义绘制"""
        super().paintEvent(event)
        
        # 这里可以添加自定义的开关绘制逻辑
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 获取指示器区域
        option = self.style().subElementRect(
            self.style().SE_CheckBoxIndicator, None, self
        )
        
        # 绘制开关背景
        if self.isChecked():
            painter.setBrush(QBrush(QColor(ColorScheme.PRIMARY)))
        else:
            painter.setBrush(QBrush(QColor(ColorScheme.BORDER_MEDIUM)))
        
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(option, 12, 12)
        
        # 绘制开关滑块
        painter.setBrush(QBrush(QColor(ColorScheme.SURFACE)))
        
        if self.isChecked():
            slider_x = option.right() - 20
        else:
            slider_x = option.left() + 2
        
        slider_rect = QRect(slider_x, option.top() + 2, 20, 20)
        painter.drawEllipse(slider_rect)


class ModernTabWidget(QTabWidget):
    """现代化标签页"""
    
    def __init__(self):
        super().__init__()
        self.setup_style()
    
    def setup_style(self):
        """设置样式"""
        self.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {ColorScheme.BORDER_LIGHT};
                background: {ColorScheme.SURFACE};
                border-radius: 8px;
                margin-top: -1px;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {ColorScheme.TEXT_SECONDARY};
                padding: 12px 20px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font: 10pt "Segoe UI";
                font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background: {ColorScheme.SURFACE};
                color: {ColorScheme.PRIMARY};
                border-bottom: 2px solid {ColorScheme.PRIMARY};
            }}
            QTabBar::tab:hover:!selected {{
                background: {ColorScheme.BACKGROUND};
                color: {ColorScheme.TEXT_PRIMARY};
            }}
        """)


class AnimationManager:
    """动画管理器"""
    
    @staticmethod
    def fade_in(widget: QWidget, duration: int = 300):
        """淡入动画"""
        widget.setWindowOpacity(0)
        widget.show()
        
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
        
        return animation
    
    @staticmethod
    def fade_out(widget: QWidget, duration: int = 300):
        """淡出动画"""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(1)
        animation.setEndValue(0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.finished.connect(widget.hide)
        animation.start()
        
        return animation
    
    @staticmethod
    def slide_in_from_left(widget: QWidget, duration: int = 400):
        """从左侧滑入"""
        start_pos = QPoint(-widget.width(), widget.y())
        end_pos = widget.pos()
        
        widget.move(start_pos)
        widget.show()
        
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(duration)
        animation.setStartValue(start_pos)
        animation.setEndValue(end_pos)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
        
        return animation
    
    @staticmethod
    def slide_in_from_right(widget: QWidget, duration: int = 400):
        """从右侧滑入"""
        parent_width = widget.parent().width() if widget.parent() else 800
        start_pos = QPoint(parent_width, widget.y())
        end_pos = widget.pos()
        
        widget.move(start_pos)
        widget.show()
        
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(duration)
        animation.setStartValue(start_pos)
        animation.setEndValue(end_pos)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
        
        return animation
    
    @staticmethod
    def scale_in(widget: QWidget, duration: int = 300):
        """缩放进入"""
        original_size = widget.size()
        widget.resize(0, 0)
        widget.show()
        
        animation = QPropertyAnimation(widget, b"size")
        animation.setDuration(duration)
        animation.setStartValue(QSize(0, 0))
        animation.setEndValue(original_size)
        animation.setEasingCurve(QEasingCurve.OutBack)
        animation.start()
        
        return animation
    
    @staticmethod
    def bounce_in(widget: QWidget, duration: int = 600):
        """弹跳进入"""
        original_size = widget.size()
        
        # 创建序列动画
        sequence = QSequentialAnimationGroup()
        
        # 第一阶段：快速放大
        scale_up = QPropertyAnimation(widget, b"size")
        scale_up.setDuration(duration // 3)
        scale_up.setStartValue(QSize(0, 0))
        scale_up.setEndValue(QSize(
            int(original_size.width() * 1.1),
            int(original_size.height() * 1.1)
        ))
        scale_up.setEasingCurve(QEasingCurve.OutCubic)
        
        # 第二阶段：回弹到正常大小
        scale_back = QPropertyAnimation(widget, b"size")
        scale_back.setDuration(duration // 3)
        scale_back.setStartValue(QSize(
            int(original_size.width() * 1.1),
            int(original_size.height() * 1.1)
        ))
        scale_back.setEndValue(QSize(
            int(original_size.width() * 0.95),
            int(original_size.height() * 0.95)
        ))
        scale_back.setEasingCurve(QEasingCurve.OutCubic)
        
        # 第三阶段：稳定到最终大小
        scale_final = QPropertyAnimation(widget, b"size")
        scale_final.setDuration(duration // 3)
        scale_final.setStartValue(QSize(
            int(original_size.width() * 0.95),
            int(original_size.height() * 0.95)
        ))
        scale_final.setEndValue(original_size)
        scale_final.setEasingCurve(QEasingCurve.OutCubic)
        
        sequence.addAnimation(scale_up)
        sequence.addAnimation(scale_back)
        sequence.addAnimation(scale_final)
        
        widget.resize(0, 0)
        widget.show()
        sequence.start()
        
        return sequence


class ModernNotification(QWidget):
    """现代化通知组件"""
    
    def __init__(self, message: str, notification_type: str = "info", parent=None):
        super().__init__(parent)
        self.message = message
        self.notification_type = notification_type
        self.setup_ui()
        self.setup_style()
        self.setup_auto_hide()
    
    def setup_ui(self):
        """设置界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        
        # 图标（可选）
        # icon_label = QLabel()
        # layout.addWidget(icon_label)
        
        # 消息文本
        message_label = QLabel(self.message)
        message_label.setFont(QFont("Segoe UI", 10))
        message_label.setWordWrap(True)
        layout.addWidget(message_label)
        
        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.hide_notification)
        layout.addWidget(close_btn)
    
    def setup_style(self):
        """设置样式"""
        colors = {
            "info": ColorScheme.INFO,
            "success": ColorScheme.SUCCESS,
            "warning": ColorScheme.WARNING,
            "error": ColorScheme.ERROR
        }
        
        bg_color = colors.get(self.notification_type, ColorScheme.INFO)
        
        self.setStyleSheet(f"""
            ModernNotification {{
                background: {bg_color};
                color: white;
                border-radius: 6px;
                border: none;
            }}
            QPushButton {{
                background: transparent;
                color: white;
                border: none;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 12px;
            }}
        """)
        
        # 添加阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
    
    def setup_auto_hide(self):
        """设置自动隐藏"""
        self.hide_timer = QTimer()
        self.hide_timer.timeout.connect(self.hide_notification)
        self.hide_timer.start(5000)  # 5秒后自动隐藏
    
    def show_notification(self):
        """显示通知"""
        self.show()
        AnimationManager.slide_in_from_right(self, 400)
    
    def hide_notification(self):
        """隐藏通知"""
        self.hide_timer.stop()
        animation = AnimationManager.fade_out(self, 300)
        animation.finished.connect(self.deleteLater)


def apply_modern_theme(app: QApplication):
    """应用现代化主题"""
    app.setStyleSheet(f"""
        * {{
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }}
        
        QWidget {{
            background-color: {ColorScheme.BACKGROUND};
            color: {ColorScheme.TEXT_PRIMARY};
        }}
        
        QMainWindow {{
            background-color: {ColorScheme.BACKGROUND};
        }}
        
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        
        QScrollBar:vertical {{
            background: {ColorScheme.BACKGROUND};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {ColorScheme.BORDER_MEDIUM};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {ColorScheme.BORDER_DARK};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QToolTip {{
            background-color: {ColorScheme.TEXT_PRIMARY};
            color: white;
            border: none;
            padding: 6px;
            border-radius: 4px;
            font: 9pt "Segoe UI";
        }}
    """)


def main():
    """测试主函数"""
    app = QApplication(sys.argv)
    apply_modern_theme(app)
    
    # 创建测试窗口
    window = QWidget()
    window.setWindowTitle("现代化组件测试")
    window.resize(600, 400)
    
    layout = QVBoxLayout(window)
    
    # 卡片测试
    card = ModernCard("测试卡片", "这是一个现代化的卡片组件")
    
    # 按钮测试
    btn_layout = QHBoxLayout()
    btn_layout.addWidget(ModernButton("主要按钮", "primary"))
    btn_layout.addWidget(ModernButton("次要按钮", "secondary"))
    btn_layout.addWidget(ModernButton("轮廓按钮", "outline"))
    btn_layout.addWidget(ModernButton("文本按钮", "text"))
    card.add_layout(btn_layout)
    
    # 输入框测试
    line_edit = ModernLineEdit("请输入内容...")
    card.add_widget(line_edit)
    
    # 进度条测试
    progress = ModernProgressBar()
    progress.setValue(60)
    card.add_widget(progress)
    
    # 开关测试
    switch = ModernSwitch("启用功能")
    card.add_widget(switch)
    
    layout.addWidget(card)
    
    # 标签页测试
    tab_widget = ModernTabWidget()
    tab_widget.addTab(QLabel("标签页1内容"), "标签页1")
    tab_widget.addTab(QLabel("标签页2内容"), "标签页2")
    tab_widget.addTab(QLabel("标签页3内容"), "标签页3")
    layout.addWidget(tab_widget)
    
    window.show()
    
    # 显示通知测试
    def show_notification():
        notification = ModernNotification("这是一个测试通知", "success", window)
        notification.move(window.width() - 300, 50)
        notification.show_notification()
    
    QTimer.singleShot(2000, show_notification)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
