"""
响应式布局和动画效果系统
提供自适应布局和流畅的动画效果
"""

import sys
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QStackedLayout,
    QScrollArea, QSplitter, QFrame, QLabel, QApplication, QSizePolicy,
    QSpacerItem, QLayout, QLayoutItem
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize,
    QPoint, pyqtSignal, QParallelAnimationGroup, QSequentialAnimationGroup,
    QVariantAnimation, QAbstractAnimation, QObject, QEvent
)
from PyQt5.QtGui import QResizeEvent, QShowEvent, QHideEvent


class BreakPoint(Enum):
    """响应式断点"""
    XS = 480   # 超小屏幕
    SM = 768   # 小屏幕
    MD = 1024  # 中等屏幕
    LG = 1280  # 大屏幕
    XL = 1920  # 超大屏幕


class AnimationDirection(Enum):
    """动画方向"""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"
    CENTER_OUT = "center_out"
    CENTER_IN = "center_in"


@dataclass
class ResponsiveConfig:
    """响应式配置"""
    xs_columns: int = 1
    sm_columns: int = 2
    md_columns: int = 3
    lg_columns: int = 4
    xl_columns: int = 6
    
    xs_spacing: int = 8
    sm_spacing: int = 12
    md_spacing: int = 16
    lg_spacing: int = 20
    xl_spacing: int = 24
    
    animation_duration: int = 300
    animation_easing: QEasingCurve.Type = QEasingCurve.OutCubic


class ResponsiveGridLayout(QGridLayout):
    """响应式网格布局"""
    
    def __init__(self, config: ResponsiveConfig = None):
        super().__init__()
        self.config = config or ResponsiveConfig()
        self.current_breakpoint = BreakPoint.MD
        self.widgets_data = []  # 存储控件和其配置
        self.animation_group = QParallelAnimationGroup()
        
        self.setContentsMargins(0, 0, 0, 0)
    
    def add_responsive_widget(self, widget: QWidget, 
                            xs_span: int = 1, sm_span: int = 1, 
                            md_span: int = 1, lg_span: int = 1, xl_span: int = 1):
        """添加响应式控件"""
        widget_data = {
            'widget': widget,
            'spans': {
                BreakPoint.XS: xs_span,
                BreakPoint.SM: sm_span,
                BreakPoint.MD: md_span,
                BreakPoint.LG: lg_span,
                BreakPoint.XL: xl_span
            }
        }
        self.widgets_data.append(widget_data)
        self.update_layout()
    
    def remove_responsive_widget(self, widget: QWidget):
        """移除响应式控件"""
        self.widgets_data = [
            data for data in self.widgets_data 
            if data['widget'] != widget
        ]
        self.removeWidget(widget)
        self.update_layout()
    
    def get_current_breakpoint(self, width: int) -> BreakPoint:
        """获取当前断点"""
        if width < BreakPoint.XS.value:
            return BreakPoint.XS
        elif width < BreakPoint.SM.value:
            return BreakPoint.SM
        elif width < BreakPoint.MD.value:
            return BreakPoint.MD
        elif width < BreakPoint.LG.value:
            return BreakPoint.LG
        else:
            return BreakPoint.XL
    
    def get_columns_for_breakpoint(self, breakpoint: BreakPoint) -> int:
        """获取断点对应的列数"""
        columns_map = {
            BreakPoint.XS: self.config.xs_columns,
            BreakPoint.SM: self.config.sm_columns,
            BreakPoint.MD: self.config.md_columns,
            BreakPoint.LG: self.config.lg_columns,
            BreakPoint.XL: self.config.xl_columns
        }
        return columns_map.get(breakpoint, self.config.md_columns)
    
    def get_spacing_for_breakpoint(self, breakpoint: BreakPoint) -> int:
        """获取断点对应的间距"""
        spacing_map = {
            BreakPoint.XS: self.config.xs_spacing,
            BreakPoint.SM: self.config.sm_spacing,
            BreakPoint.MD: self.config.md_spacing,
            BreakPoint.LG: self.config.lg_spacing,
            BreakPoint.XL: self.config.xl_spacing
        }
        return spacing_map.get(breakpoint, self.config.md_spacing)
    
    def update_layout(self, width: int = None):
        """更新布局"""
        if not width and self.parent():
            width = self.parent().width()
        elif not width:
            width = 1024  # 默认宽度
        
        new_breakpoint = self.get_current_breakpoint(width)
        
        # 如果断点没有改变，不需要重新布局
        if new_breakpoint == self.current_breakpoint and self.count() > 0:
            return
        
        self.current_breakpoint = new_breakpoint
        columns = self.get_columns_for_breakpoint(new_breakpoint)
        spacing = self.get_spacing_for_breakpoint(new_breakpoint)
        
        # 更新间距
        self.setSpacing(spacing)
        
        # 清除现有布局
        while self.count():
            self.takeAt(0)
        
        # 重新排列控件
        row, col = 0, 0
        
        for widget_data in self.widgets_data:
            widget = widget_data['widget']
            span = widget_data['spans'].get(new_breakpoint, 1)
            
            # 确保span不超过总列数
            span = min(span, columns)
            
            # 如果当前行剩余空间不足，换行
            if col + span > columns:
                row += 1
                col = 0
            
            # 添加控件到布局
            self.addWidget(widget, row, col, 1, span)
            
            col += span
            
            # 如果填满一行，换行
            if col >= columns:
                row += 1
                col = 0


class ResponsiveWidget(QWidget):
    """响应式控件容器"""
    
    breakpointChanged = pyqtSignal(BreakPoint)
    
    def __init__(self, config: ResponsiveConfig = None):
        super().__init__()
        self.config = config or ResponsiveConfig()
        self.current_breakpoint = BreakPoint.MD
        self.responsive_layout = ResponsiveGridLayout(self.config)
        self.setLayout(self.responsive_layout)
        
        # 安装事件过滤器
        self.installEventFilter(self)
    
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """事件过滤器"""
        if obj == self and event.type() == QEvent.Resize:
            self.handle_resize(event)
        return super().eventFilter(obj, event)
    
    def handle_resize(self, event: QResizeEvent):
        """处理窗口大小变化"""
        new_width = event.size().width()
        new_breakpoint = self.responsive_layout.get_current_breakpoint(new_width)
        
        if new_breakpoint != self.current_breakpoint:
            self.current_breakpoint = new_breakpoint
            self.breakpointChanged.emit(new_breakpoint)
            
            # 更新布局
            self.responsive_layout.update_layout(new_width)
    
    def add_widget(self, widget: QWidget, **kwargs):
        """添加控件"""
        self.responsive_layout.add_responsive_widget(widget, **kwargs)
    
    def remove_widget(self, widget: QWidget):
        """移除控件"""
        self.responsive_layout.remove_responsive_widget(widget)


class AnimatedStackedWidget(QStackedLayout):
    """带动画的堆叠布局"""
    
    def __init__(self):
        super().__init__()
        self.current_animation = None
        self.animation_duration = 300
        self.animation_direction = AnimationDirection.LEFT_TO_RIGHT
    
    def set_animation_direction(self, direction: AnimationDirection):
        """设置动画方向"""
        self.animation_direction = direction
    
    def set_animation_duration(self, duration: int):
        """设置动画持续时间"""
        self.animation_duration = duration
    
    def animated_set_current_index(self, index: int):
        """带动画的切换页面"""
        if index == self.currentIndex():
            return
        
        if self.current_animation and self.current_animation.state() == QAbstractAnimation.Running:
            self.current_animation.stop()
        
        old_widget = self.currentWidget()
        new_widget = self.widget(index)
        
        if not old_widget or not new_widget:
            self.setCurrentIndex(index)
            return
        
        # 创建动画
        self.current_animation = self._create_transition_animation(old_widget, new_widget)
        self.current_animation.finished.connect(lambda: self.setCurrentIndex(index))
        self.current_animation.start()
    
    def _create_transition_animation(self, old_widget: QWidget, 
                                   new_widget: QWidget) -> QAbstractAnimation:
        """创建过渡动画"""
        # 获取容器大小
        container_rect = self.geometry()
        
        # 设置新控件的初始位置
        if self.animation_direction == AnimationDirection.LEFT_TO_RIGHT:
            new_start_pos = QPoint(container_rect.width(), 0)
            new_end_pos = QPoint(0, 0)
            old_end_pos = QPoint(-container_rect.width(), 0)
        elif self.animation_direction == AnimationDirection.RIGHT_TO_LEFT:
            new_start_pos = QPoint(-container_rect.width(), 0)
            new_end_pos = QPoint(0, 0)
            old_end_pos = QPoint(container_rect.width(), 0)
        elif self.animation_direction == AnimationDirection.TOP_TO_BOTTOM:
            new_start_pos = QPoint(0, -container_rect.height())
            new_end_pos = QPoint(0, 0)
            old_end_pos = QPoint(0, container_rect.height())
        elif self.animation_direction == AnimationDirection.BOTTOM_TO_TOP:
            new_start_pos = QPoint(0, container_rect.height())
            new_end_pos = QPoint(0, 0)
            old_end_pos = QPoint(0, -container_rect.height())
        else:
            # 默认淡入淡出
            return self._create_fade_animation(old_widget, new_widget)
        
        # 显示新控件并设置初始位置
        new_widget.move(new_start_pos)
        new_widget.show()
        
        # 创建并行动画组
        animation_group = QParallelAnimationGroup()
        
        # 新控件滑入动画
        new_animation = QPropertyAnimation(new_widget, b"pos")
        new_animation.setDuration(self.animation_duration)
        new_animation.setStartValue(new_start_pos)
        new_animation.setEndValue(new_end_pos)
        new_animation.setEasingCurve(QEasingCurve.OutCubic)
        animation_group.addAnimation(new_animation)
        
        # 旧控件滑出动画
        old_animation = QPropertyAnimation(old_widget, b"pos")
        old_animation.setDuration(self.animation_duration)
        old_animation.setStartValue(QPoint(0, 0))
        old_animation.setEndValue(old_end_pos)
        old_animation.setEasingCurve(QEasingCurve.OutCubic)
        animation_group.addAnimation(old_animation)
        
        return animation_group
    
    def _create_fade_animation(self, old_widget: QWidget, 
                             new_widget: QWidget) -> QAbstractAnimation:
        """创建淡入淡出动画"""
        animation_group = QSequentialAnimationGroup()
        
        # 淡出动画
        fade_out = QPropertyAnimation(old_widget, b"windowOpacity")
        fade_out.setDuration(self.animation_duration // 2)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.OutCubic)
        animation_group.addAnimation(fade_out)
        
        # 切换控件
        def switch_widgets():
            old_widget.hide()
            new_widget.setWindowOpacity(0.0)
            new_widget.show()
        
        fade_out.finished.connect(switch_widgets)
        
        # 淡入动画
        fade_in = QPropertyAnimation(new_widget, b"windowOpacity")
        fade_in.setDuration(self.animation_duration // 2)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.OutCubic)
        animation_group.addAnimation(fade_in)
        
        return animation_group


class FlowLayout(QLayout):
    """流式布局"""
    
    def __init__(self, margin: int = 0, h_spacing: int = -1, v_spacing: int = -1):
        super().__init__()
        self.item_list = []
        self.h_space = h_spacing
        self.v_space = v_spacing
        
        self.setContentsMargins(margin, margin, margin, margin)
    
    def addItem(self, item: QLayoutItem):
        """添加项目"""
        self.item_list.append(item)
    
    def count(self) -> int:
        """获取项目数量"""
        return len(self.item_list)
    
    def itemAt(self, index: int) -> QLayoutItem:
        """获取指定索引的项目"""
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None
    
    def takeAt(self, index: int) -> QLayoutItem:
        """移除并返回指定索引的项目"""
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None
    
    def expandingDirections(self) -> Qt.Orientations:
        """返回扩展方向"""
        return Qt.Orientations(Qt.Orientation(0))
    
    def hasHeightForWidth(self) -> bool:
        """是否有基于宽度的高度"""
        return True
    
    def heightForWidth(self, width: int) -> int:
        """根据宽度计算高度"""
        height = self._do_layout(QRect(0, 0, width, 0), True)
        return height
    
    def setGeometry(self, rect: QRect):
        """设置几何形状"""
        super().setGeometry(rect)
        self._do_layout(rect, False)
    
    def sizeHint(self) -> QSize:
        """大小提示"""
        return self.minimumSize()
    
    def minimumSize(self) -> QSize:
        """最小大小"""
        size = QSize()
        
        for item in self.item_list:
            size = size.expandedTo(item.minimumSize())
        
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(),
                     margins.top() + margins.bottom())
        
        return size
    
    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        """执行布局"""
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        
        for item in self.item_list:
            widget = item.widget()
            
            space_x = self.h_space
            if space_x == -1:
                space_x = widget.style().layoutSpacing(
                    widget.sizePolicy().controlType(),
                    widget.sizePolicy().controlType(),
                    Qt.Horizontal
                )
            
            space_y = self.v_space
            if space_y == -1:
                space_y = widget.style().layoutSpacing(
                    widget.sizePolicy().controlType(),
                    widget.sizePolicy().controlType(),
                    Qt.Vertical
                )
            
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        
        return y + line_height - rect.y() + bottom


class AnimatedWidget(QWidget):
    """带动画效果的控件基类"""
    
    def __init__(self):
        super().__init__()
        self.animations = {}
        self.animation_group = QParallelAnimationGroup()
    
    def animate_property(self, property_name: str, start_value, end_value,
                        duration: int = 300, easing: QEasingCurve.Type = QEasingCurve.OutCubic):
        """动画化属性"""
        animation = QPropertyAnimation(self, property_name.encode())
        animation.setDuration(duration)
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setEasingCurve(easing)
        
        self.animations[property_name] = animation
        return animation
    
    def animate_geometry(self, start_rect: QRect, end_rect: QRect,
                        duration: int = 300):
        """动画化几何形状"""
        return self.animate_property("geometry", start_rect, end_rect, duration)
    
    def animate_opacity(self, start_opacity: float, end_opacity: float,
                       duration: int = 300):
        """动画化透明度"""
        return self.animate_property("windowOpacity", start_opacity, end_opacity, duration)
    
    def animate_size(self, start_size: QSize, end_size: QSize,
                    duration: int = 300):
        """动画化大小"""
        return self.animate_property("size", start_size, end_size, duration)
    
    def animate_position(self, start_pos: QPoint, end_pos: QPoint,
                        duration: int = 300):
        """动画化位置"""
        return self.animate_property("pos", start_pos, end_pos, duration)
    
    def start_animation(self, property_name: str):
        """开始动画"""
        if property_name in self.animations:
            self.animations[property_name].start()
    
    def start_all_animations(self):
        """开始所有动画"""
        self.animation_group.clear()
        for animation in self.animations.values():
            self.animation_group.addAnimation(animation)
        self.animation_group.start()
    
    def stop_all_animations(self):
        """停止所有动画"""
        self.animation_group.stop()
        for animation in self.animations.values():
            animation.stop()


class ResponsiveScrollArea(QScrollArea):
    """响应式滚动区域"""
    
    def __init__(self):
        super().__init__()
        self.setup_responsive_scrolling()
    
    def setup_responsive_scrolling(self):
        """设置响应式滚动"""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 设置现代化滚动条样式
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0.1);
                width: 8px;
                border-radius: 4px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 0, 0, 0.5);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
            QScrollBar:horizontal {
                background: rgba(0, 0, 0, 0.1);
                height: 8px;
                border-radius: 4px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(0, 0, 0, 0.5);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
                width: 0px;
            }
        """)


class ResponsiveMainWindow(QWidget):
    """响应式主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setup_responsive_ui()
    
    def setup_responsive_ui(self):
        """设置响应式界面"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建响应式容器
        self.responsive_container = ResponsiveWidget()
        self.responsive_container.breakpointChanged.connect(self.on_breakpoint_changed)
        
        # 使用滚动区域包装
        scroll_area = ResponsiveScrollArea()
        scroll_area.setWidget(self.responsive_container)
        
        self.main_layout.addWidget(scroll_area)
    
    def on_breakpoint_changed(self, breakpoint: BreakPoint):
        """断点变化处理"""
        print(f"断点变化: {breakpoint.name} ({breakpoint.value}px)")
        
        # 可以在这里根据断点调整UI
        if breakpoint == BreakPoint.XS:
            self.adapt_for_mobile()
        elif breakpoint in [BreakPoint.SM, BreakPoint.MD]:
            self.adapt_for_tablet()
        else:
            self.adapt_for_desktop()
    
    def adapt_for_mobile(self):
        """适配移动设备"""
        # 移动设备适配逻辑
        pass
    
    def adapt_for_tablet(self):
        """适配平板设备"""
        # 平板设备适配逻辑
        pass
    
    def adapt_for_desktop(self):
        """适配桌面设备"""
        # 桌面设备适配逻辑
        pass
    
    def add_responsive_widget(self, widget: QWidget, **kwargs):
        """添加响应式控件"""
        self.responsive_container.add_widget(widget, **kwargs)


def main():
    """测试主函数"""
    app = QApplication(sys.argv)
    
    # 创建响应式主窗口
    window = ResponsiveMainWindow()
    window.setWindowTitle("响应式布局测试")
    window.resize(1000, 600)
    
    # 添加测试控件
    for i in range(12):
        label = QLabel(f"控件 {i+1}")
        label.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                border: 1px solid #2196f3;
                border-radius: 4px;
                padding: 20px;
                margin: 5px;
                font-size: 14px;
                text-align: center;
            }
        """)
        label.setMinimumHeight(100)
        
        # 不同的响应式配置
        if i % 4 == 0:
            window.add_responsive_widget(label, xs_span=1, sm_span=2, md_span=2, lg_span=2, xl_span=3)
        elif i % 4 == 1:
            window.add_responsive_widget(label, xs_span=1, sm_span=1, md_span=1, lg_span=1, xl_span=1)
        elif i % 4 == 2:
            window.add_responsive_widget(label, xs_span=1, sm_span=2, md_span=3, lg_span=2, xl_span=2)
        else:
            window.add_responsive_widget(label, xs_span=1, sm_span=1, md_span=2, lg_span=3, xl_span=2)
    
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
