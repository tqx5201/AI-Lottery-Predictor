"""
可视化配置管理界面
提供直观的配置管理功能，支持实时预览和验证
"""

import sys
import json
import os
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QTextEdit, QGroupBox, QScrollArea,
    QSplitter, QTreeWidget, QTreeWidgetItem, QFrame, QProgressBar,
    QMessageBox, QFileDialog, QColorDialog, QFontDialog,
    QApplication, QStyleFactory, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve,
    QRect, QSize, QPoint
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPixmap, QIcon, QPainter, QBrush,
    QLinearGradient, QPen
)

logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """配置项类型"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    COLOR = "color"
    FONT = "font"
    FILE_PATH = "file_path"
    DIRECTORY = "directory"
    JSON = "json"


@dataclass
class ConfigItem:
    """配置项"""
    key: str
    name: str
    description: str
    config_type: ConfigType
    default_value: Any
    current_value: Any = None
    choices: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    validator: Optional[Callable[[Any], bool]] = None
    category: str = "general"
    requires_restart: bool = False
    
    def __post_init__(self):
        if self.current_value is None:
            self.current_value = self.default_value


class ModernButton(QPushButton):
    """现代化按钮"""
    
    def __init__(self, text: str, primary: bool = False):
        super().__init__(text)
        self.primary = primary
        self.setMinimumHeight(35)
        self.setFont(QFont("Segoe UI", 9))
        self._setup_style()
    
    def _setup_style(self):
        """设置样式"""
        if self.primary:
            style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4CAF50, stop:1 #45a049);
                    border: none;
                    border-radius: 6px;
                    color: white;
                    font-weight: bold;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5CBF60, stop:1 #55b059);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3CAF40, stop:1 #359f39);
                }
            """
        else:
            style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f8f9fa, stop:1 #e9ecef);
                    border: 1px solid #dee2e6;
                    border-radius: 6px;
                    color: #495057;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    border-color: #adb5bd;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e9ecef, stop:1 #dee2e6);
                }
            """
        self.setStyleSheet(style)


class ModernGroupBox(QGroupBox):
    """现代化分组框"""
    
    def __init__(self, title: str):
        super().__init__(title)
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #495057;
                background-color: white;
            }
        """)


class ConfigItemWidget(QWidget):
    """配置项控件"""
    
    valueChanged = pyqtSignal(str, object)  # key, value
    
    def __init__(self, config_item: ConfigItem):
        super().__init__()
        self.config_item = config_item
        self.control_widget = None
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 标签
        label = QLabel(self.config_item.name)
        label.setFont(QFont("Segoe UI", 9))
        label.setMinimumWidth(150)
        label.setToolTip(self.config_item.description)
        layout.addWidget(label)
        
        # 控件
        self.control_widget = self._create_control_widget()
        layout.addWidget(self.control_widget)
        
        # 描述标签
        if self.config_item.description:
            desc_label = QLabel(self.config_item.description)
            desc_label.setFont(QFont("Segoe UI", 8))
            desc_label.setStyleSheet("color: #6c757d;")
            desc_label.setWordWrap(True)
            desc_label.setMaximumWidth(200)
            layout.addWidget(desc_label)
        
        # 重启提示
        if self.config_item.requires_restart:
            restart_label = QLabel("*需要重启")
            restart_label.setFont(QFont("Segoe UI", 8))
            restart_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            layout.addWidget(restart_label)
        
        layout.addStretch()
    
    def _create_control_widget(self) -> QWidget:
        """创建控制控件"""
        config_type = self.config_item.config_type
        
        if config_type == ConfigType.STRING:
            return self._create_string_widget()
        elif config_type == ConfigType.INTEGER:
            return self._create_integer_widget()
        elif config_type == ConfigType.FLOAT:
            return self._create_float_widget()
        elif config_type == ConfigType.BOOLEAN:
            return self._create_boolean_widget()
        elif config_type == ConfigType.CHOICE:
            return self._create_choice_widget()
        elif config_type == ConfigType.COLOR:
            return self._create_color_widget()
        elif config_type == ConfigType.FONT:
            return self._create_font_widget()
        elif config_type == ConfigType.FILE_PATH:
            return self._create_file_path_widget()
        elif config_type == ConfigType.DIRECTORY:
            return self._create_directory_widget()
        elif config_type == ConfigType.JSON:
            return self._create_json_widget()
        else:
            return self._create_string_widget()
    
    def _create_string_widget(self) -> QWidget:
        """创建字符串控件"""
        line_edit = QLineEdit()
        line_edit.setText(str(self.config_item.current_value))
        line_edit.textChanged.connect(
            lambda text: self.valueChanged.emit(self.config_item.key, text)
        )
        return line_edit
    
    def _create_integer_widget(self) -> QWidget:
        """创建整数控件"""
        spin_box = QSpinBox()
        if self.config_item.min_value is not None:
            spin_box.setMinimum(int(self.config_item.min_value))
        if self.config_item.max_value is not None:
            spin_box.setMaximum(int(self.config_item.max_value))
        
        spin_box.setValue(int(self.config_item.current_value))
        spin_box.valueChanged.connect(
            lambda value: self.valueChanged.emit(self.config_item.key, value)
        )
        return spin_box
    
    def _create_float_widget(self) -> QWidget:
        """创建浮点数控件"""
        spin_box = QDoubleSpinBox()
        spin_box.setDecimals(3)
        
        if self.config_item.min_value is not None:
            spin_box.setMinimum(float(self.config_item.min_value))
        if self.config_item.max_value is not None:
            spin_box.setMaximum(float(self.config_item.max_value))
        
        spin_box.setValue(float(self.config_item.current_value))
        spin_box.valueChanged.connect(
            lambda value: self.valueChanged.emit(self.config_item.key, value)
        )
        return spin_box
    
    def _create_boolean_widget(self) -> QWidget:
        """创建布尔控件"""
        check_box = QCheckBox()
        check_box.setChecked(bool(self.config_item.current_value))
        check_box.stateChanged.connect(
            lambda state: self.valueChanged.emit(
                self.config_item.key, state == Qt.Checked
            )
        )
        return check_box
    
    def _create_choice_widget(self) -> QWidget:
        """创建选择控件"""
        combo_box = QComboBox()
        if self.config_item.choices:
            combo_box.addItems(self.config_item.choices)
            if self.config_item.current_value in self.config_item.choices:
                combo_box.setCurrentText(str(self.config_item.current_value))
        
        combo_box.currentTextChanged.connect(
            lambda text: self.valueChanged.emit(self.config_item.key, text)
        )
        return combo_box
    
    def _create_color_widget(self) -> QWidget:
        """创建颜色控件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 颜色显示
        color_label = QLabel()
        color_label.setFixedSize(30, 30)
        color_label.setStyleSheet(f"background-color: {self.config_item.current_value}; border: 1px solid #ccc;")
        layout.addWidget(color_label)
        
        # 选择按钮
        select_btn = QPushButton("选择颜色")
        select_btn.clicked.connect(lambda: self._select_color(color_label))
        layout.addWidget(select_btn)
        
        return widget
    
    def _select_color(self, color_label: QLabel):
        """选择颜色"""
        color = QColorDialog.getColor(QColor(self.config_item.current_value), self)
        if color.isValid():
            color_name = color.name()
            color_label.setStyleSheet(f"background-color: {color_name}; border: 1px solid #ccc;")
            self.valueChanged.emit(self.config_item.key, color_name)
    
    def _create_font_widget(self) -> QWidget:
        """创建字体控件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 字体显示
        font_label = QLabel("字体预览 AaBbCc")
        if isinstance(self.config_item.current_value, str):
            # 如果是字符串，解析为字体
            try:
                font_data = json.loads(self.config_item.current_value)
                font = QFont(font_data.get('family', 'Arial'), font_data.get('size', 12))
                font_label.setFont(font)
            except:
                pass
        layout.addWidget(font_label)
        
        # 选择按钮
        select_btn = QPushButton("选择字体")
        select_btn.clicked.connect(lambda: self._select_font(font_label))
        layout.addWidget(select_btn)
        
        return widget
    
    def _select_font(self, font_label: QLabel):
        """选择字体"""
        font, ok = QFontDialog.getFont(font_label.font(), self)
        if ok:
            font_label.setFont(font)
            font_data = {
                'family': font.family(),
                'size': font.pointSize(),
                'bold': font.bold(),
                'italic': font.italic()
            }
            self.valueChanged.emit(self.config_item.key, json.dumps(font_data))
    
    def _create_file_path_widget(self) -> QWidget:
        """创建文件路径控件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 路径输入框
        path_edit = QLineEdit()
        path_edit.setText(str(self.config_item.current_value))
        path_edit.textChanged.connect(
            lambda text: self.valueChanged.emit(self.config_item.key, text)
        )
        layout.addWidget(path_edit)
        
        # 浏览按钮
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(lambda: self._browse_file(path_edit))
        layout.addWidget(browse_btn)
        
        return widget
    
    def _browse_file(self, path_edit: QLineEdit):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", path_edit.text())
        if file_path:
            path_edit.setText(file_path)
    
    def _create_directory_widget(self) -> QWidget:
        """创建目录控件"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 路径输入框
        path_edit = QLineEdit()
        path_edit.setText(str(self.config_item.current_value))
        path_edit.textChanged.connect(
            lambda text: self.valueChanged.emit(self.config_item.key, text)
        )
        layout.addWidget(path_edit)
        
        # 浏览按钮
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(lambda: self._browse_directory(path_edit))
        layout.addWidget(browse_btn)
        
        return widget
    
    def _browse_directory(self, path_edit: QLineEdit):
        """浏览目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录", path_edit.text())
        if dir_path:
            path_edit.setText(dir_path)
    
    def _create_json_widget(self) -> QWidget:
        """创建JSON控件"""
        text_edit = QTextEdit()
        text_edit.setMaximumHeight(100)
        
        # 格式化JSON显示
        try:
            if isinstance(self.config_item.current_value, str):
                json_data = json.loads(self.config_item.current_value)
            else:
                json_data = self.config_item.current_value
            
            formatted_json = json.dumps(json_data, indent=2, ensure_ascii=False)
            text_edit.setPlainText(formatted_json)
        except:
            text_edit.setPlainText(str(self.config_item.current_value))
        
        text_edit.textChanged.connect(
            lambda: self.valueChanged.emit(self.config_item.key, text_edit.toPlainText())
        )
        
        return text_edit


class ConfigPreviewWidget(QWidget):
    """配置预览控件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("配置预览")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 预览区域
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.preview_text)
    
    def update_preview(self, config_data: Dict[str, Any]):
        """更新预览"""
        try:
            formatted_json = json.dumps(config_data, indent=2, ensure_ascii=False)
            self.preview_text.setPlainText(formatted_json)
        except Exception as e:
            self.preview_text.setPlainText(f"预览错误: {e}")


class ConfigValidationWidget(QWidget):
    """配置验证控件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("配置验证")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 验证结果
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(150)
        layout.addWidget(self.validation_text)
        
        # 验证按钮
        validate_btn = ModernButton("验证配置", primary=True)
        validate_btn.clicked.connect(self.validate_config)
        layout.addWidget(validate_btn)
    
    def validate_config(self):
        """验证配置"""
        # 这里可以添加具体的验证逻辑
        self.validation_text.setPlainText("✅ 配置验证通过")


class ConfigManagerWidget(QWidget):
    """配置管理主控件"""
    
    configChanged = pyqtSignal(str, object)  # key, value
    
    def __init__(self):
        super().__init__()
        self.config_items = {}
        self.config_widgets = {}
        self.setup_ui()
        self.load_default_config()
    
    def setup_ui(self):
        """设置界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：配置项
        left_widget = self._create_config_panel()
        splitter.addWidget(left_widget)
        
        # 右侧：预览和验证
        right_widget = self._create_preview_panel()
        splitter.addWidget(right_widget)
        
        # 设置分割比例
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
    
    def _create_config_panel(self) -> QWidget:
        """创建配置面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标题
        title = QLabel("系统配置")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        layout.addWidget(title)
        
        # 标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        reset_btn = ModernButton("重置为默认")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        import_btn = ModernButton("导入配置")
        import_btn.clicked.connect(self.import_config)
        button_layout.addWidget(import_btn)
        
        export_btn = ModernButton("导出配置")
        export_btn.clicked.connect(self.export_config)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        save_btn = ModernButton("保存配置", primary=True)
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        return widget
    
    def _create_preview_panel(self) -> QWidget:
        """创建预览面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 预览控件
        self.preview_widget = ConfigPreviewWidget()
        layout.addWidget(self.preview_widget)
        
        # 验证控件
        self.validation_widget = ConfigValidationWidget()
        layout.addWidget(self.validation_widget)
        
        return widget
    
    def load_default_config(self):
        """加载默认配置"""
        # 定义默认配置项
        default_configs = [
            # 界面配置
            ConfigItem(
                key="ui.theme",
                name="界面主题",
                description="选择界面主题风格",
                config_type=ConfigType.CHOICE,
                choices=["light", "dark", "auto"],
                default_value="light",
                category="界面"
            ),
            ConfigItem(
                key="ui.font_size",
                name="字体大小",
                description="界面字体大小",
                config_type=ConfigType.INTEGER,
                min_value=8,
                max_value=20,
                default_value=9,
                category="界面"
            ),
            ConfigItem(
                key="ui.language",
                name="语言",
                description="界面显示语言",
                config_type=ConfigType.CHOICE,
                choices=["zh_CN", "en_US"],
                default_value="zh_CN",
                category="界面",
                requires_restart=True
            ),
            ConfigItem(
                key="ui.animation_enabled",
                name="启用动画",
                description="启用界面动画效果",
                config_type=ConfigType.BOOLEAN,
                default_value=True,
                category="界面"
            ),
            
            # 数据配置
            ConfigItem(
                key="data.cache_size",
                name="缓存大小(MB)",
                description="数据缓存最大大小",
                config_type=ConfigType.INTEGER,
                min_value=10,
                max_value=1000,
                default_value=100,
                category="数据"
            ),
            ConfigItem(
                key="data.auto_backup",
                name="自动备份",
                description="启用自动数据备份",
                config_type=ConfigType.BOOLEAN,
                default_value=True,
                category="数据"
            ),
            ConfigItem(
                key="data.backup_interval",
                name="备份间隔(小时)",
                description="自动备份的时间间隔",
                config_type=ConfigType.INTEGER,
                min_value=1,
                max_value=168,
                default_value=24,
                category="数据"
            ),
            
            # 性能配置
            ConfigItem(
                key="performance.gpu_enabled",
                name="启用GPU加速",
                description="启用GPU加速计算",
                config_type=ConfigType.BOOLEAN,
                default_value=True,
                category="性能",
                requires_restart=True
            ),
            ConfigItem(
                key="performance.thread_count",
                name="线程数量",
                description="并行处理线程数量",
                config_type=ConfigType.INTEGER,
                min_value=1,
                max_value=16,
                default_value=4,
                category="性能"
            ),
            ConfigItem(
                key="performance.memory_limit",
                name="内存限制(MB)",
                description="最大内存使用限制",
                config_type=ConfigType.INTEGER,
                min_value=512,
                max_value=8192,
                default_value=2048,
                category="性能"
            ),
            
            # 网络配置
            ConfigItem(
                key="network.timeout",
                name="网络超时(秒)",
                description="网络请求超时时间",
                config_type=ConfigType.INTEGER,
                min_value=5,
                max_value=60,
                default_value=30,
                category="网络"
            ),
            ConfigItem(
                key="network.retry_count",
                name="重试次数",
                description="网络请求失败重试次数",
                config_type=ConfigType.INTEGER,
                min_value=0,
                max_value=10,
                default_value=3,
                category="网络"
            ),
            ConfigItem(
                key="network.proxy_enabled",
                name="启用代理",
                description="启用网络代理",
                config_type=ConfigType.BOOLEAN,
                default_value=False,
                category="网络"
            ),
            ConfigItem(
                key="network.proxy_host",
                name="代理主机",
                description="代理服务器地址",
                config_type=ConfigType.STRING,
                default_value="",
                category="网络"
            ),
        ]
        
        # 添加配置项
        for config_item in default_configs:
            self.add_config_item(config_item)
        
        # 更新预览
        self.update_preview()
    
    def add_config_item(self, config_item: ConfigItem):
        """添加配置项"""
        self.config_items[config_item.key] = config_item
        
        # 获取或创建标签页
        tab_name = config_item.category
        tab_widget = None
        
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == tab_name:
                tab_widget = self.tab_widget.widget(i)
                break
        
        if tab_widget is None:
            tab_widget = QScrollArea()
            tab_content = QWidget()
            tab_layout = QVBoxLayout(tab_content)
            tab_layout.setAlignment(Qt.AlignTop)
            tab_widget.setWidget(tab_content)
            tab_widget.setWidgetResizable(True)
            self.tab_widget.addTab(tab_widget, tab_name)
        
        # 创建配置项控件
        config_widget = ConfigItemWidget(config_item)
        config_widget.valueChanged.connect(self.on_config_changed)
        
        # 添加到布局
        tab_content = tab_widget.widget()
        tab_content.layout().addWidget(config_widget)
        
        self.config_widgets[config_item.key] = config_widget
    
    def on_config_changed(self, key: str, value: Any):
        """配置项变化"""
        if key in self.config_items:
            self.config_items[key].current_value = value
            self.configChanged.emit(key, value)
            self.update_preview()
    
    def update_preview(self):
        """更新预览"""
        config_data = {}
        for key, item in self.config_items.items():
            config_data[key] = item.current_value
        
        self.preview_widget.update_preview(config_data)
    
    def get_config_data(self) -> Dict[str, Any]:
        """获取配置数据"""
        return {key: item.current_value for key, item in self.config_items.items()}
    
    def set_config_data(self, config_data: Dict[str, Any]):
        """设置配置数据"""
        for key, value in config_data.items():
            if key in self.config_items:
                self.config_items[key].current_value = value
                # TODO: 更新控件显示
        
        self.update_preview()
    
    def reset_to_defaults(self):
        """重置为默认值"""
        reply = QMessageBox.question(
            self, "确认重置",
            "确定要重置所有配置为默认值吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for item in self.config_items.values():
                item.current_value = item.default_value
            
            # TODO: 更新所有控件显示
            self.update_preview()
            QMessageBox.information(self, "重置完成", "配置已重置为默认值")
    
    def import_config(self):
        """导入配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入配置", "", "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                self.set_config_data(config_data)
                QMessageBox.information(self, "导入成功", "配置已成功导入")
                
            except Exception as e:
                QMessageBox.critical(self, "导入失败", f"导入配置失败：{e}")
    
    def export_config(self):
        """导出配置"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出配置", "config.json", "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                config_data = self.get_config_data()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "导出成功", f"配置已导出到：{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出配置失败：{e}")
    
    def save_config(self):
        """保存配置"""
        try:
            config_data = self.get_config_data()
            
            # 这里应该调用实际的配置保存逻辑
            # 例如：save_to_database(config_data)
            
            QMessageBox.information(self, "保存成功", "配置已保存")
            
            # 检查是否需要重启
            restart_required = any(
                item.requires_restart for item in self.config_items.values()
            )
            
            if restart_required:
                reply = QMessageBox.question(
                    self, "需要重启",
                    "某些配置更改需要重启应用程序才能生效。\n是否现在重启？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # 重启应用程序
                    QApplication.quit()
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存配置失败：{e}")


def main():
    """测试主函数"""
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # 设置应用程序样式
    app.setStyleSheet("""
        QWidget {
            font-family: "Segoe UI", "Microsoft YaHei";
        }
        QTabWidget::pane {
            border: 1px solid #c0c0c0;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #f0f0f0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #4CAF50;
        }
    """)
    
    window = ConfigManagerWidget()
    window.setWindowTitle("配置管理器")
    window.resize(1000, 700)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
