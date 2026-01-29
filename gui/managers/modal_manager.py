"""模态框管理器 - 负责模态框显示和覆盖层管理"""

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QStackedWidget, QWidget

from gui.components import (
    UserGuideModal,
    ColdkeyAddressModal,
    ComingSoonModal,
    ModalOverlay,
)
from gui.components.import_confirmation_modals import ErrorModal


class ModalManager(QObject):
    """管理模态框的显示和覆盖层"""

    # 信号
    user_guide_completed = Signal()
    coldkey_address_submitted = Signal(str)  # address

    def __init__(
        self,
        main_window,
        modal_overlay: ModalOverlay,
        content_stack: QStackedWidget,
        screen_stack: QWidget,
        app_container: QWidget,
        parent=None
    ):
        """
        初始化模态框管理器
        
        Args:
            main_window: 主窗口对象
            modal_overlay: 模态覆盖层组件
            content_stack: 内容堆栈
            screen_stack: 屏幕堆栈
            app_container: 应用容器
            parent: 父对象
        """
        super().__init__(parent)
        self.main_window = main_window
        self.modal_overlay = modal_overlay
        self.content_stack = content_stack
        self.screen_stack = screen_stack
        self.app_container = app_container

    def show_modal(self, modal):
        """
        显示模态框并管理覆盖层
        
        Args:
            modal: 要显示的模态框对象
            
        Returns:
            模态框的返回值
        """
        # 更新覆盖层几何以匹配 screen_stack
        self.update_overlay_geometry()
        self.modal_overlay.raise_()
        self.modal_overlay.show()

        # 显示对话框
        result = modal.exec()

        # 对话框关闭后隐藏覆盖层
        self.modal_overlay.hide()

        return result

    def update_overlay_geometry(self):
        """更新覆盖层位置和大小以匹配内容区域"""
        # 仅在 app_container 可见时更新
        if self.content_stack.currentWidget() == self.app_container:
            # 获取 screen_stack 相对于中央 widget 的位置
            central_widget = self.main_window.centralWidget()
            pos = self.screen_stack.mapTo(central_widget, self.screen_stack.rect().topLeft())
            # 设置覆盖层几何以匹配 screen_stack
            self.modal_overlay.setGeometry(
                pos.x(),
                pos.y(),
                self.screen_stack.width(),
                self.screen_stack.height()
            )

    def handle_resize_event(self):
        """处理窗口大小调整事件"""
        self.update_overlay_geometry()

    def handle_stack_change(self):
        """处理堆栈变化事件"""
        # 使用 QTimer 确保布局完成后再更新
        QTimer.singleShot(0, self.update_overlay_geometry)

    def show_user_guide(self):
        """显示用户指南模态框"""
        guide_modal = UserGuideModal(parent=self.main_window)
        guide_modal.proceed_clicked.connect(self._on_user_guide_proceed)
        self.show_modal(guide_modal)

    def _on_user_guide_proceed(self):
        """用户指南完成后的处理"""
        self.user_guide_completed.emit()

    def show_coldkey_prompt(self):
        """显示 coldkey 地址输入提示"""
        coldkey_modal = ColdkeyAddressModal(parent=self.main_window)
        coldkey_modal.address_submitted.connect(self._on_coldkey_submitted)
        self.show_modal(coldkey_modal)

    def _on_coldkey_submitted(self, address: str):
        """Coldkey 地址提交后的处理"""
        self.coldkey_address_submitted.emit(address)

    def show_coming_soon(self, title: str, message: str):
        """
        显示"即将推出"模态框
        
        Args:
            title: 标题
            message: 消息内容
        """
        modal = ComingSoonModal(title, message, parent=self.main_window)
        self.show_modal(modal)

    def show_error(self, title: str, message: str):
        """
        显示错误模态框
        
        Args:
            title: 错误标题
            message: 错误消息
        """
        error_modal = ErrorModal(title, message, parent=self.main_window)
        self.show_modal(error_modal)
