"""更新管理器 - 负责检查更新和显示更新通知"""

import webbrowser
from PySide6.QtCore import QObject, Signal, QTimer

from gui.update_checker import UpdateChecker
from gui.components import UpdateAvailableModal


class UpdateManager(QObject):
    """管理应用程序更新检查和通知"""

    # 信号
    update_available = Signal(dict)  # update_info

    def __init__(self, main_window, parent=None):
        """
        初始化更新管理器
        
        Args:
            main_window: 主窗口对象
            parent: 父对象
        """
        super().__init__(parent)
        self.main_window = main_window
        self.update_checker = UpdateChecker()
        self.update_check_timer = None

    def setup(self):
        """设置更新检查器和定时器"""
        # 启动后 2 秒检查更新
        QTimer.singleShot(2000, self.check_on_startup)

        # 每 24 小时定期检查更新
        self.update_check_timer = QTimer()
        self.update_check_timer.timeout.connect(self.check_periodic)
        self.update_check_timer.start(24 * 60 * 60 * 1000)  # 24 hours in ms

    def check_on_startup(self):
        """启动时检查更新"""
        print("[UpdateManager] Checking for updates on startup...")
        update_info = self.update_checker.check_for_updates(force=True)
        if update_info:
            print(f"[UpdateManager] Update found: {update_info}")
            self.update_available.emit(update_info)
            self.show_update_modal(update_info)
        else:
            print("[UpdateManager] No updates available")

    def check_periodic(self):
        """定期检查更新"""
        print("[UpdateManager] Periodic update check...")
        update_info = self.update_checker.check_for_updates()
        if update_info:
            print(f"[UpdateManager] Update found: {update_info}")
            self.update_available.emit(update_info)
            self.show_update_modal(update_info)

    def show_update_modal(self, update_info: dict):
        """
        显示更新可用模态框
        
        Args:
            update_info: 更新信息字典
        """
        # 获取 modal_manager（如果存在）
        # 否则直接显示模态框
        if hasattr(self.main_window, 'modal_manager'):
            modal = UpdateAvailableModal(update_info, parent=self.main_window)
            modal.download_clicked.connect(lambda: self.download_update(update_info))
            modal.skip_clicked.connect(lambda: self.skip_version(update_info))
            self.main_window.modal_manager.show_modal(modal)
        else:
            # 回退到直接显示
            modal = UpdateAvailableModal(update_info, parent=self.main_window)
            modal.download_clicked.connect(lambda: self.download_update(update_info))
            modal.skip_clicked.connect(lambda: self.skip_version(update_info))
            modal.exec()

    def download_update(self, update_info: dict):
        """
        打开下载链接
        
        Args:
            update_info: 更新信息字典
        """
        download_url = self.update_checker.get_download_url(update_info)
        if download_url:
            webbrowser.open(download_url)
            print(f"Opening download URL: {download_url}")
        else:
            print("No download URL available for this platform")

    def skip_version(self, update_info: dict):
        """
        跳过此版本
        
        Args:
            update_info: 更新信息字典
        """
        self.update_checker.skip_version(update_info['new_version_code'])
        print(f"Skipped version {update_info['new_version']}")
