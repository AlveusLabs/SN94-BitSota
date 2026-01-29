"""Update Manager - handles update checking and notifications"""

import webbrowser
from PySide6.QtCore import QObject, Signal, QTimer

from gui.update_checker import UpdateChecker
from gui.components.modals.update import UpdateAvailableModal


class UpdateManager(QObject):
    """Manages application update checking and notifications"""

    # Signals
    update_available = Signal(dict)  # update_info

    def __init__(self, main_window, parent=None):
        """
        Initialize update manager
        
        Args:
            main_window: Main window object
            parent: Parent object
        """
        super().__init__(parent)
        self.main_window = main_window
        self.update_checker = UpdateChecker()
        self.update_check_timer = None

    def setup(self):
        """Setup update checker and timer"""
        # Check for updates 2 seconds after startup
        QTimer.singleShot(2000, self.check_on_startup)

        # Check for updates periodically every 24 hours
        self.update_check_timer = QTimer()
        self.update_check_timer.timeout.connect(self.check_periodic)
        self.update_check_timer.start(24 * 60 * 60 * 1000)  # 24 hours in ms

    def check_on_startup(self):
        """Check for updates on startup"""
        print("[UpdateManager] Checking for updates on startup...")
        update_info = self.update_checker.check_for_updates(force=True)
        if update_info:
            print(f"[UpdateManager] Update found: {update_info}")
            self.update_available.emit(update_info)
            self.show_update_modal(update_info)
        else:
            print("[UpdateManager] No updates available")

    def check_periodic(self):
        """Periodic update check"""
        print("[UpdateManager] Periodic update check...")
        update_info = self.update_checker.check_for_updates()
        if update_info:
            print(f"[UpdateManager] Update found: {update_info}")
            self.update_available.emit(update_info)
            self.show_update_modal(update_info)

    def show_update_modal(self, update_info: dict):
        """
        Show update available modal
        
        Args:
            update_info: Update info dictionary
        """
        # Get modal_manager (if exists)
        # Otherwise show modal directly
        if hasattr(self.main_window, 'modal_manager'):
            modal = UpdateAvailableModal(update_info, parent=self.main_window)
            modal.download_clicked.connect(lambda: self.download_update(update_info))
            modal.skip_clicked.connect(lambda: self.skip_version(update_info))
            self.main_window.modal_manager.show_modal(modal)
        else:
            # Fallback to direct display
            modal = UpdateAvailableModal(update_info, parent=self.main_window)
            modal.download_clicked.connect(lambda: self.download_update(update_info))
            modal.skip_clicked.connect(lambda: self.skip_version(update_info))
            modal.exec()

    def download_update(self, update_info: dict):
        """
        Open download link
        
        Args:
            update_info: Update info dictionary
        """
        download_url = self.update_checker.get_download_url(update_info)
        if download_url:
            webbrowser.open(download_url)
            print(f"Opening download URL: {download_url}")
        else:
            print("No download URL available for this platform")

    def skip_version(self, update_info: dict):
        """
        Skip this version
        
        Args:
            update_info: Update info dictionary
        """
        self.update_checker.skip_version(update_info['new_version_code'])
        print(f"Skipped version {update_info['new_version']}")
