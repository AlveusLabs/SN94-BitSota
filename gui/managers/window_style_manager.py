"""窗口样式管理器 - 负责窗口设置和平台特定的样式"""

import platform
from ctypes import c_void_p
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


class WindowStyleManager:
    """管理窗口样式、大小、位置和标题栏颜色"""

    def __init__(self, window):
        """
        初始化窗口样式管理器
        
        Args:
            window: QMainWindow 实例
        """
        self.window = window
        self._titlebar_set = False

    def setup_window(self):
        """设置窗口的基本属性：标题、大小、位置"""
        self.window.setWindowTitle("BitSota")
        self.window.setMinimumSize(1200, 800)
        self.window.resize(1400, 900)
        self._center_window()

    def _center_window(self):
        """将窗口居中显示在主屏幕上"""
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.window.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.window.move(window_geometry.topLeft())

    def handle_show_event(self, event):
        """
        处理窗口显示事件，用于设置标题栏颜色
        
        应该在 QMainWindow 的 showEvent() 中调用此方法
        """
        if not self._titlebar_set:
            # 延迟设置，确保窗口完全初始化
            QTimer.singleShot(100, self.set_titlebar_color)
            self._titlebar_set = True

    def set_titlebar_color(self):
        """设置标题栏颜色为 #0C0029（平台特定实现）"""
        if platform.system() == "Darwin":  # macOS
            self._set_macos_titlebar_color()
        elif platform.system() == "Windows":
            self._set_windows_titlebar_color()

    def _set_macos_titlebar_color(self):
        """设置 macOS 标题栏颜色"""
        try:
            from Cocoa import NSColor
            import objc
            from Cocoa import NSView

            # 获取原生 NSWindow
            window = self.window.windowHandle()
            if window:
                ns_view = window.winId()
                view = objc.objc_object(c_void_p=ns_view)
                ns_window = view.window()

                if ns_window:
                    # 设置标题栏颜色 #0C0029 = RGB(12, 0, 41)
                    color = NSColor.colorWithRed_green_blue_alpha_(
                        12.0 / 255.0,  # R
                        0.0 / 255.0,   # G
                        41.0 / 255.0,  # B
                        1.0            # Alpha
                    )
                    ns_window.setBackgroundColor_(color)
                    ns_window.setTitlebarAppearsTransparent_(True)

            print("macOS title bar color set to #0C0029")
        except ImportError:
            print("PyObjC (Cocoa) not installed. Install with: pip install pyobjc-framework-Cocoa")
        except Exception as e:
            print(f"Could not set macOS title bar color: {e}")

    def _set_windows_titlebar_color(self):
        """设置 Windows 标题栏颜色"""
        try:
            from ctypes import windll, c_int, byref, sizeof
            
            hwnd = int(self.window.winId())
            # DWMWA_CAPTION_COLOR = 35
            # Color in BGR format: #0C0029 -> 0x29000C
            color_value = c_int(0x0029000C)
            windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 35, byref(color_value), sizeof(color_value)
            )
            print("Windows title bar color set to #0C0029")
        except Exception as e:
            print(f"Could not set Windows title bar color: {e}")
