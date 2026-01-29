"""Window Style Manager - handles window settings and platform-specific styles"""

import platform
from ctypes import c_void_p
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


class WindowStyleManager:
    """Manages window style, size, position and titlebar color"""

    def __init__(self, window):
        """
        Initialize window style manager
        
        Args:
            window: QMainWindow instance
        """
        self.window = window
        self._titlebar_set = False

    def setup_window(self):
        """Set up basic window properties: title, size, position"""
        self.window.setWindowTitle("BitSota")
        self.window.setMinimumSize(1200, 800)
        self.window.resize(1400, 900)
        self._center_window()

    def _center_window(self):
        """Center the window on the primary screen"""
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.window.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.window.move(window_geometry.topLeft())

    def handle_show_event(self, event):
        """
        Handle window show event for setting titlebar color
        
        Should be called in QMainWindow's showEvent() method
        """
        if not self._titlebar_set:
            # Delay to ensure window is fully initialized
            QTimer.singleShot(100, self.set_titlebar_color)
            self._titlebar_set = True

    def set_titlebar_color(self):
        """Set titlebar color to #0C0029 (platform-specific implementation)"""
        if platform.system() == "Darwin":  # macOS
            self._set_macos_titlebar_color()
        elif platform.system() == "Windows":
            self._set_windows_titlebar_color()

    def _set_macos_titlebar_color(self):
        """Set macOS titlebar color"""
        try:
            from Cocoa import NSColor
            import objc

            # Get native NSWindow
            window = self.window.windowHandle()
            if window:
                ns_view = window.winId()
                view = objc.objc_object(c_void_p=ns_view)
                ns_window = view.window()

                if ns_window:
                    # Set titlebar color #0C0029 = RGB(12, 0, 41)
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
        """Set Windows titlebar color"""
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
