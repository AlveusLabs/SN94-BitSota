from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget
from typing import List, Tuple, Optional, Callable

from gui.components.common.tab_switcher import TabSwitcher


class NormalTabSwitcher(QWidget):
    """
    Generic tab switcher component with content management
    
    Args:
        tabs: List of (tab_id, tab_label) tuples
        on_tab_changed: Optional callback function(tab_id: str) -> None
        show_content: Whether to show content area below tabs (default: False)
        parent: Parent widget
    
    Example:
        # Simple usage with callback
        def on_tab_changed(tab_id: str):
            if tab_id == "direct":
                self.show_direct_mining()
            else:
                self.show_pool_mining()
        
        tabs = [("direct", "Direct Mining"), ("pool", "Pool Mining")]
        switcher = NormalTabSwitcher(tabs, on_tab_changed=on_tab_changed)
        
        # Usage with content widgets
        tabs = [("tab1", "Tab 1"), ("tab2", "Tab 2")]
        switcher = NormalTabSwitcher(tabs, show_content=True)
        switcher.add_tab_content("tab1", widget1)
        switcher.add_tab_content("tab2", widget2)
    """
    
    tab_changed = Signal(str)
    
    def __init__(
        self, 
        tabs: List[Tuple[str, str]], 
        on_tab_changed: Optional[Callable[[str], None]] = None,
        show_content: bool = False,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.tabs = tabs
        self.on_tab_changed_callback = on_tab_changed
        self.show_content = show_content
        self.content_widgets = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize the tab switcher UI"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Tab switcher container (centered)
        tab_container_layout = QHBoxLayout()
        tab_container_layout.setContentsMargins(0, 0, 0, 0)
        tab_container_layout.addStretch()
        
        # Tab switcher
        self.tab_switcher = TabSwitcher()
        for tab_id, tab_label in self.tabs:
            self.tab_switcher.add_tab(tab_id, tab_label)
        self.tab_switcher.tab_changed.connect(self._on_tab_changed)
        
        tab_container_layout.addWidget(self.tab_switcher)
        tab_container_layout.addStretch()
        
        main_layout.addLayout(tab_container_layout)
        
        # Content area (optional)
        if self.show_content:
            self.content_stack = QStackedWidget()
            main_layout.addWidget(self.content_stack)
    
    def _on_tab_changed(self, tab_id: str):
        """Handle tab change event"""
        # Emit signal
        self.tab_changed.emit(tab_id)
        
        # Call callback if provided
        if self.on_tab_changed_callback:
            self.on_tab_changed_callback(tab_id)
        
        # Switch content widget if content area is enabled
        if self.show_content and tab_id in self.content_widgets:
            widget = self.content_widgets[tab_id]
            self.content_stack.setCurrentWidget(widget)
    
    def add_tab_content(self, tab_id: str, widget: QWidget):
        """
        Add content widget for a specific tab
        
        Args:
            tab_id: The tab identifier
            widget: The widget to display when this tab is active
        """
        if not self.show_content:
            raise ValueError("Content area is not enabled. Set show_content=True in constructor.")
        
        self.content_widgets[tab_id] = widget
        self.content_stack.addWidget(widget)
        
        # If this is the current tab, show it
        if self.get_current_tab() == tab_id:
            self.content_stack.setCurrentWidget(widget)
    
    def set_active_tab(self, tab_id: str):
        """Set the active tab programmatically"""
        self.tab_switcher.set_active_tab(tab_id)
    
    def get_current_tab(self) -> str:
        """Get the currently active tab ID"""
        return self.tab_switcher.current_tab
    
    def get_current_content_widget(self) -> Optional[QWidget]:
        """Get the currently displayed content widget"""
        if not self.show_content:
            return None
        return self.content_stack.currentWidget()
