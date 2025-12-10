from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton


class TabSwitcher(QWidget):
    tab_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = {}
        self.current_tab = None
        self.setup_ui()

    def setup_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)
        self.layout.addStretch()

    def add_tab(self, tab_id: str, label: str):
        btn = QPushButton(f"/ {label}")
        btn.setObjectName("tab_switcher_inactive")
        btn.clicked.connect(lambda: self._on_tab_clicked(tab_id))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(btn.sizePolicy().horizontalPolicy(), btn.sizePolicy().verticalPolicy())
        btn.adjustSize()

        self.tabs[tab_id] = btn
        self.layout.addWidget(btn)

        if len(self.tabs) == 2:
            self.layout.addStretch()

        if self.current_tab is None:
            self.set_active_tab(tab_id)

    def _on_tab_clicked(self, tab_id: str):
        self.set_active_tab(tab_id)
        self.tab_changed.emit(tab_id)

    def set_active_tab(self, tab_id: str):
        if tab_id not in self.tabs:
            return

        for tid, btn in self.tabs.items():
            if tid == tab_id:
                btn.setObjectName("tab_switcher_active")
            else:
                btn.setObjectName("tab_switcher_inactive")
            btn.setStyleSheet("")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()

        self.current_tab = tab_id
