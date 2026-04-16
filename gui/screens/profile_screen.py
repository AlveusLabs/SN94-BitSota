from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.app_config import get_app_config
from gui.components.import_confirmation_modals import ErrorModal
from gui.merkle_claim_client import (
    MerkleClaimClient,
    MerkleClaimPackage,
    resolve_claim_endpoint,
    resolve_metadata_path,
)


class _ClaimLoadWorker(QRunnable):
    class Signals(QObject):
        loaded = Signal(object)
        error = Signal(str)

    def __init__(self, client: MerkleClaimClient, hotkey: str) -> None:
        super().__init__()
        self.client = client
        self.hotkey = str(hotkey or "").strip()
        self.signals = self.Signals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            claims = self.client.list_claim_packages(hotkey=self.hotkey)
            self.signals.loaded.emit(claims)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class _ClaimSubmitWorker(QRunnable):
    class Signals(QObject):
        finished = Signal(object)
        error = Signal(str)

    def __init__(self, client: MerkleClaimClient, signer, claim: MerkleClaimPackage) -> None:
        super().__init__()
        self.client = client
        self.signer = signer
        self.claim = claim
        self.signals = self.Signals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            result = self.client.submit_claim(signer=self.signer, claim=self.claim, transfer=True)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))


class ProfileScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.thread_pool = QThreadPool()
        self._claim_rows: list[MerkleClaimPackage] = []
        self._claimed_session: set[tuple[int, int]] = set()
        self._active_claim_key: Optional[tuple[int, int]] = None
        self._claim_buttons: dict[tuple[int, int], QPushButton] = {}
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(24)

        title = QLabel("Total Overview")
        title.setObjectName("section_title")
        main_layout.addWidget(title)

        tab_container = QWidget()
        tab_layout = QHBoxLayout(tab_container)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        self.direct_tab = QPushButton("/ Direct Mining")
        self.direct_tab.setObjectName("tab_switcher_active")
        self.direct_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.direct_tab.clicked.connect(lambda: self._switch_tab("direct"))
        tab_layout.addWidget(self.direct_tab)

        self.pool_tab = QPushButton("/ Pool Mining")
        self.pool_tab.setObjectName("tab_switcher_inactive")
        self.pool_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pool_tab.clicked.connect(lambda: self._switch_tab("pool"))
        tab_layout.addWidget(self.pool_tab)

        tab_layout.addStretch()
        main_layout.addWidget(tab_container)

        self.table_container = QWidget()
        self.table_layout = QVBoxLayout(self.table_container)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_layout.setSpacing(0)

        self.direct_table = self._create_direct_mining_table()
        self.pool_table = self._create_pool_mining_table()

        self.table_layout.addWidget(self.direct_table)
        self.table_layout.addWidget(self.pool_table)
        self.direct_table.show()
        self.pool_table.hide()

        main_layout.addWidget(self.table_container)

    def _switch_tab(self, tab_type):
        if tab_type == "direct":
            self.direct_tab.setObjectName("tab_switcher_active")
            self.pool_tab.setObjectName("tab_switcher_inactive")
            self.direct_table.show()
            self.pool_table.hide()
        else:
            self.direct_tab.setObjectName("tab_switcher_inactive")
            self.pool_tab.setObjectName("tab_switcher_active")
            self.direct_table.hide()
            self.pool_table.show()
            self.refresh_data()

        self.direct_tab.setStyleSheet("")
        self.pool_tab.setStyleSheet("")
        self.direct_tab.style().unpolish(self.direct_tab)
        self.pool_tab.style().unpolish(self.pool_tab)
        self.direct_tab.style().polish(self.direct_tab)
        self.pool_tab.style().polish(self.pool_tab)

    def _create_direct_mining_table(self) -> QWidget:
        table = QWidget()
        table.setObjectName("content_box")
        table.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(table)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = self._create_table_header(
            ["Mining Mode", "Status", "Action"]
        )
        layout.addWidget(header)
        layout.addWidget(self._separator(0.12))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 16, 0, 16)
        row_layout.setSpacing(16)

        for value in (
            "Direct Mining",
            "No on-chain claim flow wired in this screen yet",
        ):
            label = QLabel(value)
            label.setObjectName("stat_value")
            row_layout.addWidget(label, 1)

        action_label = QLabel("N/A")
        action_label.setObjectName("stat_value")
        action_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        action_label.setStyleSheet("color: rgba(21, 0, 73, 0.60);")
        row_layout.addWidget(action_label, 0)

        layout.addWidget(row)
        layout.addWidget(self._separator(0.08))
        return table

    def _create_pool_mining_table(self) -> QWidget:
        table = QWidget()
        table.setObjectName("content_box")
        table.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(table)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        header = self._create_table_header(
            ["Epoch", "Rewards", "Recipient", "Merkle Root", "Claim"]
        )
        layout.addWidget(header)
        layout.addWidget(self._separator(0.12))

        self.pool_status_label = QLabel("Open this tab to load claimable pool rewards.")
        self.pool_status_label.setObjectName("form_label")
        self.pool_status_label.setWordWrap(True)
        self.pool_status_label.setStyleSheet("color: rgba(21, 0, 73, 0.60); padding: 16px 0;")
        layout.addWidget(self.pool_status_label)

        self.pool_rows_container = QWidget()
        self.pool_rows_layout = QVBoxLayout(self.pool_rows_container)
        self.pool_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.pool_rows_layout.setSpacing(0)
        layout.addWidget(self.pool_rows_container)
        return table

    def _create_table_header(self, columns) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(16)

        for i, col in enumerate(columns):
            label = QLabel(col)
            label.setObjectName("form_label")
            if i == len(columns) - 1:
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
                layout.addWidget(label, 0)
            else:
                layout.addWidget(label, 1)

        return header

    @staticmethod
    def _separator(alpha: float) -> QFrame:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(
            f"background-color: rgba(21, 0, 73, {alpha:.2f}); max-height: 1px;"
        )
        return separator

    @staticmethod
    def _format_amount(amount_rao: int) -> str:
        return f"{(int(amount_rao) / 1e9):.6f} TAO"

    @staticmethod
    def _shorten(value: str, head: int = 8, tail: int = 6) -> str:
        value = str(value or "").strip()
        if len(value) <= head + tail + 3:
            return value
        return f"{value[:head]}...{value[-tail:]}"

    def _build_claim_client(self) -> MerkleClaimClient:
        cfg = get_app_config()
        claim_endpoint = resolve_claim_endpoint(
            explicit=cfg.merkle_claim_endpoint,
            pool_endpoint=cfg.pool_endpoint,
        )
        metadata_path = resolve_metadata_path(cfg.onchain_metadata_path)
        return MerkleClaimClient(
            claim_endpoint=claim_endpoint,
            onchain_ws_url=cfg.onchain_ws_url,
            contract_address=cfg.onchain_contract,
            metadata_path=metadata_path,
        )

    def refresh_data(self) -> None:
        wallet = getattr(self.main_window, "wallet", None) if self.main_window else None
        if wallet is None:
            self.pool_status_label.setText("Load a wallet to view claimable pool rewards.")
            self._render_claim_rows([])
            return

        client = self._build_claim_client()
        if not client.is_configured():
            self.pool_status_label.setText(
                "Merkle claiming is not configured. Open Research Setup and choose a network preset."
            )
            self._render_claim_rows([])
            return

        self.pool_status_label.setText("Loading claimable pool rewards...")
        worker = _ClaimLoadWorker(client, getattr(wallet.hotkey, "ss58_address", ""))
        worker.signals.loaded.connect(self._on_claims_loaded)
        worker.signals.error.connect(self._on_claims_error)
        self.thread_pool.start(worker)

    def _render_claim_rows(self, claims: list[MerkleClaimPackage]) -> None:
        self._claim_rows = list(claims)
        self._claim_buttons = {}
        while self.pool_rows_layout.count():
            item = self.pool_rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not claims:
            empty = QLabel("No claimable Merkle rewards found for this hotkey.")
            empty.setObjectName("stat_value")
            empty.setStyleSheet("color: rgba(21, 0, 73, 0.60); padding: 16px 0;")
            self.pool_rows_layout.addWidget(empty)
            return

        for claim in claims:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 16, 0, 16)
            layout.setSpacing(16)

            values = [
                str(claim.epoch),
                self._format_amount(claim.amount_rao),
                self._shorten(claim.recipient_address),
                self._shorten(claim.root, head=10, tail=8),
            ]
            for value in values:
                label = QLabel(value)
                label.setObjectName("stat_value")
                layout.addWidget(label, 1)

            if claim.claim_key in self._claimed_session:
                action_label = QLabel("Claimed")
                action_label.setObjectName("stat_value")
                action_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                action_label.setStyleSheet("color: rgba(21, 0, 73, 0.60);")
                layout.addWidget(action_label, 0)
            else:
                button = QPushButton("Claim")
                button.setObjectName("primary_button")
                button.setFixedSize(120, 40)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(lambda _checked=False, package=claim: self._on_claim_clicked(package))
                if self._active_claim_key == claim.claim_key:
                    button.setEnabled(False)
                    button.setText("Claiming...")
                layout.addWidget(button, 0, Qt.AlignmentFlag.AlignRight)
                self._claim_buttons[claim.claim_key] = button

            self.pool_rows_layout.addWidget(row)
            self.pool_rows_layout.addWidget(self._separator(0.08))

    def _on_claim_clicked(self, claim: MerkleClaimPackage) -> None:
        wallet = getattr(self.main_window, "wallet", None) if self.main_window else None
        if wallet is None:
            self._show_error("Wallet Required", "Load a wallet before claiming rewards.")
            return

        client = self._build_claim_client()
        if not client.is_configured():
            self._show_error(
                "Claiming Not Configured",
                "The GUI is missing Merkle claim endpoint or contract settings.",
            )
            return

        self._active_claim_key = claim.claim_key
        self.pool_status_label.setText(
            f"Submitting claim for epoch {claim.epoch} index {claim.index}... local chains can take 10-20s."
        )
        self._render_claim_rows(self._claim_rows)

        worker = _ClaimSubmitWorker(client, wallet.hotkey, claim)
        worker.signals.finished.connect(lambda result, package=claim: self._on_claim_finished(package, result))
        worker.signals.error.connect(lambda message, package=claim: self._on_claim_error(package, message))
        self.thread_pool.start(worker)

    def _on_claims_loaded(self, claims_obj) -> None:
        claims = list(claims_obj or [])
        if claims:
            self.pool_status_label.setText(
                f"Found {len(claims)} claimable pool reward epoch(s) for the connected hotkey."
            )
        else:
            self.pool_status_label.setText("No claimable Merkle rewards found for the connected hotkey.")
        self._render_claim_rows(claims)

    def _on_claims_error(self, message: str) -> None:
        self.pool_status_label.setText("Failed to load Merkle claims.")
        self._render_claim_rows([])
        self._show_error("Merkle Claim Load Failed", str(message or "Unknown error"))

    def _on_claim_finished(self, claim: MerkleClaimPackage, result_obj) -> None:
        self._active_claim_key = None
        self._claimed_session.add(claim.claim_key)
        result = dict(result_obj or {})
        extrinsic_hash = str(result.get("extrinsic_hash") or "").strip()
        if extrinsic_hash:
            self.pool_status_label.setText(
                f"Claimed epoch {claim.epoch} successfully. Tx: {self._shorten(extrinsic_hash, head=12, tail=10)}"
            )
        else:
            self.pool_status_label.setText(f"Claimed epoch {claim.epoch} successfully.")
        remaining = [row for row in self._claim_rows if row.claim_key != claim.claim_key]
        self._render_claim_rows(remaining)

    def _on_claim_error(self, claim: MerkleClaimPackage, message: str) -> None:
        self._active_claim_key = None
        self._render_claim_rows(self._claim_rows)
        client = self._build_claim_client()
        details = str(message or "Unknown error")
        if "ContractReverted" in details and client.is_locally_claimed(claim):
            self._on_claim_finished(claim, {})
            return
        if "ContractReverted" in details:
            self.pool_status_label.setText("Claim reverted on-chain. Refreshing claim list...")
            self.refresh_data()
            self._show_error(
                "Merkle Claim Reverted",
                "The claim reverted on-chain. It may already be claimed or the local chain may still be processing.",
            )
            return
        self.pool_status_label.setText("Claim transaction failed.")
        self._show_error("Merkle Claim Failed", details)

    def _show_error(self, title: str, message: str) -> None:
        modal = ErrorModal(title, message, parent=self)
        modal.exec()
