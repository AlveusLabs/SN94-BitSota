from PySide6.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QTextEdit,
    QFileDialog,
)
from typing import Optional, Dict, Any, TYPE_CHECKING
import threading

from gui.config_manager import config_manager
from gui.components import PrimaryButton, SecondaryButton
from gui.resource_path import resource_path

if TYPE_CHECKING:
    from neuroevo.miner_client import NeuroevoClient


class NeuroevoMiningTask(QRunnable):
    class Signals(QObject):
        log = Signal(str)
        error = Signal(str)
        finished = Signal()
        stats = Signal(dict)

    def __init__(self, config: Dict[str, Any], wallet):
        super().__init__()
        self.config = config
        self.wallet = wallet
        self.client: Optional["NeuroevoClient"] = None
        self.stop_event = threading.Event()
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def stop(self):
        self.stop_event.set()
        if self.client:
            self.client.stop()

    @Slot()
    def run(self):
        from neuroevo.miner_client import NeuroevoClient  # Local import to avoid heavy deps at startup

        try:
            self.client = NeuroevoClient(
                self.config["server_url"],
                miner_id=self.config.get("miner_id") or None,
                wallet=self.wallet,
                data_root=self.config.get("data_root") or None,
                log_fn=self.signals.log.emit,
                stats_callback=self.signals.stats.emit,
                stop_event=self.stop_event,
            )
            self.client.run_forever(
                capacity=self.config.get("capacity", 4),
                sleep_idle=self.config.get("sleep_idle", 1.0),
            )
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            if self.client:
                self.client.close()
            self.signals.finished.emit()


class DistributedTrainingScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.thread_pool = QThreadPool()
        self.mining_task: Optional[NeuroevoMiningTask] = None
        self.is_running = False
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(24)

        title = QLabel("Distributed Training")
        title.setObjectName("section_title")
        main_layout.addWidget(title)

        description = QLabel(
            "Run Neuroevo parameter-server training locally or against a remote coordinator."
        )
        description.setObjectName("mining_description")
        description.setWordWrap(True)
        main_layout.addWidget(description)

        content_box = QWidget()
        content_box.setObjectName("content_box")
        content_layout = QVBoxLayout(content_box)
        content_layout.setContentsMargins(24, 32, 24, 32)
        content_layout.setSpacing(24)

        config_section = self._create_config_section()
        content_layout.addWidget(config_section)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(24)
        stats_layout.addWidget(self._create_status_panel())
        stats_layout.addWidget(self._create_stats_panel())
        content_layout.addLayout(stats_layout)

        logs_section = self._create_logs_section()
        content_layout.addWidget(logs_section)

        main_layout.addWidget(content_box)

    def _create_config_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Connection Settings")
        title.setObjectName("section_title")
        layout.addWidget(title)

        # Server URL
        server_label = QLabel("Server URL")
        server_label.setObjectName("form_label")
        layout.addWidget(server_label)
        self.server_input = QLineEdit()
        self.server_input.setObjectName("form_input")
        layout.addWidget(self.server_input)

        row_one = QHBoxLayout()
        row_one.setSpacing(12)

        capacity_col = QVBoxLayout()
        capacity_label = QLabel("Capacity")
        capacity_label.setObjectName("form_label")
        self.capacity_input = QSpinBox()
        self.capacity_input.setRange(1, 1024)
        self.capacity_input.setObjectName("form_input")
        capacity_col.addWidget(capacity_label)
        capacity_col.addWidget(self.capacity_input)
        row_one.addLayout(capacity_col)

        sleep_col = QVBoxLayout()
        sleep_label = QLabel("Idle Sleep (s)")
        sleep_label.setObjectName("form_label")
        self.sleep_input = QDoubleSpinBox()
        self.sleep_input.setRange(0.0, 60.0)
        self.sleep_input.setSingleStep(0.1)
        self.sleep_input.setDecimals(2)
        self.sleep_input.setObjectName("form_input")
        sleep_col.addWidget(sleep_label)
        sleep_col.addWidget(self.sleep_input)
        row_one.addLayout(sleep_col)

        layout.addLayout(row_one)

        # Data root selection
        data_label = QLabel("Dataset Root (optional)")
        data_label.setObjectName("form_label")
        layout.addWidget(data_label)

        data_row = QHBoxLayout()
        data_row.setSpacing(8)
        self.data_root_input = QLineEdit()
        self.data_root_input.setObjectName("form_input")
        data_row.addWidget(self.data_root_input)
        browse_btn = SecondaryButton("Browse", width=120, height=40)
        browse_btn.clicked.connect(self._browse_data_root)
        data_row.addWidget(browse_btn)
        layout.addLayout(data_row)

        # Miner ID optional
        miner_label = QLabel("Miner ID (optional)")
        miner_label.setObjectName("form_label")
        layout.addWidget(miner_label)
        self.miner_id_input = QLineEdit()
        self.miner_id_input.setObjectName("form_input")
        layout.addWidget(self.miner_id_input)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self.save_button = SecondaryButton("Save Configuration", width=200, height=48)
        self.save_button.clicked.connect(self._save_config)
        button_row.addWidget(self.save_button)

        self.start_button = PrimaryButton(
            "Start Distributed Training",
            width=260,
            height=48,
            icon_path=resource_path("gui/images/play.svg"),
        )
        self.start_button.clicked.connect(self._toggle_training)
        button_row.addWidget(self.start_button)

        layout.addLayout(button_row)

        return section

    def _create_status_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("stats_box")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Connection Status")
        title.setObjectName("section_title")
        layout.addWidget(title)

        self.wallet_info = QLabel("Wallet: Not Connected")
        self.wallet_info.setObjectName("stat_label")
        layout.addWidget(self.wallet_info)

        self.connection_status = QLabel("Status: Idle")
        self.connection_status.setObjectName("stat_value")
        layout.addWidget(self.connection_status)

        self.model_version_label = QLabel("Model: -")
        self.model_version_label.setObjectName("stat_value")
        layout.addWidget(self.model_version_label)

        self.pending_label = QLabel("Pending: -")
        self.pending_label.setObjectName("stat_value")
        layout.addWidget(self.pending_label)

        layout.addStretch()
        return panel

    def _create_stats_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("stats_box")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("Training Stats")
        title.setObjectName("section_title")
        layout.addWidget(title)

        self.epoch_label = QLabel("Epoch: -")
        self.epoch_label.setObjectName("stat_label")
        layout.addWidget(self.epoch_label)

        self.avg_bits_label = QLabel("Avg Bits: -")
        self.avg_bits_label.setObjectName("stat_label")
        layout.addWidget(self.avg_bits_label)

        self.throughput_label = QLabel("Throughput: -")
        self.throughput_label.setObjectName("stat_label")
        layout.addWidget(self.throughput_label)

        layout.addStretch()
        return panel

    def _create_logs_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("Logs")
        title.setObjectName("section_title")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("clear_logs_button")
        clear_btn.clicked.connect(lambda: self.logs_text.clear())
        header.addWidget(clear_btn)

        layout.addLayout(header)

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setMinimumHeight(200)
        layout.addWidget(self.logs_text)

        return section

    def _load_config(self):
        cfg = config_manager.get_section("neuroevo") or {}
        self.server_input.setText(cfg.get("server_url", "http://127.0.0.1:8000"))
        self.capacity_input.setValue(int(cfg.get("capacity", 4) or 4))
        self.sleep_input.setValue(float(cfg.get("sleep_idle", 1.0) or 1.0))
        self.data_root_input.setText(cfg.get("data_root", ""))
        self.miner_id_input.setText(cfg.get("miner_id", ""))

    def _save_config(self):
        cfg = config_manager._config or {}
        cfg.setdefault("neuroevo", {})
        cfg["neuroevo"].update(self._collect_config())
        config_manager.save_config(cfg)
        self._append_log("[config] distributed training settings saved")

    def _collect_config(self) -> Dict[str, Any]:
        return {
            "server_url": self.server_input.text().strip() or "http://127.0.0.1:8000",
            "capacity": int(self.capacity_input.value()),
            "sleep_idle": float(self.sleep_input.value()),
            "data_root": self.data_root_input.text().strip(),
            "miner_id": self.miner_id_input.text().strip(),
        }

    def _browse_data_root(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if directory:
            self.data_root_input.setText(directory)

    def _toggle_training(self):
        if self.is_running:
            self._stop_training()
        else:
            self._start_training()

    def _start_training(self):
        if not self.main_window or not getattr(self.main_window, "wallet", None):
            self._append_log("ERROR: Wallet not connected. Please set up a wallet first.")
            return

        config = self._collect_config()
        if not config["server_url"]:
            self._append_log("ERROR: Server URL is required.")
            return

        self.mining_task = NeuroevoMiningTask(config=config, wallet=self.main_window.wallet)
        self.mining_task.signals.log.connect(self._append_log)
        self.mining_task.signals.error.connect(self._handle_error)
        self.mining_task.signals.finished.connect(self._on_task_finished)
        self.mining_task.signals.stats.connect(self._update_stats)

        self.thread_pool.start(self.mining_task)
        self.is_running = True
        self.connection_status.setText("Status: Running")
        self.start_button.update_icon(resource_path("gui/images/stop.svg"))
        self.start_button.update_text("Stop Distributed Training")
        self._append_log("[distributed] started neuroevo mining")

    def _stop_training(self):
        if self.mining_task:
            self.mining_task.stop()
            self._append_log("[distributed] stopping training...")
        self.connection_status.setText("Status: Stopping")

    def _on_task_finished(self):
        self.is_running = False
        self.mining_task = None
        self.connection_status.setText("Status: Idle")
        self.start_button.update_icon(resource_path("gui/images/play.svg"))
        self.start_button.update_text("Start Distributed Training")
        self._append_log("[distributed] training stopped")

    def _handle_error(self, message: str):
        self._append_log(f"ERROR: {message}")

    def _append_log(self, message: str):
        self.logs_text.append(message)

    def _update_stats(self, stats: Dict[str, Any]):
        if stats.get("model_version"):
            self.model_version_label.setText(f"Model: {stats['model_version']}")
        if stats.get("pending") is not None:
            self.pending_label.setText(f"Pending: {stats.get('pending')}")
        if stats.get("event") == "work":
            self.connection_status.setText("Status: Working")
            self.epoch_label.setText(f"Epoch: {stats.get('epoch')}")
            avg_bits = stats.get("avg_bits")
            if avg_bits is not None:
                self.avg_bits_label.setText(f"Avg Bits: {avg_bits:.3f}")
            throughput = stats.get("throughput")
            if throughput is not None:
                self.throughput_label.setText(f"Throughput: {throughput:.1f} tok/s")
        elif stats.get("event") == "idle":
            if self.is_running:
                self.connection_status.setText("Status: Waiting")
            else:
                self.connection_status.setText("Status: Idle")

    def update_wallet_status(self, wallet_name: str):
        if wallet_name:
            self.wallet_info.setText(f"Wallet: {wallet_name}")
        else:
            self.wallet_info.setText("Wallet: Not Connected")
