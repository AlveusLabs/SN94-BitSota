from __future__ import annotations

import shutil

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.app_config import (
    DEFAULT_LOCAL_PRESET,
    DEFAULT_TESTNET_PRESET,
    build_preset_research_agent_command,
    get_app_config,
    infer_research_agent_provider,
    research_agent_provider_label,
    save_app_config,
)
from gui.components.button import PrimaryButton, SecondaryButton


class ResearchSetupModal(QDialog):
    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg = get_app_config(force_reload=True)
        self._detected_provider = infer_research_agent_provider(
            configured_provider=getattr(self._cfg, "research_agent_provider", ""),
            agent_command=getattr(self._cfg, "research_agent_command", ""),
            llm_model=getattr(self._cfg, "research_llm_model", ""),
            llm_base_url=getattr(self._cfg, "research_llm_base_url", ""),
            llm_api_key=getattr(self._cfg, "research_llm_api_key", ""),
        )
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setMinimumSize(760, 620)
        self.resize(760, 700)
        self.setWindowTitle("Research Setup")
        self._setup_ui()
        self._load_current_values()
        self._refresh_provider_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        title = QLabel("Research Setup")
        title.setObjectName("modal_title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose a network preset and agent provider. The app will save the config for both "
            "source and packaged builds."
        )
        subtitle.setObjectName("modal_message")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(18)

        self.network_combo = QComboBox()
        self.network_combo.setObjectName("form_input")
        self.network_combo.addItem("Autoresearch Testnet", DEFAULT_TESTNET_PRESET)
        self.network_combo.addItem("Local Dev", DEFAULT_LOCAL_PRESET)
        body_layout.addWidget(self._field("Network", self.network_combo))

        self.provider_combo = QComboBox()
        self.provider_combo.setObjectName("form_input")
        self.provider_combo.addItem("Codex CLI", "codex_cli")
        self.provider_combo.addItem("Claude Code", "claude_code")
        self.provider_combo.addItem("OpenAI-compatible API", "openai_compatible")
        self.provider_combo.addItem("Custom command", "custom_command")
        self.provider_combo.currentIndexChanged.connect(self._refresh_provider_ui)
        body_layout.addWidget(self._field("Agent provider", self.provider_combo))

        self.provider_status = QLabel("")
        self.provider_status.setWordWrap(True)
        self.provider_status.setStyleSheet("color: rgba(21, 0, 73, 0.72);")
        body_layout.addWidget(self.provider_status)

        self.provider_stack = QStackedWidget()
        self.provider_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.provider_stack.addWidget(self._build_cli_preset_panel())
        self.provider_stack.addWidget(self._build_api_panel())
        self.provider_stack.addWidget(self._build_custom_command_panel())
        body_layout.addWidget(self.provider_stack)

        metadata_note = QLabel(
            "Merkle claim metadata is bundled with the app, and wallet loading stays in the existing "
            "Setup Wallet flow."
        )
        metadata_note.setWordWrap(True)
        metadata_note.setStyleSheet("color: rgba(21, 0, 73, 0.72);")
        body_layout.addWidget(metadata_note)
        body_layout.addStretch()

        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(12)

        cancel_button = SecondaryButton("Later", width=180, height=48)
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button)

        save_button = PrimaryButton("Save Setup", width=220, height=48)
        save_button.clicked.connect(self._save)
        buttons.addWidget(save_button)
        buttons.addStretch()

        layout.addLayout(buttons)

    def _field(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        field_layout = QVBoxLayout(container)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("form_label")
        field_layout.addWidget(label)
        field_layout.addWidget(widget)
        return container

    def _build_cli_preset_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.cli_help_label = QLabel("")
        self.cli_help_label.setWordWrap(True)
        self.cli_help_label.setStyleSheet("color: rgba(21, 0, 73, 0.72);")
        layout.addWidget(self.cli_help_label)
        return panel

    def _build_api_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.api_base_url_input = QLineEdit()
        self.api_base_url_input.setObjectName("form_input")
        self.api_base_url_input.setPlaceholderText("https://openrouter.ai/api/v1")
        layout.addWidget(self._field("API base URL", self.api_base_url_input))

        self.api_model_input = QLineEdit()
        self.api_model_input.setObjectName("form_input")
        self.api_model_input.setPlaceholderText("anthropic/claude-sonnet-4")
        layout.addWidget(self._field("Model", self.api_model_input))

        self.api_key_input = QLineEdit()
        self.api_key_input.setObjectName("form_input")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Optional if already exported in the shell")
        layout.addWidget(self._field("API key", self.api_key_input))
        return panel

    def _build_custom_command_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        command_help = QLabel(
            "Advanced mode. The app will pass placeholders like `{intro_path_quoted}` and "
            "`{repo_dir_quoted}` at runtime."
        )
        command_help.setWordWrap(True)
        command_help.setStyleSheet("color: rgba(21, 0, 73, 0.72);")
        layout.addWidget(command_help)

        self.custom_command_input = QTextEdit()
        self.custom_command_input.setObjectName("logs_text")
        self.custom_command_input.setMinimumHeight(120)
        self.custom_command_input.setPlaceholderText("bash -lc 'cat {intro_path_quoted} ...'")
        layout.addWidget(self._field("Custom command", self.custom_command_input))
        return panel

    def _load_current_values(self) -> None:
        current_preset = str(getattr(self._cfg, "network_preset", DEFAULT_TESTNET_PRESET) or DEFAULT_TESTNET_PRESET)
        self._set_combo_data(self.network_combo, current_preset)

        provider = self._detected_provider or "codex_cli"
        self._set_combo_data(self.provider_combo, provider)
        self.api_base_url_input.setText(str(getattr(self._cfg, "research_llm_base_url", "") or ""))
        self.api_model_input.setText(str(getattr(self._cfg, "research_llm_model", "") or ""))
        self.api_key_input.setText(str(getattr(self._cfg, "research_llm_api_key", "") or ""))
        self.custom_command_input.setPlainText(str(getattr(self._cfg, "research_agent_command", "") or ""))

    @staticmethod
    def _set_combo_data(combo: QComboBox, wanted: str) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == wanted:
                combo.setCurrentIndex(index)
                return

    def _refresh_provider_ui(self) -> None:
        provider = str(self.provider_combo.currentData() or "")
        if provider in {"codex_cli", "claude_code"}:
            self.provider_stack.setCurrentIndex(0)
            self.cli_help_label.setText(self._cli_help_text(provider))
        elif provider == "openai_compatible":
            self.provider_stack.setCurrentIndex(1)
        else:
            self.provider_stack.setCurrentIndex(2)

        self.provider_status.setText(self._provider_status_text(provider))
        self._sync_provider_stack_height()

    def _sync_provider_stack_height(self) -> None:
        current = self.provider_stack.currentWidget()
        if current is None:
            return
        current.adjustSize()
        height = max(80, current.sizeHint().height())
        self.provider_stack.setFixedHeight(height)

    def _cli_help_text(self, provider: str) -> str:
        label = research_agent_provider_label(provider) or provider
        binary = "codex" if provider == "codex_cli" else "claude"
        found = bool(shutil.which(binary))
        status = f"`{binary}` detected on PATH." if found else f"`{binary}` was not found on PATH in this shell."
        return (
            f"{label} uses the app's built-in testnet prompt template. The GUI will generate the launch "
            f"command internally. {status}"
        )

    def _provider_status_text(self, provider: str) -> str:
        if provider == "openai_compatible":
            return "Use this for OpenRouter, Ollama, or another OpenAI-compatible `/chat/completions` endpoint."
        if provider == "custom_command":
            return "Advanced override. Use only if the built-in Codex and Claude presets do not fit."
        return "The app will generate the research agent command for you."

    def _save(self) -> None:
        provider = str(self.provider_combo.currentData() or "").strip()
        preset = str(self.network_combo.currentData() or DEFAULT_TESTNET_PRESET).strip()

        updates = {
            "network_preset": preset,
            "research_agent_provider": provider,
            "research_agent_mode": "gui_managed",
        }

        if provider in {"codex_cli", "claude_code"}:
            generated = build_preset_research_agent_command(provider)
            if not generated:
                self.provider_status.setText(
                    "The built-in master prompt file is missing from this build. Rebuild the app or use Custom command."
                )
                return
            updates.update(
                {
                    "research_agent_command": "",
                    "research_llm_base_url": str(getattr(self._cfg, "research_llm_base_url", "") or ""),
                    "research_llm_model": "",
                    "research_llm_api_key": "",
                }
            )
        elif provider == "openai_compatible":
            base_url = self.api_base_url_input.text().strip()
            model = self.api_model_input.text().strip()
            api_key = self.api_key_input.text().strip()
            if not base_url or not model:
                self.provider_status.setText("API mode requires both an API base URL and a model name.")
                return
            updates.update(
                {
                    "research_agent_command": "",
                    "research_llm_base_url": base_url,
                    "research_llm_model": model,
                    "research_llm_api_key": api_key,
                }
            )
        else:
            custom_command = self.custom_command_input.toPlainText().strip()
            if not custom_command:
                self.provider_status.setText("Custom command mode requires a command template.")
                return
            updates.update(
                {
                    "research_agent_command": custom_command,
                    "research_llm_base_url": str(getattr(self._cfg, "research_llm_base_url", "") or ""),
                    "research_llm_model": "",
                    "research_llm_api_key": "",
                }
            )

        path = save_app_config(updates)
        self.saved.emit(str(path))
        self.accept()
