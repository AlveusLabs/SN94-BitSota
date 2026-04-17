from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Optional
from urllib.parse import urlparse

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import requests
from substrateinterface import Keypair

from gui.app_config import (
    build_preset_research_agent_command,
    get_app_config,
    infer_research_agent_provider,
    research_agent_provider_label,
)
from gui.components import PrimaryButton, SecondaryButton
from gui.pool_coldkey_sync import sync_declared_coldkey_to_pool_backend
from gui.resource_path import resource_path
from gui.screens.mining_screen import MiningScreen as LegacyMiningScreen
from miner.research_competitions import CompetitionMode, list_builtin_research_competitions


POOL_GRID_COLUMNS = 2
POOL_GRID_SPACING = 24
RESEARCH_POOL_MODE = "research_pool"
DEFAULT_RESEARCH_TEST_TASK_SLUG = "cifar10-matrix-decomposition-frontier"
RESEARCH_COORDINATOR_CONNECT_TIMEOUT_S = 5.0
RESEARCH_COORDINATOR_READ_TIMEOUT_S = 30.0


def _humanize_competition_mode(mode: str | None) -> str:
    raw = str(mode or "").strip().lower()
    if raw == CompetitionMode.centerless.value:
        return "Centerless"
    if raw == CompetitionMode.peer_evaluation.value:
        return "Peer Evaluation"
    if raw == CompetitionMode.standard.value:
        return "Standard"
    return "Unknown"


def _metric_label(metric_name: str | None, metric_direction: str | None) -> str:
    name = str(metric_name or "").strip() or "metric"
    direction = str(metric_direction or "").strip().lower()
    if direction == "maximize":
        return f"{name} (maximize)"
    if direction == "minimize":
        return f"{name} (minimize)"
    return name


def _research_coordinator_timeout() -> tuple[float, float]:
    return (
        float(RESEARCH_COORDINATOR_CONNECT_TIMEOUT_S),
        float(RESEARCH_COORDINATOR_READ_TIMEOUT_S),
    )


def _research_runtime_settings() -> dict[str, str]:
    cfg = get_app_config()
    configured_provider = os.getenv("BITSOTA_RESEARCH_AGENT_PROVIDER") or str(
        getattr(cfg, "research_agent_provider", "") or ""
    )
    configured_command = os.getenv("BITSOTA_RESEARCH_AGENT_COMMAND") or str(
        getattr(cfg, "research_agent_command", "") or ""
    )
    llm_base_url = (
        os.getenv("BITSOTA_RESEARCH_LLM_BASE_URL")
        or str(getattr(cfg, "research_llm_base_url", "") or "")
    ).strip()
    llm_model = (
        os.getenv("BITSOTA_RESEARCH_LLM_MODEL")
        or str(getattr(cfg, "research_llm_model", "") or "")
    ).strip()
    llm_api_key = (
        os.getenv("BITSOTA_RESEARCH_LLM_API_KEY")
        or str(getattr(cfg, "research_llm_api_key", "") or "")
    ).strip()
    provider = infer_research_agent_provider(
        configured_provider=configured_provider,
        agent_command=configured_command,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )
    agent_command = str(configured_command or "").strip()
    if not agent_command and provider in {"codex_cli", "claude_code"}:
        agent_command = build_preset_research_agent_command(provider)
    provider_label = research_agent_provider_label(provider)
    if not provider_label and agent_command:
        try:
            provider_label = shlex.split(agent_command)[0]
        except Exception:
            provider_label = "Custom command"
    if not provider_label and llm_model:
        provider_label = llm_model

    return {
        "coordinator_url": (
            os.getenv("BITSOTA_RESEARCH_COORDINATOR_URL")
            or str(getattr(cfg, "research_coordinator_endpoint", "") or "")
        ).strip(),
        "provider": provider,
        "provider_label": provider_label or "Not configured",
        "agent_command": agent_command,
        "agent_mode": (
            os.getenv("BITSOTA_RESEARCH_AGENT_MODE")
            or str(getattr(cfg, "research_agent_mode", "") or "")
        ).strip()
        or "gui_managed",
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
    }


def _optional_env_bool(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _safe_positive_int(raw: Any, default: int) -> int:
    try:
        return max(1, int(raw))
    except Exception:
        return int(default)


def _research_gui_test_settings() -> dict[str, Any]:
    cfg = get_app_config()
    enabled = bool(getattr(cfg, "test_mode", False)) and bool(getattr(cfg, "research_test_autostart", False))
    env_enabled = _optional_env_bool("BITSOTA_RESEARCH_GUI_AUTOSTART")
    if env_enabled is not None:
        enabled = env_enabled

    auto_restart = bool(getattr(cfg, "research_test_auto_restart", True))
    env_auto_restart = _optional_env_bool("BITSOTA_RESEARCH_GUI_AUTO_RESTART")
    if env_auto_restart is not None:
        auto_restart = env_auto_restart

    task_slug = (
        os.getenv("BITSOTA_RESEARCH_GUI_TASK_SLUG")
        or str(getattr(cfg, "research_test_task_slug", "") or "")
    ).strip() or DEFAULT_RESEARCH_TEST_TASK_SLUG

    restart_delay_seconds = _safe_positive_int(
        os.getenv("BITSOTA_RESEARCH_GUI_RESTART_DELAY_SECONDS")
        or getattr(cfg, "research_test_restart_delay_seconds", 5),
        5,
    )

    return {
        "enabled": bool(enabled),
        "task_slug": str(task_slug),
        "auto_restart": bool(auto_restart),
        "restart_delay_seconds": int(restart_delay_seconds),
    }


def _ephemeral_hotkey_mnemonic() -> str:
    return (
        os.getenv("BITSOTA_RESEARCH_TEST_HOTKEY_MNEMONIC")
        or os.getenv("BITSOTA_TEST_HOTKEY_MNEMONIC")
        or ""
    ).strip()


def _ephemeral_hotkey_address(mnemonic: str) -> str:
    seed = str(mnemonic or "").strip()
    if not seed:
        return ""
    try:
        return str(Keypair.create_from_mnemonic(seed).ss58_address or "")
    except Exception:
        return ""


def _select_research_test_pool(pools: list[dict[str, Any]], task_slug: str) -> Optional[dict[str, Any]]:
    wanted = str(task_slug or "").strip().lower()
    if not wanted:
        return None
    preferred: Optional[dict[str, Any]] = None
    fallback: Optional[dict[str, Any]] = None
    for pool in pools:
        if not bool(pool.get("is_research_pool")):
            continue
        if str(pool.get("task_slug") or "").strip().lower() != wanted:
            continue
        if str(pool.get("task_id") or "").strip():
            preferred = dict(pool)
            break
        fallback = dict(pool)
    return preferred or fallback


def _fetch_research_tasks(
    *,
    coordinator_url: str,
    timeout_s: float | tuple[float, float] = (RESEARCH_COORDINATOR_CONNECT_TIMEOUT_S, RESEARCH_COORDINATOR_READ_TIMEOUT_S),
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{coordinator_url.rstrip('/')}/api/v1/tasks",
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json() or []
    return [dict(task or {}) for task in payload if isinstance(task, dict)]


def _find_research_task(
    tasks: list[dict[str, Any]],
    *,
    task_id: str = "",
    task_slug: str = "",
) -> Optional[dict[str, Any]]:
    wanted_id = str(task_id or "").strip()
    wanted_slug = str(task_slug or "").strip().lower()
    if wanted_id:
        for task in tasks:
            if str(task.get("id") or "").strip() == wanted_id:
                return dict(task)
        return None
    if wanted_slug:
        for task in tasks:
            if str(task.get("slug") or "").strip().lower() == wanted_slug:
                return dict(task)
    return None


def _resolve_research_launch_task(
    *,
    coordinator_url: str,
    task_id: str = "",
    task_slug: str = "",
    timeout_s: float | tuple[float, float] = (RESEARCH_COORDINATOR_CONNECT_TIMEOUT_S, RESEARCH_COORDINATOR_READ_TIMEOUT_S),
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        tasks = _fetch_research_tasks(coordinator_url=coordinator_url, timeout_s=timeout_s)
    except Exception as exc:
        return None, f"failed to reach research coordinator {coordinator_url}: {exc}"

    matched = _find_research_task(tasks, task_id=task_id, task_slug=task_slug)
    if matched is not None:
        return matched, None

    target = f"id={task_id}" if str(task_id or "").strip() else f"slug={task_slug}"
    return None, (
        f"selected research task {target} is not live on coordinator {coordinator_url}. "
        "The current card is a fallback template, so mining was not started."
    )


def _normalize_research_task_pool(
    *,
    task: dict[str, Any],
    coordinator_url: str,
    agent_label: str,
) -> dict[str, Any]:
    parsed = urlparse(coordinator_url) if coordinator_url else None
    host = parsed.netloc or coordinator_url or "Not configured"
    competition_mode = str(task.get("competition_mode") or CompetitionMode.standard.value)
    metric_name = str(task.get("metric_name") or "").strip()
    metric_direction = str(task.get("metric_direction") or "").strip()
    return {
        "id": f"research-task:{task.get('id')}",
        "name": str(task.get("title") or task.get("slug") or "Research Task"),
        "mode": RESEARCH_POOL_MODE,
        "mode_label": "Research Agent Pool",
        "endpoint": host,
        "backend": agent_label or "Not configured",
        "workers": 1,
        "recommended": False,
        "is_research_pool": True,
        "coordinator_url": coordinator_url,
        "task_id": str(task.get("id") or ""),
        "task_slug": str(task.get("slug") or ""),
        "task_source": "live",
        "competition_mode": competition_mode,
        "competition_mode_label": _humanize_competition_mode(competition_mode),
        "metric_label": _metric_label(metric_name, metric_direction),
        "agent_model": agent_label or "Not configured",
        "card_rows": [
            ("Mode", "Research Agent Pool"),
            ("Rules", _humanize_competition_mode(competition_mode)),
            ("Metric", _metric_label(metric_name, metric_direction)),
            ("Agent", agent_label or "Not configured"),
            ("Coordinator", host),
        ],
    }


def _fallback_research_pools(*, coordinator_url: str, agent_label: str) -> list[dict[str, Any]]:
    parsed = urlparse(coordinator_url) if coordinator_url else None
    host = parsed.netloc or coordinator_url or "Not configured"
    cards: list[dict[str, Any]] = []
    for template in list_builtin_research_competitions():
        cards.append(
            {
                "id": f"research-template:{template.slug}",
                "name": template.title,
                "mode": RESEARCH_POOL_MODE,
                "mode_label": "Research Agent Pool",
                "endpoint": host,
                "backend": agent_label or "Not configured",
                "workers": 1,
                "recommended": False,
                "is_research_pool": True,
                "coordinator_url": coordinator_url,
                "task_id": "",
                "task_slug": template.slug,
                "task_source": "template",
                "competition_mode": "",
                "competition_mode_label": "From coordinator",
                "metric_label": _metric_label(template.metric_name, template.metric_direction.value),
                "agent_model": agent_label or "Not configured",
                "card_rows": [
                    ("Mode", "Research Agent Pool"),
                    ("Rules", "From coordinator"),
                    ("Metric", _metric_label(template.metric_name, template.metric_direction.value)),
                    ("Agent", agent_label or "Not configured"),
                    ("Coordinator", host),
                ],
            }
        )
    return cards


def _configured_research_pools() -> list[dict[str, Any]]:
    settings = _research_runtime_settings()
    coordinator_url = settings["coordinator_url"]
    agent_label = settings["provider_label"]
    if not coordinator_url:
        return _fallback_research_pools(coordinator_url=coordinator_url, agent_label=agent_label)

    try:
        payload = _fetch_research_tasks(
            coordinator_url=coordinator_url,
            timeout_s=_research_coordinator_timeout(),
        )
    except Exception:
        return _fallback_research_pools(coordinator_url=coordinator_url, agent_label=agent_label)

    cards = [
        _normalize_research_task_pool(task=dict(task or {}), coordinator_url=coordinator_url, agent_label=agent_label)
        for task in payload
        if bool((task or {}).get("is_active", True))
    ]
    return cards or _fallback_research_pools(coordinator_url=coordinator_url, agent_label=agent_label)


def _configured_pools() -> list[dict]:
    cfg = get_app_config()
    endpoint = str(getattr(cfg, "pool_endpoint", "") or "").strip()
    parsed = urlparse(endpoint) if endpoint else None
    host = parsed.netloc or endpoint or "Not configured"
    backend = str(os.getenv("BITSOTA_MINER_BACKEND", "") or "").strip().lower()
    backend_label = "C++" if backend in {"cpp", "cpp_baseline", "automl_zero_cpp"} else "Python"
    workers = max(1, int(getattr(cfg, "miner_workers", 1) or 1))

    classic = [
        {
            "id": "pool_lease",
            "name": "Lease Pool",
            "mode": "pool_lease",
            "mode_label": "Lease coordinator",
            "endpoint": host,
            "backend": backend_label,
            "workers": workers,
            "recommended": True,
            "card_rows": [
                ("Mode", "Lease coordinator"),
                ("Endpoint", host),
                ("Backend", backend_label),
                ("Workers", str(workers)),
            ],
        },
        {
            "id": "pool_task",
            "name": "Task Pool",
            "mode": "pool",
            "mode_label": "Task batches",
            "endpoint": host,
            "backend": backend_label,
            "workers": workers,
            "recommended": False,
            "card_rows": [
                ("Mode", "Task batches"),
                ("Endpoint", host),
                ("Backend", backend_label),
                ("Workers", str(workers)),
            ],
        },
    ]
    return classic + _configured_research_pools()


class PoolCard(QWidget):
    join_clicked = Signal(dict)

    def __init__(self, pool_data: dict, parent=None):
        super().__init__(parent)
        self.pool_data = dict(pool_data)
        self.setObjectName("pool_card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_row = QHBoxLayout()
        title = QLabel(self.pool_data.get("name", "Pool"))
        title.setObjectName("config_section_title")
        title_row.addWidget(title)
        if self.pool_data.get("recommended"):
            badge = QLabel("Recommended")
            badge.setObjectName("pool_recommended_badge")
            title_row.addWidget(badge)
        title_row.addStretch()
        layout.addLayout(title_row)

        rows = QVBoxLayout()
        rows.setSpacing(12)
        for idx, (label, value) in enumerate(self.pool_data.get("card_rows") or []):
            _add_stat_row(rows, str(label), str(value))
            if idx == 0:
                rows.addWidget(_create_divider())
        layout.addLayout(rows)
        layout.addStretch()

        join_btn = QPushButton("Join Pool")
        join_btn.setObjectName("pool_join_btn")
        join_btn.setFixedHeight(48)
        join_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        join_btn.clicked.connect(lambda: self.join_clicked.emit(self.pool_data))
        layout.addWidget(join_btn)


class _HiddenTaskSelection:
    def __init__(self, label: str):
        self._label = str(label)

    def currentText(self) -> str:
        return self._label

    def setCurrentText(self, label: str):
        self._label = str(label)


class PoolDetailScreen(LegacyMiningScreen):
    switch_pool_requested = Signal()

    def __init__(self, pool_data: dict, main_window=None, parent=None):
        self.pool_data = dict(pool_data)
        self._runtime_started_at: float | None = None
        super().__init__(main_window=main_window, parent=parent)
        self._runtime_timer = QTimer(self)
        self._runtime_timer.timeout.connect(self._update_runtime)
        self._research_log_timer = QTimer(self)
        self._research_log_timer.timeout.connect(self._poll_research_process)
        self._research_log_path: Optional[Path] = None
        self._research_log_offset = 0
        self._manual_stop_requested = False
        self._auto_restart_enabled = False
        self._auto_restart_delay_ms = 5000
        self._last_research_exit_code: Optional[int] = None
        self._auto_restart_timer = QTimer(self)
        self._auto_restart_timer.setSingleShot(True)
        self._auto_restart_timer.timeout.connect(self._auto_restart_research_mining)

    def configure_research_test(self, *, auto_restart: bool, restart_delay_seconds: int) -> None:
        self._auto_restart_enabled = bool(auto_restart)
        self._auto_restart_delay_ms = max(1000, int(restart_delay_seconds) * 1000)

    def setup_ui(self):
        self.task_type_map = {
            str(self.pool_data.get("mode_label", "Pool")): str(self.pool_data.get("mode") or "pool_lease")
        }

        only_label = next(iter(self.task_type_map.keys()))
        self.task_type_combo = _HiddenTaskSelection(only_label)
        self.workers_combo = _HiddenTaskSelection(str(self.pool_data.get("workers", 1)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        layout.addWidget(self._create_config_section())

        stats_row = QHBoxLayout()
        stats_row.setSpacing(24)
        stats_row.addWidget(self._create_miner_stats(), 1)
        stats_row.addWidget(self._create_mining_status(), 1)
        stats_row.addWidget(self._create_pool_status(), 1)
        layout.addLayout(stats_row)

        layout.addWidget(self._create_logs_section(), 1)

    def _create_config_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title_row = QHBoxLayout()
        title = QLabel("Mining Configuration")
        title.setObjectName("config_section_title")
        title_row.addWidget(title)

        icon_label = QLabel()
        icon_label.setFixedSize(20, 20)
        renderer = QSvgRenderer(resource_path("gui/images/config_sliders.svg"))
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        icon_label.setPixmap(pixmap)
        title_row.addWidget(icon_label)
        title_row.addStretch()
        layout.addLayout(title_row)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(24)
        metrics_row.addWidget(self._create_metric_box("Mode", self.pool_data.get("mode_label", "-")), 1)
        metrics_row.addWidget(self._create_metric_box("Backend", self.pool_data.get("backend", "-")), 1)
        metrics_row.addWidget(self._create_metric_box("Workers", str(self.pool_data.get("workers", "-"))), 1)
        layout.addLayout(metrics_row)

        pool_row = QHBoxLayout()
        pool_row.setSpacing(24)

        pool_container = QWidget()
        pool_layout = QVBoxLayout(pool_container)
        pool_layout.setContentsMargins(0, 0, 0, 0)
        pool_layout.setSpacing(8)

        pool_label = QLabel("Current Pool")
        pool_label.setObjectName("form_label")
        pool_layout.addWidget(pool_label)

        pool_display = QWidget()
        pool_display.setObjectName("pool_display")
        pool_display.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pool_display.setFixedHeight(48)
        pool_display_layout = QHBoxLayout(pool_display)
        pool_display_layout.setContentsMargins(14, 8, 14, 8)
        pool_display_layout.setSpacing(8)

        pool_icon = QLabel()
        pool_icon.setFixedSize(20, 20)
        renderer = QSvgRenderer(resource_path("gui/images/pool_nodes.svg"))
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        pool_icon.setPixmap(pixmap)
        pool_display_layout.addWidget(pool_icon)

        pool_name = QLabel(self.pool_data.get("name", "Pool"))
        pool_name.setObjectName("pool_display_name")
        pool_display_layout.addWidget(pool_name)
        pool_display_layout.addStretch()
        pool_layout.addWidget(pool_display)
        pool_row.addWidget(pool_container, 1)

        buttons_wrapper = QWidget()
        buttons_layout = QVBoxLayout(buttons_wrapper)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)
        buttons_layout.addStretch()

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)
        self.switch_pool_btn = SecondaryButton("Switch Pool", width=200, height=48)
        self.switch_pool_btn.clicked.connect(self._on_switch_pool)
        buttons_row.addWidget(self.switch_pool_btn)

        self.start_mining_btn = PrimaryButton(
            "Start Mining",
            width=200,
            height=48,
            icon_path=resource_path("gui/images/play.svg"),
        )
        self.start_mining_btn.clicked.connect(self._toggle_mining)
        buttons_row.addWidget(self.start_mining_btn)
        buttons_layout.addLayout(buttons_row)
        pool_row.addWidget(buttons_wrapper)

        layout.addLayout(pool_row)
        return section

    def _create_metric_box(self, label: str, value: str) -> QWidget:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(4)
        title = QLabel(label)
        title.setObjectName("metric_label")
        header.addWidget(title)
        header.addStretch()
        box_layout.addLayout(header)

        value_label = QLabel(value)
        value_label.setObjectName("metric_value")
        value_label.setWordWrap(True)
        box_layout.addWidget(value_label)
        return box

    def _create_miner_stats(self) -> QWidget:
        panel = _create_stats_panel("Miner Stats")
        content = panel.layout().itemAt(1).layout()
        self.tasks_completed_label = _add_stat_row(content, "Tasks Completed", "0")
        content.addWidget(_create_divider())
        self.successful_submissions_label = _add_stat_row(content, "Lease Submissions", "0")
        self.best_score_label = _add_stat_row(content, "Local Best", "-")
        return panel

    def _create_mining_status(self) -> QWidget:
        panel = _create_stats_panel("Mining Status")
        content = panel.layout().itemAt(1).layout()
        self.wallet_status_label = _add_stat_row(content, "Wallet", "Not Connected")

        row = QHBoxLayout()
        row.addWidget(_build_stat_label("Status"))
        row.addStretch()
        self.status_dot, self.mining_status_label, status_container = self._create_status_indicator(
            "Idle",
            "status_dot_idle",
            "status_text_idle",
        )
        row.addWidget(status_container)
        content.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(_build_stat_label("Connection"))
        row.addStretch()
        self.connection_dot, self.connection_status_label, conn_container = self._create_status_indicator(
            "Disconnected",
            "status_dot_disconnected",
            "status_text_disconnected",
        )
        row.addWidget(conn_container)
        content.addLayout(row)

        self.runtime_label = _add_stat_row(content, "Runtime", "0h 0m 0s")
        self.resource_contributed_label = _add_stat_row(content, "Resource Contributed", "0")
        return panel

    def _create_pool_status(self) -> QWidget:
        panel = _create_stats_panel("Pool Status")
        content = panel.layout().itemAt(1).layout()
        if self._is_research_pool():
            self.global_sota_label = _add_stat_row(content, "Metric", self.pool_data.get("metric_label", "-"))
            content.addWidget(_create_divider())
            self.endpoint_label = _add_stat_row(content, "Coordinator", self.pool_data.get("endpoint", "-"))
            self.backend_label = _add_stat_row(content, "Agent Model", self.pool_data.get("agent_model", "Not configured"))
            self.mode_label = _add_stat_row(content, "Rules", self.pool_data.get("competition_mode_label", "-"))
        else:
            self.global_sota_label = _add_stat_row(content, "SOTA", "-")
            content.addWidget(_create_divider())
            self.endpoint_label = _add_stat_row(content, "Endpoint", self.pool_data.get("endpoint", "-"))
            self.backend_label = _add_stat_row(content, "Backend", self.pool_data.get("backend", "-"))
            self.mode_label = _add_stat_row(content, "Mode", self.pool_data.get("mode_label", "-"))
        return panel

    def _create_logs_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("logs_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title = QLabel("Mining Logs")
        title.setObjectName("logs_title")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear Logs")
        clear_btn.setObjectName("clear_logs_button")
        clear_btn.setFixedHeight(32)
        clear_btn.clicked.connect(self._clear_logs)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        self.logs_text.document().setMaximumBlockCount(5000)
        layout.addWidget(self.logs_text)
        return section

    def _clear_logs(self):
        self.logs_text.clear()

    def _append_log(self, message: str):
        self.logs_text.append(message)

    def _create_status_indicator(self, text: str, dot_name: str, text_name: str):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        dot = QWidget()
        dot.setObjectName(dot_name)
        dot.setFixedSize(8, 8)
        dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout.addWidget(dot)

        label = QLabel(text)
        label.setObjectName(text_name)
        layout.addWidget(label)
        return dot, label, container

    @staticmethod
    def _refresh_widget_style(widget: QWidget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _on_switch_pool(self):
        if self.is_mining:
            self._append_log("[pool] stop mining before switching pools")
            return
        self.switch_pool_requested.emit()

    def _is_research_pool(self) -> bool:
        return bool(self.pool_data.get("is_research_pool")) or str(self.pool_data.get("mode")) == RESEARCH_POOL_MODE

    def _research_log_dir(self) -> Path:
        root = Path.cwd() / ".bitsota_agent_logs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _start_research_pool_mining(self) -> None:
        if self.is_mining:
            self._append_log("ERROR: Mining is already running.")
            return
        if not self.main_window:
            self._append_log("ERROR: Main window reference not available.")
            return
        wallet = getattr(self.main_window, "wallet", None)
        hotkey_mnemonic = _ephemeral_hotkey_mnemonic()
        hotkey_address = _ephemeral_hotkey_address(hotkey_mnemonic)
        if wallet is None and not hotkey_mnemonic:
            self._append_log("ERROR: No wallet loaded. Please load a wallet first.")
            return

        settings = _research_runtime_settings()
        coordinator_url = str(self.pool_data.get("coordinator_url") or settings["coordinator_url"] or "").strip()
        provider = str(settings.get("provider") or "").strip()
        agent_command = settings["agent_command"]
        agent_mode = settings["agent_mode"] or "gui_managed"
        llm_base_url = settings["llm_base_url"]
        llm_model = settings["llm_model"]
        if not coordinator_url:
            self._append_log(
                "ERROR: Research coordinator URL is missing. Open Research Setup and choose a network preset."
            )
            return
        if not agent_command:
            if provider == "openai_compatible":
                if not llm_base_url:
                    self._append_log(
                        "ERROR: Research API base URL is missing. Open Research Setup and configure the API endpoint."
                    )
                    return
                if not llm_model:
                    self._append_log(
                        "ERROR: Research API model is missing. Open Research Setup and choose the model to use."
                    )
                    return
            else:
                self._append_log(
                    "ERROR: Research agent is not configured. Open Research Setup and choose Codex CLI, Claude Code, or an OpenAI-compatible API."
                )
                return

        task_id = str(self.pool_data.get("task_id") or "").strip()
        task_slug = str(self.pool_data.get("task_slug") or "").strip()
        resolved_task, launch_error = _resolve_research_launch_task(
            coordinator_url=coordinator_url,
            task_id=task_id,
            task_slug=task_slug,
        )
        if launch_error is not None:
            self._append_log(f"ERROR: {launch_error}")
            return
        if resolved_task is not None:
            resolved_task_id = str(resolved_task.get("id") or "").strip()
            if resolved_task_id:
                task_id = resolved_task_id
                self.pool_data["task_id"] = resolved_task_id

        declared_coldkey = str(getattr(self.main_window, "coldkey_address", "") or "").strip()
        if wallet is not None and declared_coldkey:
            try:
                sync_declared_coldkey_to_pool_backend(
                    wallet=wallet,
                    coldkey_address=declared_coldkey,
                )
                self._append_log("[research-pool] recipient coldkey synced to Pool backend.")
            except Exception as exc:
                self._append_log(f"ERROR: Failed to sync recipient coldkey to Pool backend: {exc}")
                return

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "neurons.research_agent_miner",
            "loop",
            "--coordinator-url",
            coordinator_url,
            "--participation-style",
            "pool",
            "--workspace-root",
            str((Path.cwd() / ".bitsota_agent_workspace").resolve()),
            "--interval-seconds",
            "5",
        ]
        if agent_command:
            cmd.extend(
                [
                    "--agent-command",
                    agent_command,
                    "--agent-mode",
                    agent_mode,
                ]
            )
        else:
            cmd.extend(
                [
                    "--llm-base-url",
                    llm_base_url,
                    "--llm-model",
                    llm_model,
                ]
            )
        if hotkey_mnemonic:
            cmd.extend(["--hotkey-mnemonic", hotkey_mnemonic])
        else:
            cmd.extend(
                [
                    "--wallet-name",
                    str(getattr(wallet, "name", "default")),
                    "--wallet-hotkey",
                    str(getattr(wallet, "hotkey_str", "default")),
                    "--wallet-path",
                    str(getattr(wallet, "path", "~/.bittensor/wallets/")),
                ]
            )
        if task_id:
            cmd.extend(["--task-id", task_id])
        elif task_slug:
            cmd.extend(["--task-slug", task_slug])
        if settings["llm_api_key"] and not agent_command:
            cmd.extend(["--llm-api-key", settings["llm_api_key"]])
        if str(self.pool_data.get("competition_mode") or "") == CompetitionMode.peer_evaluation.value:
            cmd.append("--allow-peer-evaluation")

        log_name = str(self.pool_data.get("task_slug") or self.pool_data.get("id") or "research-pool")
        log_path = self._research_log_dir() / f"{log_name}.log"
        self._research_log_path = log_path
        self._research_log_offset = 0
        log_path.write_text("", encoding="utf-8")
        self._manual_stop_requested = False
        self._last_research_exit_code = None
        self._auto_restart_timer.stop()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            self.miner_process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        finally:
            log_handle.close()

        self.is_mining = True
        self._runtime_started_at = time.time()
        self._update_runtime()
        self._runtime_timer.start(1000)
        self._research_log_timer.start(1000)

        self.start_mining_btn.update_icon("gui/images/stop.svg")
        self.start_mining_btn.update_text("Stop Mining")
        self.start_mining_btn.setObjectName("stop_mining_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)

        self.update_connection_status(True)
        wallet_label = str(getattr(wallet, "name", "Connected")) if wallet is not None else (
            f"Test {hotkey_address[:6]}...{hotkey_address[-4:]}" if hotkey_address else "Test Hotkey"
        )
        self.wallet_status_label.setText(wallet_label)
        self._append_log(f"[research-pool] starting {self.pool_data.get('name')} via coordinator {coordinator_url}")
        try:
            self._append_log(f"[research-pool] agent miner started (pid={int(self.miner_process.pid)}).")
        except Exception:
            pass

    def _poll_research_process(self) -> None:
        if not self._research_log_path:
            return
        try:
            with self._research_log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._research_log_offset)
                chunk = handle.read()
                self._research_log_offset = handle.tell()
        except Exception:
            chunk = ""

        for raw_line in str(chunk or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            self._append_log(line)
            if line.startswith("[research-agent] submitted"):
                self._update_stats(
                    {
                        "tasks_completed": int(getattr(self, "tasks_completed", 0) or 0) + 1,
                        "successful_submissions": int(getattr(self, "successful_submissions", 0) or 0) + 1,
                        "best_score": getattr(self, "best_score", None),
                    }
                )
            elif line.startswith("[research-agent] peer-evaluated"):
                self._update_stats(
                    {
                        "tasks_completed": int(getattr(self, "tasks_completed", 0) or 0) + 1,
                        "successful_submissions": int(getattr(self, "successful_submissions", 0) or 0),
                        "best_score": getattr(self, "best_score", None),
                    }
                )

        if self.miner_process is not None and self.miner_process.poll() is not None:
            self._last_research_exit_code = int(self.miner_process.poll() or 0)
            self._append_log(f"[research-pool] agent miner exited (code={self._last_research_exit_code}).")
            self._on_mining_finished()

    def _start_mining(self):
        if self._is_research_pool():
            self._start_research_pool_mining()
            return
        was_mining = self.is_mining
        super()._start_mining()
        if not was_mining and self.is_mining:
            self._runtime_started_at = time.time()
            self._update_runtime()
            self._runtime_timer.start(1000)

    def _stop_mining(self):
        self._manual_stop_requested = True
        self._auto_restart_timer.stop()
        self._research_log_timer.stop()
        super()._stop_mining()
        self._runtime_timer.stop()

    def _on_mining_finished(self):
        should_restart = bool(
            self._is_research_pool()
            and self._auto_restart_enabled
            and not self._manual_stop_requested
        )
        self._research_log_timer.stop()
        super()._on_mining_finished()
        self._runtime_timer.stop()
        if should_restart:
            delay_s = max(1.0, self._auto_restart_delay_ms / 1000.0)
            self._append_log(
                f"[research-pool] scheduling auto-restart in {delay_s:.1f}s after exit code "
                f"{self._last_research_exit_code if self._last_research_exit_code is not None else 'unknown'}."
            )
            self._auto_restart_timer.start(self._auto_restart_delay_ms)

    def _auto_restart_research_mining(self) -> None:
        if self.is_mining:
            return
        self._append_log("[research-pool] auto-restarting research miner.")
        self._start_research_pool_mining()

    def _update_runtime(self):
        if self._runtime_started_at is None:
            self.runtime_label.setText("0h 0m 0s")
            return
        elapsed = max(0, int(time.time() - self._runtime_started_at))
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.runtime_label.setText(f"{hours}h {minutes}m {seconds}s")

    def _update_stats(self, stats: dict):
        tasks = int(stats.get("tasks_completed", 0) or 0)
        submissions = int(stats.get("successful_submissions", 0) or 0)
        best_score = stats.get("best_score")

        self.tasks_completed = tasks
        self.successful_submissions = submissions
        if best_score is not None:
            self.best_score = best_score

        self.tasks_completed_label.setText(str(tasks))
        self.successful_submissions_label.setText(str(submissions))
        self.resource_contributed_label.setText(str(tasks))
        if self.best_score is None:
            self.best_score_label.setText("-")
        else:
            self.best_score_label.setText(f"{float(self.best_score):.4f}")

    def update_wallet_status(self, wallet_name: str):
        self.wallet_status_label.setText(wallet_name or "Not Connected")

    def update_connection_status(self, connected: bool):
        if connected:
            self.connection_status_label.setText("Connected")
            self.connection_status_label.setObjectName("status_text_connected")
            self.connection_dot.setObjectName("status_dot_connected")
            self.mining_status_label.setText("Running")
            self.mining_status_label.setObjectName("status_text_running")
            self.status_dot.setObjectName("status_dot_running")
        else:
            self.connection_status_label.setText("Disconnected")
            self.connection_status_label.setObjectName("status_text_disconnected")
            self.connection_dot.setObjectName("status_dot_disconnected")
            self.mining_status_label.setText("Idle")
            self.mining_status_label.setObjectName("status_text_idle")
            self.status_dot.setObjectName("status_dot_idle")

        for widget in (
            self.connection_status_label,
            self.connection_dot,
            self.mining_status_label,
            self.status_dot,
        ):
            self._refresh_widget_style(widget)


class PoolMiningScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.pool_cards: list[PoolCard] = []
        self.pool_detail_view: Optional[PoolDetailScreen] = None
        self._research_test_settings = _research_gui_test_settings()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.pool_list_view = self._create_pool_list_view()
        self.stack.addWidget(self.pool_list_view)
        layout.addWidget(self.stack)

    def _create_pool_list_view(self) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(16)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(6)
        title = QLabel("Research Pools")
        title.setObjectName("config_section_title")
        title_box.addWidget(title)

        subtitle = QLabel(
            "Use the guided setup to choose a network preset and agent provider. The app will manage the runtime command for you."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: rgba(21, 0, 73, 0.72);")
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        self.research_setup_btn = SecondaryButton("Research Setup", width=220, height=48)
        self.research_setup_btn.clicked.connect(self._open_research_setup)
        header_layout.addWidget(self.research_setup_btn, 0, Qt.AlignmentFlag.AlignTop)

        container_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        self.pool_grid = QGridLayout()
        self.pool_grid.setHorizontalSpacing(POOL_GRID_SPACING)
        self.pool_grid.setVerticalSpacing(POOL_GRID_SPACING)
        scroll_layout.addLayout(self.pool_grid)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        container_layout.addWidget(scroll)

        self.reload_pools()
        return container

    def _open_research_setup(self) -> None:
        if self.main_window and hasattr(self.main_window, "open_research_setup_modal"):
            saved = bool(self.main_window.open_research_setup_modal())
            if saved:
                self.reload_pools()

    def research_gui_test_enabled(self) -> bool:
        return bool(self._research_test_settings.get("enabled"))

    def reload_pools(self) -> list[dict[str, Any]]:
        pools = _configured_pools()
        self._load_pools(pools)
        return pools

    def refresh_runtime_config(self) -> list[dict[str, Any]]:
        current_pool = dict(self.pool_detail_view.pool_data) if self.pool_detail_view is not None else None
        showing_detail = self.pool_detail_view is not None and self.stack.currentWidget() is self.pool_detail_view
        pools = self.reload_pools()

        if not current_pool:
            return pools

        if self.pool_detail_view is not None and self.pool_detail_view.is_mining:
            return pools

        match = self._find_matching_pool(pools, current_pool)
        if match is None:
            if showing_detail:
                self.show_pool_list()
            return pools

        if showing_detail:
            self._on_join_pool(match)
        return pools

    @staticmethod
    def _find_matching_pool(pools: list[dict[str, Any]], current_pool: dict[str, Any]) -> Optional[dict[str, Any]]:
        current_task_id = str(current_pool.get("task_id") or "").strip()
        current_task_slug = str(current_pool.get("task_slug") or "").strip().lower()
        current_id = str(current_pool.get("id") or "").strip()

        if current_task_id:
            for pool in pools:
                if str(pool.get("task_id") or "").strip() == current_task_id:
                    return dict(pool)

        if current_task_slug:
            for pool in pools:
                if str(pool.get("task_slug") or "").strip().lower() == current_task_slug:
                    return dict(pool)

        if current_id:
            for pool in pools:
                if str(pool.get("id") or "").strip() == current_id:
                    return dict(pool)
        return None

    def _load_pools(self, pools: list[dict]):
        for card in self.pool_cards:
            card.deleteLater()
        self.pool_cards.clear()

        for idx, pool_data in enumerate(pools):
            card = PoolCard(pool_data)
            card.join_clicked.connect(self._on_join_pool)
            row = idx // POOL_GRID_COLUMNS
            col = idx % POOL_GRID_COLUMNS
            self.pool_grid.addWidget(card, row, col)
            self.pool_cards.append(card)

    def _on_join_pool(self, pool_data: dict):
        if self.pool_detail_view is not None:
            self.stack.removeWidget(self.pool_detail_view)
            self.pool_detail_view.deleteLater()

        self.pool_detail_view = PoolDetailScreen(pool_data=pool_data, main_window=self.main_window)
        self.pool_detail_view.switch_pool_requested.connect(self.show_pool_list)
        self.stack.addWidget(self.pool_detail_view)
        self.stack.setCurrentWidget(self.pool_detail_view)

        wallet = getattr(self.main_window, "wallet", None) if self.main_window else None
        if wallet is not None:
            self.pool_detail_view.update_wallet_status(getattr(wallet, "name", "") or "Connected")

    def maybe_autostart_research_test(self) -> bool:
        if not self.research_gui_test_enabled():
            return False
        target_slug = str(self._research_test_settings.get("task_slug") or "").strip()
        if not target_slug:
            return False

        pools = self.reload_pools()
        selected = _select_research_test_pool(pools, target_slug)
        if selected is None:
            return False

        current_slug = (
            str(self.pool_detail_view.pool_data.get("task_slug") or "").strip().lower()
            if self.pool_detail_view is not None
            else ""
        )
        if current_slug != target_slug.lower():
            self._on_join_pool(selected)

        if self.pool_detail_view is None:
            return False

        self.pool_detail_view.configure_research_test(
            auto_restart=bool(self._research_test_settings.get("auto_restart")),
            restart_delay_seconds=int(self._research_test_settings.get("restart_delay_seconds") or 5),
        )
        if not self.pool_detail_view.is_mining:
            self.pool_detail_view._append_log(
                f"[research-pool-test] auto-starting task slug={target_slug}."
            )
            self.pool_detail_view._start_mining()
        return True

    def show_pool_list(self):
        self.stack.setCurrentWidget(self.pool_list_view)

    def update_wallet_status(self, wallet_name: str):
        if self.pool_detail_view is not None:
            self.pool_detail_view.update_wallet_status(wallet_name)


def _create_stats_panel(title_text: str) -> QWidget:
    panel = QWidget()
    panel.setObjectName("stats_box")
    panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(16)

    title = QLabel(title_text)
    title.setObjectName("config_section_title")
    layout.addWidget(title)

    stats_container = QVBoxLayout()
    stats_container.setSpacing(12)
    layout.addLayout(stats_container)
    layout.addStretch()
    return panel


def _add_stat_row(layout: QVBoxLayout, label_text: str, value_text: str) -> QLabel:
    row = QHBoxLayout()
    row.setSpacing(0)
    row.addWidget(_build_stat_label(label_text))
    row.addStretch()

    value = QLabel(str(value_text))
    value.setObjectName("stat_value")
    value.setAlignment(Qt.AlignmentFlag.AlignRight)
    row.addWidget(value)
    layout.addLayout(row)
    return value


def _build_stat_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("stat_label")
    return label


def _create_divider() -> QWidget:
    divider = QWidget()
    divider.setObjectName("stat_divider")
    divider.setFixedHeight(1)
    divider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return divider
