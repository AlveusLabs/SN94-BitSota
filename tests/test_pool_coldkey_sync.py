from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from gui.pool_coldkey_sync import sync_declared_coldkey_to_pool_backend
from gui.screens.mining.pool_mining_screen import PoolDetailScreen
from gui.screens.mining_screen import MiningScreen


class _SignedBytes(bytes):
    def hex(self) -> str:  # pragma: no cover - inherited behavior is fine, kept explicit for clarity.
        return super().hex()


class _DummyHotkey:
    ss58_address = "5HotkeyAddr"

    def sign(self, message):
        self.last_message = message
        return _SignedBytes(b"\xaa\xbb")


class _DummyWallet:
    def __init__(self) -> None:
        self.hotkey = _DummyHotkey()
        self.name = "wallet-1"
        self.hotkey_str = "hk"
        self.path = "~/.bittensor/wallets"


class _DummyResponse:
    def __init__(self, payload=None, status_code: int = 200) -> None:
        self._payload = payload if payload is not None else {"status": "success"}
        self.status_code = status_code
        self.content = b"{}"

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _TimerStub:
    def start(self, *_args, **_kwargs) -> None:
        return None

    def stop(self) -> None:
        return None


class _StyleStub:
    def unpolish(self, *_args, **_kwargs) -> None:
        return None

    def polish(self, *_args, **_kwargs) -> None:
        return None


class _ButtonStub:
    def update_icon(self, *_args, **_kwargs) -> None:
        return None

    def update_text(self, *_args, **_kwargs) -> None:
        return None

    def setObjectName(self, *_args, **_kwargs) -> None:
        return None

    def setStyleSheet(self, *_args, **_kwargs) -> None:
        return None

    def style(self) -> _StyleStub:
        return _StyleStub()


class _LabelStub:
    def setText(self, *_args, **_kwargs) -> None:
        return None


def test_sync_declared_coldkey_to_pool_backend_posts_signed_request() -> None:
    calls: list[dict] = []
    wallet = _DummyWallet()

    class _Session:
        @staticmethod
        def post(url, *, json, headers, timeout):
            calls.append(
                {
                    "url": url,
                    "json": json,
                    "headers": dict(headers),
                    "timeout": timeout,
                }
            )
            return _DummyResponse({"status": "success"})

    payload = sync_declared_coldkey_to_pool_backend(
        wallet=wallet,
        coldkey_address="5DeclaredColdkey",
        cfg=SimpleNamespace(
            pool_endpoint="https://pool.bitsota.example/",
            pool_coldkey_update_endpoint="",
        ),
        session=_Session(),
    )

    assert payload == {"status": "success"}
    assert calls == [
        {
            "url": "https://pool.bitsota.example/coldkey_address/update",
            "json": {"coldkey_address": "5DeclaredColdkey"},
            "headers": {
                "X-Key": "5HotkeyAddr",
                "X-Timestamp": calls[0]["headers"]["X-Timestamp"],
                "X-Signature": "aabb",
            },
            "timeout": 10.0,
        }
    ]
    assert calls[0]["headers"]["X-Timestamp"].isdigit()
    assert (
        wallet.hotkey.last_message
        == f"recipient_coldkey:update:{calls[0]['headers']['X-Timestamp']}:5DeclaredColdkey"
    )


def test_legacy_mining_screen_sends_declared_coldkey_to_pool_backend(monkeypatch) -> None:
    calls: list[dict] = []
    logs: list[str] = []

    def _fake_sync_declared_coldkey_to_pool_backend(*, wallet, coldkey_address, **_kwargs):
        calls.append({"wallet": wallet, "coldkey_address": coldkey_address})
        return {"status": "success"}

    screen = MiningScreen.__new__(MiningScreen)
    screen.main_window = SimpleNamespace(
        wallet=_DummyWallet(),
        coldkey_address="5DeclaredColdkey",
    )
    screen._append_log = logs.append
    screen._no_relay_test_enabled = lambda: False

    monkeypatch.setattr(
        "gui.screens.mining_screen.sync_declared_coldkey_to_pool_backend",
        _fake_sync_declared_coldkey_to_pool_backend,
    )

    assert screen._send_coldkey_address() is True
    assert calls == [
        {
            "wallet": screen.main_window.wallet,
            "coldkey_address": "5DeclaredColdkey",
        }
    ]
    assert logs[-1] == "Recipient coldkey sent to Pool backend successfully"


def test_research_pool_start_syncs_declared_coldkey_before_launch(monkeypatch, tmp_path: Path) -> None:
    sync_calls: list[dict] = []

    def _fake_sync_declared_coldkey_to_pool_backend(*, wallet, coldkey_address, **_kwargs):
        sync_calls.append({"wallet": wallet, "coldkey_address": coldkey_address})
        return {"status": "success"}

    class _DummyProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.sync_declared_coldkey_to_pool_backend",
        _fake_sync_declared_coldkey_to_pool_backend,
    )
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._research_runtime_settings",
        lambda: {
            "coordinator_url": "https://coordinator.bitsota.example",
            "provider": "claude_code",
            "provider_label": "Claude Code",
            "agent_command": "claude code",
            "agent_mode": "gui_managed",
            "llm_base_url": "",
            "llm_model": "",
            "llm_api_key": "",
        },
    )
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._resolve_research_launch_task",
        lambda **_kwargs: (None, None),
    )
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.subprocess.Popen",
        lambda *args, **kwargs: _DummyProcess(),
    )

    screen = PoolDetailScreen.__new__(PoolDetailScreen)
    wallet = _DummyWallet()
    screen.is_mining = False
    screen.main_window = SimpleNamespace(wallet=wallet, coldkey_address="5DeclaredColdkey")
    screen.pool_data = {
        "coordinator_url": "https://coordinator.bitsota.example",
        "name": "BitNet test pool",
        "task_slug": "bitnet-cpu-ternary-kernel",
    }
    screen._append_log = lambda *_args, **_kwargs: None
    screen._research_log_dir = lambda: tmp_path
    screen._auto_restart_timer = _TimerStub()
    screen._research_log_timer = _TimerStub()
    screen._runtime_timer = _TimerStub()
    screen.start_mining_btn = _ButtonStub()
    screen.wallet_status_label = _LabelStub()
    screen.update_connection_status = lambda *_args, **_kwargs: None
    screen._update_runtime = lambda *_args, **_kwargs: None
    screen.miner_process = None
    screen._research_log_path = None
    screen._research_log_offset = 0
    screen._manual_stop_requested = False
    screen._last_research_exit_code = None
    screen._runtime_started_at = None

    screen._start_research_pool_mining()

    assert sync_calls == [
        {
            "wallet": wallet,
            "coldkey_address": "5DeclaredColdkey",
        }
    ]
