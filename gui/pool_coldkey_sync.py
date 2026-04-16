from __future__ import annotations

import time
from typing import Any

import requests

from gui.app_config import get_app_config, resolve_pool_coldkey_update_endpoint


def _coldkey_update_headers(*, wallet: Any, coldkey_address: str) -> dict[str, str]:
    hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if not hotkey:
        raise RuntimeError("Wallet hotkey ss58 address unavailable")
    ts = str(int(time.time()))
    coldkey = str(coldkey_address or "").strip()
    msg = f"recipient_coldkey:update:{ts}:{coldkey}"
    sig = getattr(wallet.hotkey, "sign")(msg).hex()
    return {
        "X-Key": str(hotkey),
        "X-Timestamp": ts,
        "X-Signature": sig,
    }


def sync_declared_coldkey_to_pool_backend(
    *,
    wallet: Any,
    coldkey_address: str,
    timeout_s: float = 10.0,
    cfg: Any = None,
    session: Any = requests,
) -> dict[str, Any]:
    active_cfg = cfg or get_app_config()
    endpoint = resolve_pool_coldkey_update_endpoint(
        explicit=str(getattr(active_cfg, "pool_coldkey_update_endpoint", "") or ""),
        pool_endpoint=str(getattr(active_cfg, "pool_endpoint", "") or ""),
    )
    if not endpoint:
        raise RuntimeError("Pool coldkey update endpoint is not configured.")

    response = session.post(
        f"{endpoint}/coldkey_address/update",
        json={"coldkey_address": str(coldkey_address or "").strip()},
        headers=_coldkey_update_headers(wallet=wallet, coldkey_address=coldkey_address),
        timeout=float(timeout_s),
    )
    response.raise_for_status()

    try:
        payload = response.json() if getattr(response, "content", None) else {}
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        status = str(payload.get("status") or "").strip().lower()
        if status and status not in {"success", "ok", "updated"}:
            raise RuntimeError(f"Pool coldkey update rejected: {payload}")
        return payload

    return {}
