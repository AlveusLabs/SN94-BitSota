from __future__ import annotations

from pathlib import Path
import sys


COMMUNITY_REPO = Path("/home/mekaneeky/repos/94-agent-community")


def test_snapshot_binding_status_rejects_missing_reward_without_crashing() -> None:
    sys.path.insert(0, str(COMMUNITY_REPO))
    from experiments.base_protocol_design.sota_base_indexer.errors import SotaBaseError
    from experiments.base_protocol_design.sota_base_indexer.store import SotaBaseStore

    store = SotaBaseStore()
    try:
        try:
            store.snapshot_binding_status(coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM", reward_address="")
        except SotaBaseError as exc:
            assert exc.code == "invalid_binding_status_request"
            assert exc.status_code == 422
        else:
            raise AssertionError("expected SotaBaseError")
    finally:
        store.close()


def test_snapshot_binding_status_returns_not_submitted_for_valid_empty_lookup() -> None:
    sys.path.insert(0, str(COMMUNITY_REPO))
    from experiments.base_protocol_design.sota_base_indexer.store import SotaBaseStore

    store = SotaBaseStore()
    try:
        status = store.snapshot_binding_status(
            coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            reward_address="0xE93daE9Bb94aa2f2abA57C7CadEC822b800461Fc",
        )
    finally:
        store.close()

    assert status["schema"] == "sota-snapshot-binding-status/v1"
    assert status["status"] == "not_submitted"
    assert status["accepted"] is False
    assert status["reward_address"] == "0xe93dae9bb94aa2f2aba57c7cadec822b800461fc"
