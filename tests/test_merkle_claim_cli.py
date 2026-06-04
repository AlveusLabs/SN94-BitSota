from __future__ import annotations

from types import SimpleNamespace

from gui import merkle_claim_cli


def test_list_command_prints_claim_packages(monkeypatch, capsys) -> None:
    claim = SimpleNamespace(
        epoch=7,
        index=2,
        identity_address="5MinerHotkey",
        recipient_address="5RecipientColdkey",
        amount_rao=123,
        proof=["0x11", "0x22"],
        root="0x33",
    )

    class _FakeClient:
        def list_claim_packages(self, *, hotkey: str, max_epochs: int):
            assert hotkey == "5MinerHotkey"
            assert max_epochs == 12
            return [claim]

    monkeypatch.setattr(merkle_claim_cli, "_build_client", lambda _args: _FakeClient())

    result = merkle_claim_cli.main(["list", "--hotkey", "5MinerHotkey", "--max-epochs", "12"])

    output = capsys.readouterr().out
    assert result == 0
    assert '"epoch": 7' in output
    assert '"recipient_coldkey": "5RecipientColdkey"' in output


def test_claim_command_submits_first_available_claim(monkeypatch, capsys) -> None:
    claim = SimpleNamespace(epoch=11, index=0)
    calls: list[dict] = []

    class _FakeClient:
        def list_claim_packages(self, *, hotkey: str, max_epochs: int):
            assert hotkey == "5MinerHotkey"
            return [claim]

        def submit_claim(self, *, signer, claim, transfer: bool):
            calls.append(
                {
                    "signer": signer,
                    "claim": claim,
                    "transfer": transfer,
                }
            )
            return {"ok": True, "epoch": 11, "index": 0}

    monkeypatch.setattr(merkle_claim_cli, "_build_client", lambda _args: _FakeClient())
    monkeypatch.setattr(
        merkle_claim_cli,
        "load_signer_keypair",
        lambda **_kwargs: SimpleNamespace(ss58_address="5Signer"),
    )

    result = merkle_claim_cli.main(
        [
            "claim",
            "--hotkey",
            "5MinerHotkey",
            "--signer-mnemonic",
            "example mnemonic",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert calls[0]["transfer"] is True
    assert calls[0]["claim"] is claim
    assert '"ok": true' in output
