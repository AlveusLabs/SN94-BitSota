from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_prepare_fresh_testnet_tester.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_prepare_fresh_testnet_tester", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_refresh_website_public_artifacts_copies_current_testnet_json(tmp_path: Path) -> None:
    module = _load_module()
    artifacts_dir = tmp_path / "artifacts"
    website_repo = tmp_path / "website"
    (website_repo / "public").mkdir(parents=True)
    for filename in module.PUBLIC_ARTIFACT_FILES:
        (artifacts_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / filename).write_text(f'{{"name":"{filename}","current":true}}\n', encoding="utf-8")
    args = argparse.Namespace(
        artifacts_dir=artifacts_dir,
        website_repo=website_repo,
        skip_website_public_refresh=False,
    )

    result = module._refresh_website_public_artifacts(args)

    assert result["status"] == "green"
    assert len(result["copied"]) == len(module.PUBLIC_ARTIFACT_FILES) * 2
    for filename in module.PUBLIC_ARTIFACT_FILES:
        expected = (artifacts_dir / filename).read_text(encoding="utf-8")
        assert (website_repo / "public" / filename).read_text(encoding="utf-8") == expected
        assert (website_repo / "public" / "base-sota" / filename).read_text(encoding="utf-8") == expected


def test_refresh_website_public_artifacts_can_be_skipped(tmp_path: Path) -> None:
    module = _load_module()
    args = argparse.Namespace(
        artifacts_dir=tmp_path / "artifacts",
        website_repo=tmp_path / "website",
        skip_website_public_refresh=True,
    )

    result = module._refresh_website_public_artifacts(args)

    assert result["status"] == "skipped"
