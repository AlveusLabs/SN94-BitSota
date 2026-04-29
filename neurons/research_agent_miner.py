from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

from miner.research_agent import (
    AgentMinerConfig,
    OpenAICompatibleChatClient,
    ResearchAgentMiner,
    submit_claimed_workspace,
)
from miner.research_competitions import ParticipationStyle, list_builtin_research_competitions
from miner.research_coordinator_client import CoordinatorClient
from miner.wallet_inputs import EphemeralWallet, load_wallet


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BitSota research-agent miner for coordinator-backed competitions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-builtins", help="List the built-in research competition templates.")

    list_tasks = subparsers.add_parser("list-tasks", help="List tasks from a coordinator.")
    list_tasks.add_argument("--coordinator-url", required=True)

    mine_once = subparsers.add_parser("mine-once", help="Run one claim -> submit cycle with an OpenAI-compatible model.")
    _add_agent_args(mine_once)

    peer_eval = subparsers.add_parser("peer-evaluate-once", help="Run one peer-evaluation cycle on a peer_evaluation task.")
    _add_agent_args(peer_eval)

    loop = subparsers.add_parser("loop", help="Continuously mine and optionally peer-evaluate.")
    _add_agent_args(loop)
    loop.add_argument("--cycles", type=int, default=0, help="Number of cycles to run. 0 means forever.")
    loop.add_argument("--interval-seconds", type=float, default=15.0, help="Delay between cycles.")
    loop.add_argument(
        "--allow-peer-evaluation",
        action="store_true",
        help="Try a peer-evaluation pass before mining when possible.",
    )

    submit_workspace = subparsers.add_parser(
        "submit-workspace",
        help="Submit a claimed workspace using repo changes plus a submission.json sidecar.",
    )
    submit_workspace.add_argument("--coordinator-url", required=True)
    submit_workspace.add_argument("--claim-id", required=True)
    submit_workspace.add_argument("--repo-dir", required=True)
    submit_workspace.add_argument("--submission-file", default="submission.json")
    submit_workspace.add_argument("--base-ref", default="")
    submit_workspace.add_argument("--hotkey-mnemonic", default="")
    submit_workspace.add_argument("--wallet-name", default="default")
    submit_workspace.add_argument("--wallet-hotkey", default="default")
    submit_workspace.add_argument("--wallet-path", default="~/.bittensor/wallets/")
    submit_workspace.add_argument("--wallet-file", default="")

    signed_request = subparsers.add_parser(
        "signed-request",
        help="Send a coordinator request with SN94 wallet signing handled by the public client.",
    )
    signed_request.add_argument("--coordinator-url", required=True)
    signed_request.add_argument("--method", required=True)
    signed_request.add_argument("--path", required=True)
    signed_request.add_argument("--body-json", default="")
    signed_request.add_argument("--body-file", default="")
    signed_request.add_argument("--params-json", default="")
    signed_request.add_argument("--unsigned", action="store_true")
    signed_request.add_argument("--hotkey-mnemonic", default="")
    signed_request.add_argument("--wallet-name", default="default")
    signed_request.add_argument("--wallet-hotkey", default="default")
    signed_request.add_argument("--wallet-path", default="~/.bittensor/wallets/")
    signed_request.add_argument("--wallet-file", default="")

    return parser


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--agent-command", default="")
    parser.add_argument("--agent-mode", choices=["gui_managed", "autonomous"], default="gui_managed")
    parser.add_argument("--submission-file", default="submission.json")
    parser.add_argument("--submission-result-file", default="submission_result.json")
    parser.add_argument("--task-id")
    parser.add_argument("--task-slug")
    parser.add_argument(
        "--participation-style",
        choices=[style.value for style in ParticipationStyle],
        default=ParticipationStyle.direct.value,
    )
    parser.add_argument("--workspace-root", default=".bitsota_agent_workspace")
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1800)
    parser.add_argument("--disable-local-execution", action="store_true")
    parser.add_argument("--execution-attempts", type=int, default=2)
    parser.add_argument("--keep-execution-checkout", action="store_true")
    parser.add_argument("--max-execution-log-chars", type=int, default=32000)
    parser.add_argument("--hotkey-mnemonic", default="")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets/")
    parser.add_argument("--wallet-file", default="")


def _load_wallet(args: argparse.Namespace) -> EphemeralWallet | Any:
    return load_wallet(
        hotkey_mnemonic=str(getattr(args, "hotkey_mnemonic", "") or ""),
        wallet_file=str(getattr(args, "wallet_file", "") or ""),
        wallet_name=str(getattr(args, "wallet_name", "default")),
        wallet_hotkey=str(getattr(args, "wallet_hotkey", "default")),
        wallet_path=str(getattr(args, "wallet_path", "~/.bittensor/wallets/")),
    )


def _build_agent(args: argparse.Namespace) -> ResearchAgentMiner:
    wallet = _load_wallet(args)
    coordinator = CoordinatorClient(
        base_url=str(args.coordinator_url),
        wallet=wallet,
    )
    agent_command = str(getattr(args, "agent_command", "") or "").strip()
    llm = None
    if not agent_command:
        llm_base_url = str(getattr(args, "llm_base_url", "") or "").strip()
        llm_model = str(getattr(args, "llm_model", "") or "").strip()
        if not llm_base_url or not llm_model:
            raise ValueError("either --agent-command or both --llm-base-url and --llm-model are required")
        llm = OpenAICompatibleChatClient(
            base_url=llm_base_url,
            model=llm_model,
            api_key=str(args.llm_api_key or ""),
        )
    config = AgentMinerConfig(
        participation_style=ParticipationStyle(str(args.participation_style)),
        workspace_root=Path(str(args.workspace_root)),
        llm_temperature=float(args.llm_temperature),
        llm_max_tokens=int(args.llm_max_tokens),
        execution_enabled=not bool(args.disable_local_execution),
        execution_max_attempts=int(args.execution_attempts),
        keep_execution_checkout=bool(args.keep_execution_checkout),
        max_execution_log_chars=int(args.max_execution_log_chars),
        external_agent_command=agent_command,
        external_agent_mode=str(getattr(args, "agent_mode", "gui_managed")),
        submission_filename=str(getattr(args, "submission_file", "submission.json")),
        submission_result_filename=str(
            getattr(args, "submission_result_file", "submission_result.json")
        ),
    )
    return ResearchAgentMiner(
        coordinator=coordinator,
        llm=llm,
        config=config,
        log=lambda message: print(message),  # noqa: T201
    )


def _list_tasks(args: argparse.Namespace) -> int:
    from substrateinterface import Keypair

    wallet = EphemeralWallet(hotkey=Keypair.create_from_mnemonic(Keypair.generate_mnemonic()))
    coordinator = CoordinatorClient(base_url=str(args.coordinator_url), wallet=wallet)
    tasks = coordinator.list_tasks()
    print(json.dumps(tasks, indent=2))  # noqa: T201
    return 0


def _list_builtins() -> int:
    payload = [
        {
            "slug": template.slug,
            "title": template.title,
            "repository": template.repository,
            "base_ref": template.base_ref,
            "metric_name": template.metric_name,
            "metric_direction": template.metric_direction.value,
            "summary": template.summary,
        }
        for template in list_builtin_research_competitions()
    ]
    print(json.dumps(payload, indent=2))  # noqa: T201
    return 0


def _mine_once(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    result = agent.mine_once(
        task_id=getattr(args, "task_id", None),
        task_slug=getattr(args, "task_slug", None),
        participation_style=ParticipationStyle(str(args.participation_style)),
    )
    print(json.dumps(result, indent=2))  # noqa: T201
    return 0


def _peer_evaluate_once(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    result = agent.peer_evaluate_once(
        task_id=getattr(args, "task_id", None),
        task_slug=getattr(args, "task_slug", None),
    )
    print(json.dumps(result, indent=2))  # noqa: T201
    return 0


def _loop(args: argparse.Namespace) -> int:
    agent = _build_agent(args)
    completed = 0
    while int(args.cycles) == 0 or completed < int(args.cycles):
        ran = False
        if bool(args.allow_peer_evaluation):
            try:
                result = agent.peer_evaluate_once(
                    task_id=getattr(args, "task_id", None),
                    task_slug=getattr(args, "task_slug", None),
                )
                print(json.dumps(result, indent=2))  # noqa: T201
                ran = True
            except Exception as exc:
                print(f"[research-agent] peer-evaluate skipped: {exc}")  # noqa: T201
        if not ran:
            try:
                result = agent.mine_once(
                    task_id=getattr(args, "task_id", None),
                    task_slug=getattr(args, "task_slug", None),
                    participation_style=ParticipationStyle(str(args.participation_style)),
                )
                print(json.dumps(result, indent=2))  # noqa: T201
            except Exception as exc:
                print(f"[research-agent] mining cycle failed: {exc}")  # noqa: T201
        completed += 1
        if int(args.cycles) != 0 and completed >= int(args.cycles):
            break
        time.sleep(max(1.0, float(args.interval_seconds)))
    return 0


def _submit_workspace(args: argparse.Namespace) -> int:
    wallet = _load_wallet(args)
    coordinator = CoordinatorClient(
        base_url=str(args.coordinator_url),
        wallet=wallet,
    )
    result = submit_claimed_workspace(
        coordinator=coordinator,
        claim_id=str(args.claim_id),
        repo_dir=Path(str(args.repo_dir)).expanduser().resolve(),
        submission_file=Path(str(args.submission_file)).expanduser().resolve(),
        default_base_ref=str(getattr(args, "base_ref", "") or ""),
    )
    print(json.dumps(result, indent=2))  # noqa: T201
    return 0


def _load_optional_json(value: str, *, source_name: str) -> Any:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_name} is not valid JSON") from exc


def _load_request_body(args: argparse.Namespace) -> Any:
    body_json = str(getattr(args, "body_json", "") or "").strip()
    body_file = str(getattr(args, "body_file", "") or "").strip()
    if body_json and body_file:
        raise ValueError("use either --body-json or --body-file, not both")
    if body_json:
        return _load_optional_json(body_json, source_name="--body-json")
    if body_file:
        raw = Path(body_file).expanduser().read_text(encoding="utf-8")
        return _load_optional_json(raw, source_name="--body-file")
    return None


def _signed_request(args: argparse.Namespace) -> int:
    wallet = _load_wallet(args)
    coordinator = CoordinatorClient(
        base_url=str(args.coordinator_url),
        wallet=wallet,
    )
    body = _load_request_body(args)
    params = _load_optional_json(str(getattr(args, "params_json", "") or ""), source_name="--params-json")
    if params is not None and not isinstance(params, dict):
        raise ValueError("--params-json must decode to a JSON object")
    response = coordinator.request(
        str(args.method),
        str(args.path),
        body=body,
        params=params,
        sign=not bool(getattr(args, "unsigned", False)),
    )
    try:
        payload = response.json()
    except Exception:
        print(str(response.text or ""))  # noqa: T201
        return 0
    print(json.dumps(payload, indent=2))  # noqa: T201
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list-builtins":
        return _list_builtins()
    if args.command == "list-tasks":
        return _list_tasks(args)
    if args.command == "mine-once":
        return _mine_once(args)
    if args.command == "peer-evaluate-once":
        return _peer_evaluate_once(args)
    if args.command == "loop":
        return _loop(args)
    if args.command == "submit-workspace":
        return _submit_workspace(args)
    if args.command == "signed-request":
        return _signed_request(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
