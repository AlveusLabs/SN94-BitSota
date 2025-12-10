import json
import threading
import time
import os
from datetime import datetime
from pathlib import Path
import requests

from web3 import Web3
from web3.exceptions import ContractLogicError
from substrateinterface import Keypair


class ForcePostHTTPProvider(Web3.HTTPProvider):
    """Force JSON POST for nodes that reject GETs. Also uses a monotonically
    increasing id to avoid some proxy caches mixing responses."""

    _req_id = 0

    def make_request(self, method, params):
        ForcePostHTTPProvider._req_id += 1
        response = requests.post(
            self.endpoint_uri,
            json={
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": ForcePostHTTPProvider._req_id,
            },
            headers={"Content-Type": "application/json"},
            timeout=self._request_kwargs.get("timeout", 10),
        )
        response.raise_for_status()
        return response.json()


class ContractManager:
    def __init__(
        self,
        rpc_url,
        contract_address,
        abi,
        bt_wallet=None,
        evm_private_key=None,
        evm_key_path=None,
        no_wallet=False,
    ):
        self.w3 = Web3(ForcePostHTTPProvider(rpc_url))
        self.wallet = bt_wallet
        self.no_wallet = no_wallet
        self.eth_account = None

        if not self.no_wallet:
            if self.wallet is None:
                raise ValueError("bt_wallet must be provided if no_wallet is False")

            # ---- EVM key load (env → explicit path → wallet h160 path) ----
            if evm_private_key is None:
                if evm_key_path:
                    evm_file = Path(evm_key_path).expanduser()
                    if not evm_file.exists():
                        raise FileNotFoundError(
                            f"EVM key file not found at custom path: {evm_key_path}"
                        )
                    evm_private_key = json.loads(evm_file.read_text())["private_key"]
                else:
                    evm_private_key = os.environ.get("EVM_PRIVATE_KEY")
                    if not evm_private_key:
                        evm_file = (
                            Path(self.wallet.path).expanduser()
                            / self.wallet.name
                            / "h160"
                            / self.wallet.hotkey_str
                        )
                        if not evm_file.exists():
                            raise FileNotFoundError(
                                f"EVM key file not found at {evm_file}. "
                                "Set EVM_PRIVATE_KEY, pass evm_private_key, or create the h160 key file."
                            )
                        evm_private_key = json.loads(evm_file.read_text())[
                            "private_key"
                        ]

            if not str(evm_private_key).startswith("0x"):
                evm_private_key = "0x" + str(evm_private_key)

            self.eth_account = self.w3.eth.account.from_key(evm_private_key)

        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)

        # sota caching
        self._sota_cache = {
            "value": None,
            "timestamp": 0,
            "lock": threading.Lock(),
            "fetch_in_progress": False,
            "last_error": None,
        }
        self.sota_cache_duration = 300  # 5 minutes

    def _fee_fields(self):
        """Return a dict of gas fee fields appropriate for the node (EIP-1559 or legacy)."""
        latest = self.w3.eth.get_block("latest")
        base = latest.get("baseFeePerGas", None)
        if base is not None:
            # EIP-1559
            try:
                priority = self.w3.eth.max_priority_fee  # web3.py v6 property
                if priority is None:
                    raise ValueError
            except Exception:
                priority = 1_000_000_000  # 1 gwei fallback
            max_fee = base * 2 + priority
            return {
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": priority,
                "type": 2,
            }
        # Legacy
        return {"gasPrice": self.w3.eth.gas_price}

    def _tx_envelope(self):
        return {
            "from": self.eth_account.address,
            "nonce": self.w3.eth.get_transaction_count(
                self.eth_account.address, block_identifier="pending"
            ),
            "chainId": self.w3.eth.chain_id,
            **self._fee_fields(),
        }

    def ss58_to_bytes32(self, ss58_address: str) -> bytes:
        """Convert SS58 address to raw 32-byte pubkey (as required by the precompile/contract)."""
        kp = Keypair(ss58_address=ss58_address)
        return kp.public_key  # 32 bytes

    def _already_voted_for(self, recipient_bytes32, scaled_score) -> bool:
        """True if this EOA already voted in the *current round* for the same params."""
        if self.no_wallet:
            return False
        try:
            cur_recipient, cur_score, vote_count, _ = (
                self.contract.functions.getVotingStatus().call()
            )
            if vote_count == 0:
                return False
            if cur_recipient == recipient_bytes32 and int(cur_score) == int(
                scaled_score
            ):
                is_trustee, has_voted, *_ = self.contract.functions.checkAddressStatus(
                    self.eth_account.address
                ).call()
                # ignore is_trustee here; we only care if we've voted this round
                return bool(has_voted)
            return False
        except Exception:
            # Don’t block on RPC quirks; allow the tx to try.
            return False

    def submit_contract_entry(self, recipient_ss58_address: str, new_score: float, verbose=False):
        """Cast a vote for (recipient, score). Also best-effort trigger attemptPayout afterwards."""
        if self.no_wallet:
            raise Exception("Cannot submit entry in no_wallet mode")
        # Optional state printouts (only ABI-listed views)
        if verbose:
            try:
                print("chain_id:", self.w3.eth.chain_id)
                print("sender:", self.eth_account.address)
                print(
                    "is trustee?:",
                    self.contract.functions.isTrustee(self.eth_account.address).call(),
                )
                print(
                    "paused?:",
                    self.contract.functions.paused().call(),
                    "burned?:",
                    self.contract.functions.burned().call(),
                )
                print(
                    "contract balance:",
                    self.w3.from_wei(self.contract.functions.getBalance().call(), "ether"),
                )
                print("voting status:", self.contract.functions.getVotingStatus().call())
            except Exception:
                pass

        # Params
        recipient_bytes32 = self.ss58_to_bytes32(recipient_ss58_address)
        scaled_score = int(new_score * 10**18)

        # Prevent obvious double-vote from same EOA in same round/params
        if self._already_voted_for(recipient_bytes32, scaled_score):
            raise RuntimeError("already voted this round for the same recipient/score")

        # Preflight (surface revert reasons like not trustee / paused / etc.)
        try:
            self.contract.functions.releaseReward(recipient_bytes32, scaled_score).call(
                {"from": self.eth_account.address}
            )
        except ContractLogicError as e:
            raise RuntimeError(f"would revert: {e}") from None

        # Build, estimate gas, sign, send
        tx_fields = self._tx_envelope()
        try:
            gas_est = self.contract.functions.releaseReward(
                recipient_bytes32, scaled_score
            ).estimate_gas({"from": self.eth_account.address})
            tx_fields["gas"] = int(gas_est * 1.20)
        except ContractLogicError as e:
            # If estimation reverts but preflight passed, give a sane ceiling
            tx_fields["gas"] = 300_000

        tx = self.contract.functions.releaseReward(
            recipient_bytes32, scaled_score
        ).build_transaction(tx_fields)
        signed = self.w3.eth.account.sign_transaction(tx, self.eth_account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

        if rcpt.status != 1:
            # Try to extract a reason deterministically
            try:
                self.contract.functions.releaseReward(
                    recipient_bytes32, scaled_score
                ).call({"from": self.eth_account.address})
            except ContractLogicError as e:
                raise RuntimeError(f"reverted on-chain: {e}") from None
            raise RuntimeError("tx failed without explicit reason")

        # Contract auto-distributes rewards when 2/3 votes are reached
        # No need for separate attemptPayout() call

        return tx_hash.hex()

    def attempt_payout(self):
        """Manually trigger payout (contract will enforce threshold/funds)."""
        if self.no_wallet:
            raise Exception("Cannot attempt payout in no_wallet mode")
        # Preflight: allows clean revert reasons (optional)
        self.contract.functions.attemptPayout().call({"from": self.eth_account.address})

        tx_fields = self._tx_envelope()
        try:
            gas_est = self.contract.functions.attemptPayout().estimate_gas(
                {"from": self.eth_account.address}
            )
            tx_fields["gas"] = int(gas_est * 1.20)
        except ContractLogicError:
            tx_fields["gas"] = 250_000

        tx = self.contract.functions.attemptPayout().build_transaction(tx_fields)
        signed = self.w3.eth.account.sign_transaction(tx, self.eth_account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

        if rcpt.status != 1:
            # Try to surface reason
            try:
                self.contract.functions.attemptPayout().call(
                    {"from": self.eth_account.address}
                )
            except ContractLogicError as e:
                raise RuntimeError(f"payout reverted: {e}") from None
            raise RuntimeError("payout tx failed without explicit reason")

        return tx_hash.hex()

    def get_current_sota_threshold(self, force_refresh=False):
        """Cached latestScore (scaled 1e18 → float)."""
        with self._sota_cache["lock"]:
            current_time = time.time()
            if (
                not force_refresh
                and self._sota_cache["value"] is not None
                and current_time - self._sota_cache["timestamp"]
                < self.sota_cache_duration
            ):
                return self._sota_cache["value"]
            if self._sota_cache["fetch_in_progress"]:
                time.sleep(0.1)
                if self._sota_cache["value"] is not None:
                    return self._sota_cache["value"]
                raise Exception("Sota fetch in progress, no cached value available")
            self._sota_cache["fetch_in_progress"] = True

        try:
            scaled_value = self.contract.functions.latestScore().call()
            new_value = scaled_value / 10**18
            with self._sota_cache["lock"]:
                self._sota_cache["value"] = new_value
                self._sota_cache["timestamp"] = current_time
                self._sota_cache["last_error"] = None
                self._sota_cache["fetch_in_progress"] = False
            return new_value
        except Exception as e:
            with self._sota_cache["lock"]:
                self._sota_cache["last_error"] = str(e)
                self._sota_cache["fetch_in_progress"] = False
                if self._sota_cache["value"] is not None:
                    print(f"Using stale sota value due to error: {e}")
                    return self._sota_cache["value"]
            raise e

    def get_sota_cache_status(self):
        with self._sota_cache["lock"]:
            age = (
                time.time() - self._sota_cache["timestamp"]
                if self._sota_cache["timestamp"]
                else None
            )
            return {
                "value": self._sota_cache["value"],
                "age_seconds": age,
                "is_stale": age > self.sota_cache_duration if age else True,
                "last_updated": (
                    datetime.fromtimestamp(self._sota_cache["timestamp"]).isoformat()
                    if self._sota_cache["timestamp"]
                    else None
                ),
                "last_error": self._sota_cache["last_error"],
                "fetch_in_progress": self._sota_cache["fetch_in_progress"],
            }
