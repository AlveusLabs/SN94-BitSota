import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validator.winner_selection import select_best_result


def _result(miner_hotkey: str, score: float, timestamp: str) -> dict:
    return {
        "miner_hotkey": miner_hotkey,
        "validator_score": score,
        "timestamp": timestamp,
    }


class WinnerSelectionTests(unittest.TestCase):
    def test_select_best_result_picks_unique_top_score(self):
        selection = select_best_result(
            [
                _result("miner-a", 0.70, "2026-03-19T10:10:00+00:00"),
                _result("miner-b", 0.75, "2026-03-19T10:11:00+00:00"),
            ],
            0.70,
        )

        self.assertEqual(selection.result["miner_hotkey"], "miner-b")
        self.assertEqual(selection.best_score, 0.75)
        self.assertEqual(selection.tied_count, 1)
        self.assertEqual(selection.reason, "unique_best")

    def test_select_best_result_keeps_current_local_winner_on_equal_tie(self):
        selection = select_best_result(
            [
                _result("copycat", 0.7515, "2026-03-19T10:20:00+00:00"),
                _result("original", 0.7515, "2026-03-19T10:00:00+00:00"),
            ],
            0.7515,
            current_local_winner_hotkey="original",
        )

        self.assertEqual(selection.result["miner_hotkey"], "original")
        self.assertEqual(selection.tied_count, 2)
        self.assertEqual(selection.reason, "keep_local_winner")

    def test_select_best_result_prefers_latest_relay_frontier_winner_on_cold_start(
        self,
    ):
        selection = select_best_result(
            [
                _result("copycat", 0.7515, "2026-03-19T10:20:00+00:00"),
                _result("original", 0.7515, "2026-03-19T10:00:00+00:00"),
            ],
            0.7515,
            latest_sota_event={"miner_hotkey": "original", "score": 0.7515},
        )

        self.assertEqual(selection.result["miner_hotkey"], "original")
        self.assertEqual(selection.tied_count, 2)
        self.assertEqual(selection.reason, "relay_frontier_winner")

    def test_select_best_result_falls_back_to_oldest_submission_without_other_signal(
        self,
    ):
        selection = select_best_result(
            [
                _result("copycat", 0.7515, "2026-03-19T10:20:00+00:00"),
                _result("original", 0.7515, "2026-03-19T10:00:00+00:00"),
            ],
            0.7515,
        )

        self.assertEqual(selection.result["miner_hotkey"], "original")
        self.assertEqual(selection.tied_count, 2)
        self.assertEqual(selection.reason, "oldest_submission")


if __name__ == "__main__":
    unittest.main()
