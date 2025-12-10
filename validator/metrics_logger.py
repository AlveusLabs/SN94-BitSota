import json
import time
from datetime import datetime
from typing import Dict, Any, List
import os


class ValidatorMetricsLogger:
    def __init__(self, log_file: str = "validator_metrics.log"):
        self.log_file = log_file
        self.session_id = str(int(time.time()))
        self.session_start = time.time()
        self.miners_seen = set()
        self.miners_pushed_sota = set()
        
    def log_session_start(self):
        """Log the start of a validation session"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "session_start"
        }
        self._write_log(entry)
    
    def log_evaluation_batch(self, num_results: int, sota_threshold: float, 
                           evaluation_start_time: float):
        """Log batch of evaluation results"""
        evaluation_time = time.time() - evaluation_start_time
        avg_time_per_eval = evaluation_time / num_results if num_results > 0 else 0
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "evaluation_batch",
            "num_evaluations": num_results,
            "total_evaluation_time_seconds": evaluation_time,
            "avg_time_per_eval_seconds": avg_time_per_eval,
            "sota_threshold": sota_threshold,
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        self._write_log(entry)
    
    def log_miner_result(self, miner_hotkey: str, miner_score: float, 
                        validator_score: float, sota_threshold: float, 
                        result_status: str, pushed_sota: bool = False):
        """Log individual miner result processing"""
        self.miners_seen.add(miner_hotkey)
        if pushed_sota:
            self.miners_pushed_sota.add(miner_hotkey)
            
        delta_improvement = validator_score - sota_threshold if validator_score > sota_threshold else 0
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "miner_result",
            "miner_hotkey": miner_hotkey[:8] + "...",  # Truncated for privacy
            "miner_score": miner_score,
            "validator_score": validator_score,
            "sota_threshold": sota_threshold,
            "delta_improvement": delta_improvement,
            "result_status": result_status,  # "passed", "failed_sota", "failed_validation", "blacklisted"
            "pushed_sota": pushed_sota,
            "unique_miners_seen": len(self.miners_seen),
            "miners_pushed_sota_count": len(self.miners_pushed_sota),
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        self._write_log(entry)
    
    def log_sota_update(self, old_sota: float, new_sota: float, miner_hotkey: str):
        """Log SOTA threshold updates"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "sota_update",
            "old_sota": old_sota,
            "new_sota": new_sota,
            "improvement": new_sota - old_sota,
            "triggering_miner": miner_hotkey[:8] + "...",
            "unique_miners_seen": len(self.miners_seen),
            "miners_pushed_sota_count": len(self.miners_pushed_sota),
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        self._write_log(entry)
    
    def log_contract_submission(self, miner_hotkey: str, score: float, tx_hash: str, 
                              submission_time: float):
        """Log contract submission"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "contract_submission",
            "miner_hotkey": miner_hotkey[:8] + "...",
            "score": score,
            "tx_hash": tx_hash,
            "submission_time_seconds": submission_time,
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        self._write_log(entry)
    
    def log_periodic_summary(self):
        """Log periodic summary of validator performance"""
        runtime_hours = (time.time() - self.session_start) / 3600
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "periodic_summary",
            "unique_miners_seen": len(self.miners_seen),
            "miners_pushed_sota_count": len(self.miners_pushed_sota),
            "sota_success_rate": len(self.miners_pushed_sota) / len(self.miners_seen) if self.miners_seen else 0,
            "runtime_hours": runtime_hours
        }
        self._write_log(entry)
    
    def log_session_end(self):
        """Log end of validation session"""
        runtime_hours = (time.time() - self.session_start) / 3600
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "session_end",
            "unique_miners_seen": len(self.miners_seen),
            "miners_pushed_sota_count": len(self.miners_pushed_sota),
            "final_sota_success_rate": len(self.miners_pushed_sota) / len(self.miners_seen) if self.miners_seen else 0,
            "runtime_hours": runtime_hours
        }
        self._write_log(entry)
    
    def _write_log(self, entry: Dict[str, Any]):
        """Write log entry to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Failed to write validator metrics log: {e}")