import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os


class MinerMetricsLogger:
    def __init__(self, log_file: str = "miner_metrics.log"):
        self.log_file = log_file
        self.session_id = str(int(time.time()))
        self.session_start = time.time()
        
    def log_session_start(self, task_type: str, engine_type: str):
        """Log the start of a mining session"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "session_start",
            "task_type": task_type,
            "engine_type": engine_type
        }
        self._write_log(entry)
    
    def log_generation(self, generation: int, best_score: float, pop_scores: list, 
                      sota_threshold: float, total_algorithms_evaluated: int):
        """Log evolution generation metrics"""
        valid_scores = [s for s in pop_scores if s != -float('inf')]
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "generation",
            "generation": generation,
            "best_score": best_score,
            "pop_mean": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
            "pop_max": max(valid_scores) if valid_scores else -float('inf'),
            "sota_threshold": sota_threshold,
            "distance_to_sota": sota_threshold - max(valid_scores) if valid_scores else float('inf'),
            "total_algorithms_evaluated": total_algorithms_evaluated,
            "valid_algorithms": len(valid_scores),
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        self._write_log(entry)
    
    def log_sota_breakthrough(self, generation: int, winning_score: float, 
                            sota_threshold: float, num_sota_breakers: int):
        """Log SOTA breakthrough event"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "sota_breakthrough", 
            "generation": generation,
            "winning_score": winning_score,
            "sota_threshold": sota_threshold,
            "score_improvement": winning_score - sota_threshold,
            "num_sota_breakers": num_sota_breakers,
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        self._write_log(entry)
    
    def log_submission(self, submission_result: Dict[str, Any], score: float, generation: int):
        """Log solution submission"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "submission",
            "status": submission_result.get("status"),
            "score": score,
            "generation": generation,
            "runtime_minutes": (time.time() - self.session_start) / 60
        }
        if submission_result.get("status") == "failed":
            entry["error"] = submission_result.get("error")
        
        self._write_log(entry)
    
    def log_session_end(self, total_submissions: int, total_sota_breaks: int):
        """Log end of mining session with summary"""
        runtime_hours = (time.time() - self.session_start) / 3600
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event": "session_end",
            "total_submissions": total_submissions,
            "total_sota_breaks": total_sota_breaks,
            "runtime_hours": runtime_hours,
            "submissions_per_hour": total_submissions / runtime_hours if runtime_hours > 0 else 0
        }
        self._write_log(entry)
    
    def _write_log(self, entry: Dict[str, Any]):
        """Write log entry to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Failed to write metrics log: {e}")