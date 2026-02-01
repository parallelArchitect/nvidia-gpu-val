#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/events.py

Event Logging System
Tracks GPU validation runs over time for drift detection.

Storage: ~/.pascalval/runs/{gpu_uuid}/
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


class EventLogger:
    """
    Logs validation runs to track performance over time.

    Storage: ~/.pascalval/runs/{gpu_uuid}/
    Format: {timestamp}.json with full run results
    """

    def __init__(self, gpu_uuid: str):
        self.gpu_uuid = gpu_uuid
        self.runs_dir = Path.home() / ".pascalval" / "runs" / gpu_uuid
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, results: Dict) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "gpu_uuid": self.gpu_uuid,
            "results": results,
        }

        log_file = self.runs_dir / f"{run_id}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        return run_id

    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        run_files = sorted(self.runs_dir.glob("*.json"), reverse=True)
        if limit:
            run_files = run_files[:limit]

        history: List[Dict] = []
        for run_file in run_files:
            try:
                with open(run_file) as f:
                    history.append(json.load(f))
            except Exception:
                continue
        return history

    @staticmethod
    def _extract_sgemm_fields(run: Dict) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        """
        Return (gflops, matrix_n, engine) if present.
        This is defensive because schemas drift.
        """
        sgemm = run.get("results", {}).get("sgemm", {}) or {}

        # Common nesting patterns you’ve had
        result = sgemm.get("result", {}) or {}
        gflops = result.get("gflops", None)

        # Try multiple keys for matrix size
        n = (
            result.get("matrix_n")
            or result.get("n")
            or result.get("size")
            or result.get("matrix_size")
        )

        # Engine may live at sgemm level or inside result
        engine = sgemm.get("engine") or result.get("engine")

        try:
            gflops_f = float(gflops) if gflops is not None else None
        except Exception:
            gflops_f = None

        try:
            n_i = int(n) if n is not None else None
        except Exception:
            n_i = None

        if engine is not None:
            engine = str(engine)

        return gflops_f, n_i, engine

    def get_performance_trend(
        self,
        window_size: int = 30,
        matrix_n: Optional[int] = 4096,
        engine: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Windowed SGEMM trend summary (most recent N runs), filtered to avoid mixing.

        Filters (applied only when the field exists in the log):
          - matrix_n: e.g. 4096
          - engine: "custom" / "cublas" (optional)

        Returns:
          {
            "samples": int,
            "total_runs": int,
            "window_size": int,
            "avg_gflops": float,
            "min_gflops": float,
            "max_gflops": float,
            "first_gflops": float,   # oldest in window
            "latest_gflops": float,  # newest in window
            "change_percent": float, # (latest-first)/first*100
            "trend": "stable" | "improving" | "declining" | "insufficient_data" | "no_data",
          }
        """
        all_history = self.get_history(limit=None)
        history = all_history[: max(1, int(window_size))]

        total_runs = len(all_history)

        if len(history) < 2:
            return {"samples": len(history), "total_runs": total_runs, "window_size": window_size, "trend": "insufficient_data"}

        gflops_values: List[float] = []

        # history is newest->oldest
        for run in history:
            gflops, n, eng = self._extract_sgemm_fields(run)

            if gflops is None or gflops <= 0:
                continue

            # Apply matrix filter ONLY if the log contains size info
            if matrix_n is not None and n is not None and n != int(matrix_n):
                continue

            # Apply engine filter ONLY if engine info exists
            if engine is not None and eng is not None and str(eng).lower() != str(engine).lower():
                continue

            gflops_values.append(float(gflops))

        if len(gflops_values) < 2:
            return {"samples": len(gflops_values), "total_runs": total_runs, "window_size": window_size, "trend": "no_data"}

        # newest first in gflops_values because we iterate newest->oldest
        latest = gflops_values[0]
        first = gflops_values[-1]

        avg = sum(gflops_values) / len(gflops_values)
        mn = min(gflops_values)
        mx = max(gflops_values)

        change_percent = ((latest - first) / first) * 100.0 if first > 0 else 0.0

        # Keep wording professional (no "DEGRADING" shouting)
        if change_percent < -5.0:
            trend = "declining"
        elif change_percent > 5.0:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "samples": len(gflops_values),
            "total_runs": total_runs,
            "window_size": int(window_size),
            "first_gflops": float(first),
            "latest_gflops": float(latest),
            "avg_gflops": float(avg),
            "min_gflops": float(mn),
            "max_gflops": float(mx),
            "trend": trend,
            "change_percent": float(change_percent),
            "matrix_n": matrix_n,
            "engine": engine,
        }


def display_trend(trend: Dict, console=None) -> None:
    """
    Print ONLY the windowed trend summary in the order you specified.
    """
    t = trend.get("trend")
    if t in ("insufficient_data", "no_data"):
        msg = "Not enough comparable SGEMM history for trend summary (need 2+ valid samples)."
        if console:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        return

    samples = int(trend.get("samples", 0))
    avg = trend.get("avg_gflops", None)
    mn = trend.get("min_gflops", None)
    mx = trend.get("max_gflops", None)
    change = float(trend.get("change_percent", 0.0))

    if console:
        from rich.table import Table
        table = Table(title="Performance Trend", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        table.add_row("Runs Analyzed", f"{samples} (most recent)")
        table.add_row("Average", f"{float(avg):.1f} GFLOPS" if avg is not None else "n/a")
        table.add_row("Minimum", f"{float(mn):.1f} GFLOPS" if mn is not None else "n/a")
        table.add_row("Maximum", f"{float(mx):.1f} GFLOPS" if mx is not None else "n/a")
        table.add_row("Recent Change", f"{change:+.1f}%")
        console.print(table)
    else:
        print("Performance Trend")
        print(f"Runs Analyzed: {samples} (most recent)")
        print(f"Average:       {float(avg):.1f} GFLOPS" if avg is not None else "Average:       n/a")
        print(f"Minimum:       {float(mn):.1f} GFLOPS" if mn is not None else "Minimum:       n/a")
        print(f"Maximum:       {float(mx):.1f} GFLOPS" if mx is not None else "Maximum:       n/a")
        print(f"Recent Change: {change:+.1f}%")
