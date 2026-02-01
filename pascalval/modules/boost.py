#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/modules/boost.py

Boost / thermal / power sampling (lightweight, deterministic).

What this module does:
  - Samples per-GPU telemetry via NVML over a short window
  - Produces cooling evidence inputs (temp slope, clock stability, power range)
  - Emits a tiny high-signal event stream (start/end only by default)
  - Optional progress output to stderr (safe with --json)

No technician conclusions. Facts only.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

import pynvml  # provided by nvidia-ml-py


def _p(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, file=sys.stderr, flush=True)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _decode(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _event_stream_or_none(gpu_uuid: str):
    """
    Use pascalval.events.EventStream if present; otherwise a tiny compatible stream.
    """
    try:
        from pascalval.events import EventStream  # type: ignore
        return EventStream(gpu_uuid=gpu_uuid, module="boost")
    except Exception:
        class _MiniStream:
            def __init__(self, gpu_uuid: str):
                self.gpu_uuid = gpu_uuid
                self._items: List[Dict[str, Any]] = []
                self._t0 = time.monotonic()

            def record(self, event: str, ctx: Dict[str, Any]):
                self._items.append({
                    "module": "boost",
                    "gpu_uuid": self.gpu_uuid,
                    "event": event,
                    "ctx": ctx,
                    "t_s": float(time.monotonic() - self._t0),
                })

            def to_list(self) -> List[Dict[str, Any]]:
                return list(self._items)

        return _MiniStream(gpu_uuid)


def _lin_slope_per_min(ts: List[float], ys: List[float]) -> Optional[float]:
    # Simple least-squares slope, returns °C/min
    if len(ts) < 2 or len(ts) != len(ys):
        return None
    n = len(ts)
    t_mean = sum(ts) / n
    y_mean = sum(ys) / n
    num = sum((ts[i] - t_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((ts[i] - t_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    slope_per_s = num / den
    return float(slope_per_s * 60.0)


def run_all(
    duration_s: float = 10.0,
    interval_s: float = 0.5,
    event_sink: Optional[List[Dict[str, Any]]] = None,
    max_events_per_gpu: int = 500,
    progress: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "schema": "boost.v0",
        "gpus": {
          "<GPU-UUID>": {
             "duration_s": ...,
             "interval_s": ...,
             "samples": [...],
             "summary": {...}
          }
        }
      }
    """
    if duration_s <= 0:
        duration_s = 1.0
    if interval_s <= 0:
        interval_s = 0.5

    pynvml.nvmlInit()
    try:
        out: Dict[str, Any] = {"schema": "boost.v0", "gpus": {}}

        dev_count = _safe(lambda: pynvml.nvmlDeviceGetCount(), 0) or 0
        total_expected = int(max(1, round(duration_s / interval_s)))

        _p(f"[pascalval] boost: sampling {duration_s:.0f}s @ {interval_s:.2f}s ({total_expected} ticks)", progress)

        # Pre-open handles once
        handles = []
        for i in range(int(dev_count)):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = _decode(_safe(lambda: pynvml.nvmlDeviceGetUUID(h), f"GPU-INDEX-{i}"))
            name = _decode(_safe(lambda: pynvml.nvmlDeviceGetName(h), ""))
            handles.append((i, h, uuid, name))

            stream = _event_stream_or_none(uuid)
            stream.record("BOOST_START", {"duration_s": duration_s, "interval_s": interval_s, "device_index": i, "name": name})
            if event_sink is not None:
                items = stream.to_list()
                if max_events_per_gpu and len(items) > max_events_per_gpu:
                    items = items[:max_events_per_gpu]
                event_sink.extend(items)

            out["gpus"][uuid] = {
                "device_index": i,
                "name": name,
                "duration_s": duration_s,
                "interval_s": interval_s,
                "samples": [],
                "summary": {},
            }

        t0 = time.monotonic()
        last_report = t0

        # Sample loop
        tick = 0
        while True:
            now = time.monotonic()
            t_s = now - t0
            if t_s >= duration_s:
                break

            for (i, h, uuid, _name) in handles:
                temp_c = _safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU), None)
                power_mw = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h), None)
                power_w = (float(power_mw) / 1000.0) if power_mw is not None else None
                sm_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), None)
                mem_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM), None)
                util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h), None)
                util_gpu = int(getattr(util, "gpu", 0)) if util is not None else None
                util_mem = int(getattr(util, "memory", 0)) if util is not None else None

                out["gpus"][uuid]["samples"].append({
                    "t_s": float(t_s),
                    "temp_c": temp_c,
                    "power_w": power_w,
                    "sm_clock_mhz": sm_clock,
                    "mem_clock_mhz": mem_clock,
                    "util_gpu_pct": util_gpu,
                    "util_mem_pct": util_mem,
                })

            tick += 1

            # Lightweight progress heartbeat (stderr only)
            if progress and (now - last_report) >= 5.0:
                last_report = now
                _p(f"[pascalval] boost: tick {tick}/{total_expected}", True)

            # Sleep to next interval
            time.sleep(interval_s)

        # Summaries
        for (i, h, uuid, name) in handles:
            samples = out["gpus"][uuid]["samples"]
            ts = [float(s["t_s"]) for s in samples if s.get("t_s") is not None]
            temps = [float(s["temp_c"]) for s in samples if s.get("temp_c") is not None]
            pows = [float(s["power_w"]) for s in samples if s.get("power_w") is not None]
            clocks = [int(s["sm_clock_mhz"]) for s in samples if s.get("sm_clock_mhz") is not None]

            temp_slope = _lin_slope_per_min(ts, temps) if (ts and temps and len(ts) == len(temps)) else None

            summary = {
                "temp_c_min": min(temps) if temps else None,
                "temp_c_max": max(temps) if temps else None,
                "power_w_min": min(pows) if pows else None,
                "power_w_max": max(pows) if pows else None,
                "sm_clock_mhz_min": min(clocks) if clocks else None,
                "sm_clock_mhz_max": max(clocks) if clocks else None,
                "temp_slope_c_per_min": temp_slope,
            }
            out["gpus"][uuid]["summary"] = summary

            stream = _event_stream_or_none(uuid)
            stream.record("BOOST_END", {
                "device_index": i,
                "name": name,
                "temp_c_min": summary["temp_c_min"],
                "temp_c_max": summary["temp_c_max"],
                "sm_clock_mhz_min": summary["sm_clock_mhz_min"],
                "sm_clock_mhz_max": summary["sm_clock_mhz_max"],
                "temp_slope_c_per_min": summary["temp_slope_c_per_min"],
            })
            if event_sink is not None:
                items = stream.to_list()
                if max_events_per_gpu and len(items) > max_events_per_gpu:
                    items = items[:max_events_per_gpu]
                event_sink.extend(items)

        _p("[pascalval] boost: done", progress)
        return out
    finally:
        _safe(lambda: pynvml.nvmlShutdown(), None)
