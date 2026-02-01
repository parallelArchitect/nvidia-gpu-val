#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: run_validation.py

Description:
  Canonical orchestrator for PascalVal.

  Policy:
    - Gate order: PCIe → Memory → Compute
    - Early-exit: FAIL/ERROR stops immediately (no wasted compute)
    - Modules are execution-only (return dicts only)
    - Orchestrator owns: policy, routing, UI, logging
    - cuBLAS is the portable fallback engine for non-GP104 or when custom engine isn't present.

Logging:
  results/json/<run_id>/run.json
  results/json/<run_id>/events.jsonl

Truth invariant (fixed):
  - If a compute route exists, it is persisted at: results.route
  - Final verdict is persisted at: results.final
  - This eliminates route/final nulls in run.json when the run completes.

Usage:
  PYTHONPATH=. python3 run_validation.py
  PYTHONPATH=. python3 run_validation.py --compute-engine auto
  PYTHONPATH=. python3 run_validation.py --compute-engine custom
  PYTHONPATH=. python3 run_validation.py --compute-engine cublas

References:
  NVML:   https://developer.nvidia.com/management-library-nvml
  cuBLAS: https://docs.nvidia.com/cuda/cublas/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# Repo root
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Optional Rich UI
# -----------------------------------------------------------------------------
RICH = False
console = None
Table = None
Panel = None

try:
    from rich.console import Console  # type: ignore
    from rich.table import Table as _Table  # type: ignore
    from rich.panel import Panel as _Panel  # type: ignore

    console = Console()
    Table = _Table
    Panel = _Panel
    RICH = True
except Exception:
    RICH = False
    console = None
    Table = None
    Panel = None


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def box(title: str) -> None:
    if RICH and console and Panel:
        console.print(Panel.fit(title, style="green"))
        return
    print(f"\n[{title}]\n")


def _print_line(s: str) -> None:
    # Always flush so Phase-1 message appears BEFORE sudo prompt emitted by PCIe module.
    print(s, flush=True)


# -----------------------------------------------------------------------------
# JSON paths
# -----------------------------------------------------------------------------
@dataclass
class RunPaths:
    run_id: str
    root: Path
    run_json: Path
    events_jsonl: Path


def _paths_for_run(run_id: str) -> RunPaths:
    root = Path("results") / "json" / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=run_id, root=root, run_json=root / "run.json", events_jsonl=root / "events.jsonl")


# -----------------------------------------------------------------------------
# Event stream (JSONL)
# -----------------------------------------------------------------------------
class EventSink:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, module: str, event: str, ctx: Dict[str, Any]) -> None:
        rec = {"ts": utc_ts(), "module": module, "event": event, "ctx": ctx}
        _ensure_results_schema(rec)
        self._fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Verdict helpers
# -----------------------------------------------------------------------------
def _extract_verdict(d: Any) -> str:
    if isinstance(d, dict):
        for k in ("verdict", "status"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
    return "UNKNOWN"


def _is_failish(v: str) -> bool:
    v = (v or "").upper()
    return v in ("FAIL", "FAILED", "ERROR")


# -----------------------------------------------------------------------------
# Sudo status
# -----------------------------------------------------------------------------
def _sudo_status() -> bool:
    """True if sudo credentials are already cached (no prompt needed)."""
    try:
        r = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return r.returncode == 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# NVML snapshot + sampling
# -----------------------------------------------------------------------------
def _nvml_snapshot(device_index: int = 0) -> Optional[Dict[str, Any]]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        clk = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        p_mw = float(pynvml.nvmlDeviceGetPowerUsage(h))
        pynvml.nvmlShutdown()
        return {"clock_sm_mhz": clk, "temp_c": temp, "power_w": p_mw / 1000.0}
    except Exception:
        return None


class _DuringSampler:
    """
    Samples NVML in a background thread while SGEMM runs.
    Records max clock/temp/power observed during the window.
    """

    def __init__(self, device_index: int = 0, interval_s: float = 0.10):
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.ok = False
        self.err: Optional[str] = None
        self._max = {"clock_sm_mhz": 0.0, "temp_c": 0.0, "power_w": 0.0}

    def _loop(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.ok = True

            while not self._stop.is_set():
                clk = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                p_mw = float(pynvml.nvmlDeviceGetPowerUsage(h))
                pw = p_mw / 1000.0

                if clk > self._max["clock_sm_mhz"]:
                    self._max["clock_sm_mhz"] = clk
                if temp > self._max["temp_c"]:
                    self._max["temp_c"] = temp
                if pw > self._max["power_w"]:
                    self._max["power_w"] = pw

                time.sleep(self.interval_s)

            pynvml.nvmlShutdown()
        except Exception as e:
            self.err = repr(e)
            self.ok = False

    def start(self) -> None:
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> Optional[Dict[str, Any]]:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
        return self._max if self.ok else None


# -----------------------------------------------------------------------------
# Compute: theoretical + verdict policy
# -----------------------------------------------------------------------------
def _fp32_theoretical_gflops(caps: Dict[str, Any], fallback_mhz: Optional[float]) -> Optional[float]:
    """
    FP32 peak (GFLOP/s) ~= cuda_cores_total * SM_clock(GHz) * 2 (FMA).
    """
    cores = caps.get("cuda_cores_total") or caps.get("cuda_cores")
    if cores is None:
        return None
    try:
        cores_f = float(cores)
    except Exception:
        return None

    mhz = caps.get("sm_clock_mhz_max") or caps.get("clock_sm_mhz_max") or caps.get("clock_sm_mhz") or fallback_mhz
    if mhz is None:
        return None
    try:
        ghz = float(mhz) / 1000.0
    except Exception:
        return None
    return cores_f * ghz * 2.0


def _compute_verdict_from_ratio(ratio: Optional[float]) -> str:
    if ratio is None:
        return "UNKNOWN"
    if ratio >= 0.70:
        return "OK"
    if ratio >= 0.65:
        return "WARN"
    return "FAIL"


# -----------------------------------------------------------------------------
# Caps
# -----------------------------------------------------------------------------
def _get_caps() -> Dict[str, Any]:
    import pascalval.caps as caps_mod  # type: ignore

    candidates = [
        "get_caps",
        "probe_caps",
        "collect_caps",
        "caps",
        "read_caps",
        "hardware_caps",
        "detect_caps",
        "gather_caps",
    ]
    for name in candidates:
        fn = getattr(caps_mod, name, None)
        if callable(fn):
            out = fn()
            if isinstance(out, dict):
                return out
            raise RuntimeError(f"caps function '{name}()' returned non-dict: {type(out)}")

    for name in ("CAPS", "caps", "gpu_caps", "GPU_CAPS"):
        obj = getattr(caps_mod, name, None)
        if isinstance(obj, dict):
            return obj

    raise RuntimeError("No caps entrypoint found in pascalval.caps")


def _gpu_uuid_from_caps(caps: Dict[str, Any]) -> str:
    v = caps.get("uuid") or caps.get("gpu_uuid") or caps.get("nvml_uuid")
    return v.strip() if isinstance(v, str) and v.strip() else "UNKNOWN"


# -----------------------------------------------------------------------------
# Compute routing (cuBLAS is portable fallback)
# -----------------------------------------------------------------------------
def _compute_route(caps: Dict[str, Any], compute_engine: str) -> Dict[str, Any]:
    compute_engine = (compute_engine or "auto").strip().lower()

    silicon_id = caps.get("silicon_id")
    cc = str(caps.get("compute_capability"))
    pci_device_id = caps.get("pci_device_id")

    custom_bin = Path("public_release") / "pascalval_sgemm_public"
    custom_present = custom_bin.exists()

    # manual override
    if compute_engine in ("custom", "cublas"):
        return {
            "mode": "manual",
            "selected": compute_engine,
            "reason": "manual_override",
            "ctx": {
                "silicon_id": silicon_id,
                "compute_capability": cc,
                "pci_device_id": pci_device_id,
                "custom_binary_present": bool(custom_present),
                "requested_engine": compute_engine,
            },
        }

    # strict GP104 policy: use custom if present, else cuBLAS
    if silicon_id == "GP104" and cc == "6.1" and custom_present:
        return {
            "mode": "auto",
            "selected": "custom",
            "reason": "policy_gp104_sm61_custom_present",
            "ctx": {
                "silicon_id": silicon_id,
                "compute_capability": cc,
                "pci_device_id": pci_device_id,
                "custom_binary_present": True,
            },
        }

    return {
        "mode": "auto",
        "selected": "cublas",
        "reason": "policy_portable_cublas_fallback",
        "ctx": {
            "silicon_id": silicon_id,
            "compute_capability": cc,
            "pci_device_id": pci_device_id,
            "custom_binary_present": bool(custom_present),
        },
    }


# -----------------------------------------------------------------------------
# Module runners (execution-only)
# -----------------------------------------------------------------------------
def _run_pcie(caps: Dict[str, Any], sink: EventSink) -> Dict[str, Any]:
    from pascalval.modules.pcie import run_all as pcie_run_all  # type: ignore

    sink.emit("pcie", "START", {})
    out = pcie_run_all(caps)
    out_d = out if isinstance(out, dict) else {"module": "pcie", "verdict": "ERROR", "error": "non-dict output"}
    sink.emit("pcie", "RESULT", {"verdict": _extract_verdict(out_d)})
    return out_d


def _run_memory(caps: Dict[str, Any], sink: EventSink) -> Dict[str, Any]:
    from pascalval.modules.memory import run_all as memory_run_all  # type: ignore

    sink.emit("memory", "START", {})
    out = memory_run_all(caps)
    out_d = out if isinstance(out, dict) else {"module": "memory", "verdict": "ERROR", "error": "non-dict output"}

    # Contract: prefetch-only validation is explicit (no type mixing)
    if isinstance(out_d, dict) and out_d.get("naive_gbs") is None:
        out_d.setdefault("naive_measured", False)
        out_d.setdefault("naive_skip_reason", "validation_prefetch_only")
        out_d.setdefault("source_url", "https://github.com/parallelArchitect/pascal-um-benchmark")

    sink.emit("memory", "RESULT", {"verdict": _extract_verdict(out_d)})
    return out_d


def _run_compute(
    caps: Dict[str, Any],
    route: Dict[str, Any],
    sink: EventSink,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    device_id = int(caps.get("device_id") or 0)

    selected = (route.get("selected") or "auto").strip()
    mode = (route.get("mode") or ("manual" if selected != "auto" else "auto")).strip()
    reason = (route.get("reason") or ("manual_override" if mode == "manual" else "policy")).strip()

    sink.emit("compute", "START", {"engine": selected, "mode": mode, "reason": reason})

    before_sgemm = _nvml_snapshot(device_id)

    sampler = _DuringSampler(device_id, interval_s=0.10)
    sampler.start()

    # Execute chosen engine
    out: Dict[str, Any]
    if selected == "custom":
        from pascalval.modules.sgemm_engine import run_all as sgemm_engine_run_all  # type: ignore

        out_any = sgemm_engine_run_all(caps, engine="custom", n=4096, reps=20)
        out = out_any if isinstance(out_any, dict) else {"module": "sgemm", "engine": "custom", "error": "non-dict output"}
    else:
        # cuBLAS fallback is portable across supported GPUs
        from pascalval.modules.cublas_sgemm import run_all as cublas_run_all  # type: ignore

        out_any = cublas_run_all(caps, n=4096, reps=20)
        out = out_any if isinstance(out_any, dict) else {"module": "sgemm", "engine": "cublas", "error": "non-dict output"}

    during = sampler.stop()
    after = _nvml_snapshot(device_id)

    # Theoretical from caps, falling back to BEFORE snapshot clock
    fallback_mhz = None
    if isinstance(before_sgemm, dict):
        fallback_mhz = before_sgemm.get("clock_sm_mhz")
    theo = _fp32_theoretical_gflops(caps, fallback_mhz=fallback_mhz)

    ratio = None
    if theo is not None and out.get("gflops") is not None:
        try:
            ratio = float(out["gflops"]) / float(theo)
        except Exception:
            ratio = None

    out["theoretical_gflops"] = theo
    out["theoretical_ratio"] = ratio
    out["verdict"] = _compute_verdict_from_ratio(ratio)

    out.setdefault("telemetry", {})
    out["telemetry"]["before_sgemm"] = before_sgemm
    out["telemetry"]["during_test"] = during
    out["telemetry"]["after"] = after

    sink.emit("compute", "RESULT", {"engine": out.get("engine", selected), "verdict": out.get("verdict")})
    return out, before_sgemm, during, after


# -----------------------------------------------------------------------------
# Pretty renderers
# -----------------------------------------------------------------------------
def _render_route(route: Dict[str, Any]) -> None:
    mode = route.get("mode", "-")
    sel = route.get("selected", "-")
    reason = route.get("reason", "-")

    if RICH and console and Table:
        t = Table(title="Compute Route", show_header=True)
        t.add_column("Field")
        t.add_column("Value", justify="right")
        t.add_row("mode", str(mode))
        t.add_row("selected", str(sel))
        t.add_row("reason", str(reason))
        console.print(t)
    else:
        print(f"Compute route: mode={mode} selected={sel} reason={reason}")


def _render_memory_table(mem: Dict[str, Any]) -> None:
    prefetch = mem.get("prefetch_gbs")
    if prefetch is None:
        return

    if RICH and console and Table:
        t = Table(title="Memory Bandwidth Results", show_header=True)
        t.add_column("Test")
        t.add_column("Bandwidth", justify="right")
        t.add_row("Prefetch UM", f"{float(prefetch):.1f} GB/s")
        console.print(t)
    else:
        print(f"Prefetch UM: {float(prefetch):.1f} GB/s")


def _render_sgemm_table(comp: Dict[str, Any]) -> None:
    n = comp.get("n")
    gflops = comp.get("gflops")
    tms = comp.get("time_ms")

    theo = comp.get("theoretical_gflops")
    ratio = comp.get("theoretical_ratio")
    verdict = _extract_verdict(comp)

    if not (n or gflops or tms):
        return

    if RICH and console and Table:
        t = Table(title="SGEMM Performance Results", show_header=True)
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        if n is not None:
            t.add_row("Matrix Size", f"{int(n)}x{int(n)}")
        if gflops is not None:
            t.add_row("Performance", f"{float(gflops):.1f} GFLOPS")
        if tms is not None:
            t.add_row("Time", f"{float(tms):.2f} ms")
        if theo is not None and ratio is not None:
            t.add_row("vs Theoretical", f"{ratio * 100.0:.1f}% ({float(theo):.0f} GFLOPS)")
        t.add_row("Status", verdict)
        console.print(t)
    else:
        print(f"SGEMM: n={n} gflops={gflops} time_ms={tms} verdict={verdict}")


def _render_gpu_state_table(
    cold_start: Optional[Dict[str, Any]],
    before_sgemm: Optional[Dict[str, Any]],
    during: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
) -> None:
    if not any([cold_start, before_sgemm, during, after]):
        return

    def fmt(s: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
        if not s:
            return ("-", "-", "-")
        try:
            clk = f"{float(s.get('clock_sm_mhz')):.0f} MHz"
        except Exception:
            clk = "-"
        try:
            temp = f"{float(s.get('temp_c')):.0f}°C"
        except Exception:
            temp = "-"
        try:
            pw = f"{float(s.get('power_w')):.1f}W"
        except Exception:
            pw = "-"
        return (clk, temp, pw)

    rows = [
        ("Cold Start",) + fmt(cold_start),
        ("Before SGEMM",) + fmt(before_sgemm),
        ("During Test",) + fmt(during),
        ("After",) + fmt(after),
    ]

    if RICH and console and Table:
        t = Table(title="GPU State During Test", show_header=True)
        t.add_column("Phase")
        t.add_column("Clocks", justify="right")
        t.add_column("Temp", justify="right")
        t.add_column("Power", justify="right")
        for r in rows:
            t.add_row(*r)
        console.print(t)
    else:
        print("GPU State During Test:")
        for r in rows:
            print(f"  {r[0]}: {r[1]} {r[2]} {r[3]}")


def _render_summary(pcie_v: str, mem_v: str, comp_v: str, final_v: str) -> None:
    if RICH and console and Table:
        t = Table(title="Summary", show_header=True)
        t.add_column("Check")
        t.add_column("Verdict", justify="right")
        t.add_row("PCIe", pcie_v)
        t.add_row("Memory", mem_v)
        t.add_row("Compute", comp_v)
        t.add_row("Final", final_v)
        console.print(t)
    else:
        print(f"PCIe:   {pcie_v}")
        print(f"Memory: {mem_v}")
        print(f"Compute:{comp_v}")
        print(f"Final:  {final_v}")


# -----------------------------------------------------------------------------
# Finalize (populate top-level caps fields from cold snapshot)
# -----------------------------------------------------------------------------
def _finalize(
    paths: RunPaths,
    run_id: str,
    gpu_uuid: str,
    status: str,
    failed_gate: Optional[str],
    caps: Dict[str, Any],
    pcie: Dict[str, Any],
    mem: Dict[str, Any],
    comp: Dict[str, Any],
    sink: EventSink,
    route: Optional[Dict[str, Any]],
    cold_snapshot: Optional[Dict[str, Any]],
) -> None:
    caps_out = dict(caps)

    # Populate stable top-level caps fields for analysis tooling
    if isinstance(cold_snapshot, dict):
        caps_out["clock_sm_mhz"] = cold_snapshot.get("clock_sm_mhz")
        caps_out["temperature_c"] = cold_snapshot.get("temp_c")
        caps_out["power_draw_w"] = cold_snapshot.get("power_w")

        # If caps has "features", reflect what we actually observed
        feat = caps_out.get("features")
        if isinstance(feat, dict):
            # If we can take a snapshot, we effectively have clocks/temp/power
            feat = dict(feat)
            feat["clocks"] = True
            feat["temp"] = True
            feat["power"] = True
            caps_out["features"] = feat

    doc: Dict[str, Any] = {
        "schema": "pascalval.run.v1",
        "run_id": run_id,
        "timestamp": local_ts(),
        "gpu_uuid": gpu_uuid,
        "status": status,
        "failed_gate": failed_gate,
        "caps": caps_out,
        "gpu_state": {
          "cold_start": cold_snapshot,
          "before_sgemm": (comp.get("telemetry") or {}).get("before_sgemm"),
          "during_sgemm": ((comp.get("telemetry") or {}).get("during_sgemm") or (comp.get("telemetry") or {}).get("during_test")),
          "after_sgemm": ((comp.get("telemetry") or {}).get("after_sgemm") or (comp.get("telemetry") or {}).get("after")),
        },
        "results": {
            "pcie": pcie,
            "memory": mem,
            "compute_route": route,
            "compute": comp,
        },
        "summary": {
            "pcie": _extract_verdict(pcie),
            "memory": _extract_verdict(mem),
            "compute": _extract_verdict(comp),
            "verdict": status,
        },
    }

    # PROMOTE_ROUTE_INTO_COMPUTE
    # Copy compute routing fields into results.compute so analysis tooling never sees nulls.
    if isinstance(route, dict) and isinstance(doc.get("results"), dict):
        comp_obj = doc["results"].get("compute")
        if isinstance(comp_obj, dict):
            comp_obj["mode"] = route.get("mode")
            comp_obj["selected"] = route.get("selected")
            comp_obj["reason"] = route.get("reason")
            comp_obj["ctx"] = route.get("ctx")

    # -------------------------------------------------------------------------
    # FIX: TRUTH INVARIANT FOR JSON
    # UI prints compute route + final verdict; JSON must persist both.
    # -------------------------------------------------------------------------
    if isinstance(doc.get("results"), dict):
        doc["results"]["route"] = route
        doc["results"]["final"] = status
    # -------------------------------------------------------------------------

    with open(paths.run_json, "w", encoding="utf-8") as f:
        def _strip_private_paths(x):
            if isinstance(x, dict):
                # Remove absolute-path leaks
                if "binary" in x:
                    del x["binary"]

                # Sanitize cached baseline path (privacy)
                if "cache_file" in x and isinstance(x.get("cache_file"), str):
                    cf = x["cache_file"]
                    if "/" in cf:
                        import os
                        x["cache_file_basename"] = os.path.basename(cf)
                        del x["cache_file"]
                for v in x.values():
                    _strip_private_paths(v)
            elif isinstance(x, list):
                for v in x:
                    _strip_private_paths(v)

        _strip_private_paths(doc)

        _ensure_results_schema(doc)
        json.dump(doc, f, indent=2, ensure_ascii=True)
        f.write("\n")

    sink.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# MULTI_GPU_SUBPROCESS_V1 helpers
# -----------------------------------------------------------------------------
def _list_physical_gpu_indices() -> list[int]:
    """
    Enumerate physical GPU indices using nvidia-smi (stable on NVIDIA driver installs).
    Returns [0..N-1].
    """
    import subprocess
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("nvidia-smi failed; cannot enumerate GPUs")
    # Each GPU line starts with "GPU 0:" etc.
    idx = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("GPU "):
            try:
                n = int(line.split()[1].rstrip(":"))
                idx.append(n)
            except Exception:
                pass
    return sorted(set(idx))


def _run_child_for_gpu(phys_index: int, args) -> int:
    """
    Spawn a child process with CUDA_VISIBLE_DEVICES set to the selected physical GPU.
    Child will see that GPU as device 0, preserving your current modules unchanged.
    """
    import os, sys, subprocess
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(phys_index)
    env["PASCALVAL_PHYS_GPU_INDEX"] = str(phys_index)  # for reporting/debug if you want
    cmd = [sys.executable, __file__, "--_child"]
    if getattr(args, "compute_engine", None) is not None:
        cmd += ["--compute-engine", args.compute_engine]
    r = subprocess.run(cmd, env=env)
    return r.returncode

def _ensure_results_schema(obj: dict) -> None:
    """
    Enforce stable results schema so forensic tooling never sees:
      - results.route == null
      - results.memory == {}
      - results.compute == {}
    This does NOT change gate logic; it only fills missing fields explicitly.
    """
    if not isinstance(obj, dict):
        return
    obj.setdefault("results", {})
    res = obj["results"]
    if not isinstance(res, dict):
        obj["results"] = {}
        res = obj["results"]

    # final must exist (string)
    if res.get("final") is None:
        res["final"] = "UNKNOWN"

    # route must exist (object)
    if res.get("route") is None:
        res["route"] = {"mode": "unknown", "selected": "unknown", "reason": "missing_route"}

    # pcie/memory should be dicts
    if not isinstance(res.get("pcie"), dict):
        res["pcie"] = {"verdict":"FAILED"}
    if not isinstance(res.get("memory"), dict) or res.get("memory") == {}:
        res["memory"] = {"verdict":"FAILED"}

    # compute must be dict and not empty
    comp = res.get("compute")
    if not isinstance(comp, dict) or comp == {}:
        res["compute"] = {
            "module": "sgemm",
            "engine": "unknown",
            "mode": "unknown",
            "selected": "unknown",
            "reason": "missing_compute",
            "gflops": None,
            "time_ms": None,
            "theoretical_ratio": None,
            "verdict":"FAILED",
            "error": "compute_not_recorded",
            "telemetry": None,
        }
    else:
        comp.setdefault("module", "sgemm")
        comp.setdefault("engine", "unknown")
        comp.setdefault("mode", "unknown")
        comp.setdefault("selected", comp.get("engine","unknown"))
        comp.setdefault("reason", "ok")
        comp.setdefault("gflops", None)
        comp.setdefault("time_ms", None)
        comp.setdefault("theoretical_ratio", None)
        comp.setdefault("verdict", "UNKNOWN")
        comp.setdefault("error", None)
        comp.setdefault("telemetry", None)




def _assert_ship_contract(run: dict) -> None:

    allowed = {"OK", "FAILED"}

    results = run.get("results") if isinstance(run, dict) else None

    if not isinstance(results, dict):

        raise RuntimeError("SHIP CONTRACT: missing results block")

    for gate in ("pcie", "memory", "compute"):

        blk = results.get(gate)

        if not isinstance(blk, dict):

            raise RuntimeError(f"SHIP CONTRACT: {gate} missing")

        v = blk.get("verdict")

        if v not in allowed:

            raise RuntimeError(f"SHIP CONTRACT: illegal {gate}.verdict={v!r}")

    final = results.get("final")

    if final not in allowed:

        raise RuntimeError(f"SHIP CONTRACT: illegal final={final!r}")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    # MULTI_GPU_SUBPROCESS_V1
    ap.add_argument("--all-gpus", action="store_true", help="Run validation on all GPUs (one run per GPU)")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--compute-engine", choices=["auto", "custom", "cublas"], default="auto")
    args = ap.parse_args()
    # MULTI_GPU_SUBPROCESS_V1 dispatch
    # Parent mode: enumerate physical GPUs and spawn one child run per GPU.
    # Child mode: executes the normal single-GPU pipeline unchanged.
    if args.all_gpus and not args._child:
        gidx = _list_physical_gpu_indices()
        if not gidx:
            raise SystemExit("No GPUs detected via nvidia-smi -L")
        rc = 0
        for i in gidx:
            _print_line("")
            _print_line("="*72)
            _print_line(f"Multi-GPU: validating physical GPU {i} (CUDA_VISIBLE_DEVICES={i})")
            _print_line("="*72)
            r = _run_child_for_gpu(i, args)
            if r != 0:
                rc = 2
        return rc

    _print_line("[pascalval] starting...")

    run_id = make_run_id()
    paths = _paths_for_run(run_id)
    sink = EventSink(paths.events_jsonl)

    caps = _get_caps()
    gpu_uuid = _gpu_uuid_from_caps(caps)

    device_id = int(caps.get("device_id") or 0)
    cold = _nvml_snapshot(device_id)

    box("PASCAL-GPU-VAL v1.0")
    _print_line(f"GPU: {caps.get('name', 'UNKNOWN')} ({caps.get('silicon_id', 'UNKNOWN')})")
    _print_line(f"UUID: {gpu_uuid}")
    if cold:
        _print_line(f"Initial State: {cold['clock_sm_mhz']:.0f} MHz, {cold['temp_c']:.0f}°C, {cold['power_w']:.1f}W")

    sink.emit("orchestrator", "RUN_START", {"run_id": run_id, "gpu_uuid": gpu_uuid})

    # PHASE 1: PCIe
    box("PHASE 1: PCIe Validation")

    # Exactly ONE Phase-1 sudo line, printed BEFORE PCIe module runs, always flushed.
    _print_line("(sudo already authorized)" if _sudo_status() else "(Requires sudo - you may be prompted for password)")

    pcie = _run_pcie(caps, sink)
    pcie_v = _extract_verdict(pcie)
    if _is_failish(pcie_v):
        box("VALIDATION COMPLETE")
        _render_summary(pcie_v, "FAILED", "FAILED", "FAILED")
        sink.emit("orchestrator", "RUN_END", {"status": "FAIL", "failed_gate": "pcie"})
        _finalize(paths, run_id, gpu_uuid, "FAIL", "pcie", caps, pcie, {}, {}, sink, route=None, cold_snapshot=cold)
        return 2

    # PHASE 2: Memory
    box("PHASE 2: Memory Validation")
    mem = _run_memory(caps, sink)
    _render_memory_table(mem)
    mem_v = _extract_verdict(mem)
    if _is_failish(mem_v):
        box("VALIDATION COMPLETE")
        _render_summary(pcie_v, mem_v, "FAILED", "FAILED")
        sink.emit("orchestrator", "RUN_END", {"status": "FAIL", "failed_gate": "memory"})
        _finalize(paths, run_id, gpu_uuid, "FAIL", "memory", caps, pcie, mem, {}, sink, route=None, cold_snapshot=cold)
        return 2

    # Compute route (pretty)
    route = _compute_route(caps, args.compute_engine)
    _render_route(route)

    # PHASE 3: SGEMM
    box("PHASE 3: SGEMM Validation")
    comp, before_sgemm, during, after = _run_compute(caps, route, sink)
    _render_sgemm_table(comp)
    _render_gpu_state_table(cold, before_sgemm, during, after)

    comp_v = _extract_verdict(comp)

    status = "OK"
    failed_gate: Optional[str] = None
    if _is_failish(comp_v):
        status = "FAIL"
        failed_gate = "compute"

    box("VALIDATION COMPLETE")
    _render_summary(pcie_v, mem_v, comp_v, status)
    _print_line(f"Run logged: {paths.run_json.as_posix()}")

    sink.emit("orchestrator", "RUN_END", {"status": status, "failed_gate": failed_gate})
    _finalize(paths, run_id, gpu_uuid, status, failed_gate, caps, pcie, mem, comp, sink, route=route, cold_snapshot=cold)
    return 0 if status == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
