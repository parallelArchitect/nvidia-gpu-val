#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/caps.py

Description:
  Capability probe (integration layer).

  This module is the single source of truth for GPU identity and capability fields
  consumed by the orchestrator and routing policy.

  Design:
    - RIGHT BRAIN: raw probe (hardware_layer.py + NVML identity)
    - LEFT BRAIN: interpretation (pascal_kb.PascalKnowledgeBase)
    - Output: stable, JSON-safe capability map

Key guarantees (shipping contract):
  - Always returns:
      pci_vendor_id, pci_device_id, pci_vendor_device_id, pci_bus_id
    (or explicit None values if NVML is unavailable).
  - No absolute repo paths are emitted by this module.

NVML reference:
  https://developer.nvidia.com/management-library-nvml
"""

from __future__ import annotations

from typing import Any, Dict
import json
import subprocess
from pathlib import Path

from .pascal_kb import PascalKnowledgeBase

def _split_pci_vendor_device_id(v: object):
    """Return (vendor_id, device_id, vendor_device_id) as hex strings.

    Accepts NVML-style or mixed inputs like:
      - '0x1B8010DE' (device+vendor)
      - '0x1b80' (device only) -> vendor unknown
      - int
      - None
    """
    if v is None:
        return (None, None, None)
    try:
        if isinstance(v, int):
            hx = f"0x{v:08X}"
        else:
            hx = str(v).strip()
            if not hx:
                return (None, None, None)
            if not hx.lower().startswith('0x'):
                # tolerate raw hex without 0x
                hx = '0x' + hx
    except Exception:
        return (None, None, None)

    # Normalize casing for stable JSON (vendor_device_id commonly shown uppercase)
    # Split if we have 8 hex digits after 0x (vendor+device packed).
    body = hx[2:]
    if len(body) == 8:
        vendor = '0x' + body[4:].lower()
        device = '0x' + body[:4].lower()
        combined = '0x' + body.upper()
        return (vendor, device, combined)

    # Device-only (4 hex digits). Vendor unknown here.
    if len(body) == 4:
        return (None, '0x' + body.lower(), None)

    # Unknown format
    return (None, None, None)



def _nvml_pci_identity(device_index: int = 0) -> Dict[str, Any]:
    """
    Return PCI identity from NVML.

    Returns:
      {
        pci_vendor_id: "0x10de",
        pci_device_id: "0x1b80",
        pci_vendor_device_id: "0x1B8010DE",  # device + vendor (Pascal KB compatible)
        pci_bus_id: "00000000:01:00.0"
      }
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            try:
                pci = pynvml.nvmlDeviceGetPciInfo_v3(h)
            except Exception:
                pci = pynvml.nvmlDeviceGetPciInfo(h)

            # NVML exposes numeric IDs in newer structs; guard for both shapes.
            vendor = getattr(pci, "pciVendorId", None)
            device = getattr(pci, "pciDeviceId", None)

            # busId is consistently present (bytes in some bindings)
            bus_id = getattr(pci, "busId", None)
            if isinstance(bus_id, (bytes, bytearray)):
                bus_id = bus_id.decode(errors="ignore")

            out: Dict[str, Any] = {
                "pci_vendor_id": None,
                "pci_device_id": None,
                "pci_vendor_device_id": None,
                "pci_bus_id": bus_id,
            }

            if isinstance(vendor, int) and isinstance(device, int):
                vendor_id = f"0x{vendor:04x}"
                device_id = f"0x{device:04x}"

                # Pascal KB mapping expects "0x<DEVICE><VENDOR>" for the combined form.
                combined = f"0x{device:04X}{vendor:04X}"

                out.update(
                    {
                        "pci_vendor_id": vendor_id,
                        "pci_device_id": device_id,
                        "pci_vendor_device_id": combined,
                    }
                )
            return out
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return {
            "pci_vendor_id": None,
            "pci_device_id": None,
            "pci_vendor_device_id": None,
            "pci_bus_id": None,
        }


def get_gpu_raw_data(device_id: int = 0) -> Dict[str, Any]:
    """
    RIGHT BRAIN:
      - Run hardware_layer.py (JSON)
      - Enrich with NVML PCI identity (authoritative for device IDs)

    Returns a minimal raw dict suitable for PascalKnowledgeBase.interpret_gpu().
    """
    hw_script = Path(__file__).resolve().parent / "hardware_layer.py"
    raw: Dict[str, Any] = {}

    # 1) hardware_layer JSON (best-effort)
    try:
        result = subprocess.run(
            ["python3", str(hw_script)],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        data = json.loads(result.stdout or "{}")
        gpus = data.get("gpus") or []
        if isinstance(gpus, list) and len(gpus) > device_id and isinstance(gpus[device_id], dict):
            gpu = gpus[device_id]

            raw.update(
                {
                    "name": gpu.get("name", "Unknown GPU"),
                    "uuid": gpu.get("uuid", "unknown"),
                    "memory_total_mb": int(gpu.get("vram_bytes", 0) // (1024**2)),
                    "memory_free_mb": int(gpu.get("vram_bytes", 0) // (1024**2)),
                    "multiprocessor_count": int(gpu.get("multiprocessor_count", 0)),
                    "compute_capability_major": int(gpu.get("compute_capability_major", 0)),
                    "compute_capability_minor": int(gpu.get("compute_capability_minor", 0)),
                    # keep bus_id if hardware_layer had it
                    "pci_bus_id": gpu.get("pci_bus_id"),
                }
            )

            # Some older hardware_layer versions provide only a device_id field.
            # Keep it, but NVML enrichment below is authoritative.
            if gpu.get("device_id"):
                raw["pci_device_id"] = str(gpu.get("device_id"))

    except Exception:
        pass

    # 2) NVML PCI identity (authoritative)
    pci = _nvml_pci_identity(device_index=device_id)

    # Normalize output keys expected by the rest of the stack:
    raw["pci_vendor_id"] = pci.get("pci_vendor_id")
    raw["pci_device_id"] = pci.get("pci_device_id") or raw.get("pci_device_id") or "0x0000"
    raw["pci_vendor_device_id"] = pci.get("pci_vendor_device_id")
    raw["pci_bus_id"] = pci.get("pci_bus_id") or raw.get("pci_bus_id") or "unknown"

    # Ensure required keys exist even on failure:
    raw.setdefault("name", "Unknown GPU")
    raw.setdefault("uuid", "unknown")
    raw.setdefault("memory_total_mb", 0)
    raw.setdefault("memory_free_mb", raw.get("memory_total_mb", 0))

    return raw


def probe_caps(device_id: int = 0) -> Dict[str, Any]:
    """
    LEFT BRAIN integration:
      - interpret_gpu() assigns silicon_id, cores, limits, baselines, etc.
      - we return a stable, production capability map
    """
    kb = PascalKnowledgeBase()
    raw_data = get_gpu_raw_data(device_id)
    # Normalize PCI IDs (vendor/device) from NVML-style combined values
    # Accept combined from any of: pci_vendor_device_id, pci_device_id (legacy), device_id (legacy)
    _pci_combined = (raw_data.get("pci_vendor_device_id") or raw_data.get("pci_device_id") or raw_data.get("device_id"))
    pci_vendor_id, pci_device_id, pci_vendor_device_id = _split_pci_vendor_device_id(_pci_combined)
    raw_data["pci_vendor_id"] = pci_vendor_id
    raw_data["pci_device_id"] = pci_device_id
    raw_data["pci_vendor_device_id"] = pci_vendor_device_id
    interpreted = kb.interpret_gpu(raw_data)

    features = {
        "pcie": True,
        "clocks": "clock_sm_mhz" in interpreted and interpreted.get("clock_sm_mhz") is not None,
        "power": "power_draw_w" in interpreted and interpreted.get("power_draw_w") is not None,
        "temp": "temperature_c" in interpreted and interpreted.get("temperature_c") is not None,
        "ecc": kb.check_feature_support("ecc", interpreted),
        "nvlink": kb.check_feature_support("nvlink", interpreted),
        "tensor_cores": kb.check_feature_support("tensor_cores", interpreted),
        "full_fp16": kb.check_feature_support("full_fp16", interpreted),
    }

    memory_test_plan = kb.plan_memory_tests(interpreted)
    optimal_sgemm_n = kb.get_optimal_sgemm_size(interpreted)
    performance_baseline = kb.get_baseline_performance(interpreted)

    caps: Dict[str, Any] = {
        # Core identification
        "device_id": device_id,
        "uuid": raw_data.get("uuid", "unknown"),
        "name": interpreted.get("name", raw_data.get("name", "Unknown GPU")),
        "silicon_id": interpreted.get("silicon_id", "UNKNOWN"),
        "compute_capability": interpreted.get("compute_capability", "UNKNOWN"),

        # PCI identity (shipping requirement)
        "pci_vendor_id": raw_data.get("pci_vendor_id"),
        "pci_device_id": raw_data.get("pci_device_id"),
        "pci_vendor_device_id": raw_data.get("pci_vendor_device_id"),
        "pci_bus_id": raw_data.get("pci_bus_id"),

        # Hardware specs
        "cuda_cores_total": interpreted.get("cuda_cores_total", 0),
        "multiprocessor_count": interpreted.get("multiprocessor_count", 0),
        "cores_per_sm": interpreted.get("cores_per_sm", 0),

        # Memory info
        "memory_total_mb": interpreted.get("memory_total_mb", 0),
        "memory_free_mb": interpreted.get("memory_free_mb", 0),
        "physical_max_vram_mb": interpreted.get("physical_max_vram_mb", 0),
        "safe_test_max_mb": interpreted.get("safe_test_max_mb", 0),
        "memory_bus_bits": interpreted.get("physical_memory_bus_bits", 0),

        # Performance characteristics
        "l2_cache_kb": interpreted.get("physical_l2_cache_kb", 0),
        "fp16_performance": interpreted.get("fp16_performance", "unknown"),
        "fp64_ratio": interpreted.get("fp64_ratio", 0.0),

        # Current state
        "temperature_c": interpreted.get("temperature_c"),
        "power_draw_w": interpreted.get("power_draw_w"),
        "clock_sm_mhz": interpreted.get("clock_sm_mhz"),

        # Feature support map
        "features": features,

        # Intelligent test planning
        "recommended_tests": {
            "memory_sizes_mb": memory_test_plan,
            "optimal_sgemm_matrix_size": optimal_sgemm_n,
            "sgemm_reason": f"Fits in {interpreted.get('physical_l2_cache_kb', 0)}KB L2 cache",
        },

        # Performance baselines
        "performance_baseline": performance_baseline,

        # Debug (keep, but these are internal)
        "_interpreted": interpreted,
        "_raw": raw_data,
    }
    return caps


if __name__ == "__main__":
    caps = probe_caps(device_id=0)
    print(json.dumps(caps, indent=2, sort_keys=True))
