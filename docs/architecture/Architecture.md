

---

# Architecture.md

## PascalVal Architecture

**Hardware-first GPU validation with explicit policy separation**

---

```
┌──────────────────────────────────────────────────────────────┐
│ USER INTERFACE                                                │
│ run_validation.py                                             │
│ ORCHESTRATOR: phases, gates, routing, UI                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ INTEGRATION LAYER (caps.py)                                   │
│ Combines hardware facts + knowledge model                     │
└──────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
┌──────────────────────────────┐   ┌──────────────────────────┐
│ RIGHT BRAIN                  │   │ LEFT BRAIN               │
│ hardware_layer.py            │   │ pascal_kb.py              │
│                              │   │                           │
│ • NVML facts                 │   │ • PCI → silicon map       │
│ • sysfs facts                │   │ • physical limits         │
│ • live sensors               │   │ • expected capabilities   │
│ • current state              │   │ • baseline guidance       │
└──────────────────────────────┘   └──────────────────────────┘
                    │                          │
                    └──────────────┬───────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│ EXECUTION MODULES (NO POLICY)                                 │
│ pcie.py | memory.py | sgemm_engine.py                         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ CANONICAL RUN LOGS                                            │
│ results/json/<run_id>/run.json                                │
│ results/json/<run_id>/events.jsonl                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Execution modules contain **zero policy**

Modules under `pascalval/modules/`:

* `pcie.py`
* `memory.py`
* `sgemm_engine.py`
* `cublas_sgemm.py`
* `sgemm_custom_pascal.py`

These modules **only measure and report**.
They never decide routing, pass/fail, or execution order.

This guarantees:

* deterministic behavior
* reusable measurement code
* no hidden heuristics

---

### 2. Early-exit validation (hardware-aware)

Validation order follows real failure causality:

1. PCIe
2. Memory
3. Compute

If PCIe or Memory fails:

* execution stops immediately
* SGEMM is not run
* power, heat, and time are not wasted

PascalVal validates **the path**, not just the result.

---

### 3. Integration layer = reasoning boundary

`caps.py` is the only layer where:

* live hardware facts
* silicon knowledge
* user intent (auto vs manual)

are combined.

This keeps reasoning:

* explainable
* auditable
* extendable across GPU generations

---

### 4. Compute routing is explicit and logged

Routing decisions are **data**, not console output.

Example recorded in `run.json`:

```json
"route": {
  "mode": "auto",
  "selected": "custom",
  "reason": "policy_gp104_sm61_custom_present",
  "ctx": {
    "silicon_id": "GP104",
    "compute_capability": "6.1",
    "requested_engine": "cublas"
  }
}
```

All routing decisions are preserved.

---

## Multi-GPU Architecture

### Parent / child execution model

* Validation modules remain unchanged
* Each GPU runs as an independent child
* A parent process coordinates execution and aggregation

```
results/json/
├── 20260126_032645/          ← per-GPU child run
│   ├── run.json
│   └── events.jsonl
└── 20260126_084206_MULTIGPU/ ← parent summary
    ├── multigpu_summary.json
    └── gpu0_stdout.txt
```

This enables:

* per-GPU isolation
* mixed-health systems
* scalable diagnostics

---

## Event Model

Each run emits a bounded, deterministic sequence:

```
orchestrator RUN_START
pcie         START
pcie         RESULT
memory       START
memory       RESULT
compute      START
compute      RESULT
orchestrator RUN_END
```

Events are:

* monotonic
* JSONL
* automation-safe

---

## Scope

Originally Pascal-focused, PascalVal applies to:

* any NVIDIA GPU (Linux)
* PCIe transport health
* memory behavior validation
* compute sanity checks

The name reflects origin, not limitation.

---

## Summary

PascalVal’s architecture is:

* hardware-first
* policy-separated
* early-exit aware
* multi-GPU capable
* evidence-driven

This is validation infrastructure — not benchmarking theater.

---




