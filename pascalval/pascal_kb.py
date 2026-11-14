#!/usr/bin/env python3
"""
Author: Joe McLaren (Human–AI collaborative engineering)
Project: pascal-gpu-val
File: pascalval/pascal_kb.py

Left Brain Knowledge Base (Hardware Expectations)

Purpose:
  Provide a minimal, factual knowledge layer used to interpret raw hardware facts
  (PCI IDs, compute capability, bandwidth ceilings, etc.) without embedding policy.

Architecture:
  - RIGHT BRAIN (hardware_layer.py): collects raw facts (NVML, sysfs, CUDA)
  - LEFT BRAIN  (this file): maps facts -> expectations (what is physically plausible)
  - ORCHESTRATOR (run_validation.py): applies gates/routing and writes artifacts

Knowledge sources:
  - Public NVIDIA specifications and documentation
  - Factual limits and mappings only (no verbatim reproduction of copyrighted text)
"""

from typing import Dict, List, Optional, Tuple

# =============================================================================
# MINIMAL KNOWLEDGEBASE - Physical Silicon Specifications
# =============================================================================
# Source: NVIDIA official documentation (publicly available specifications)
# These are FACTS about physical hardware - not copyrightable
# Attribution: NVIDIA Corporation technical specifications

PASCAL_SILICON_LIMITS = {
    # GP107 - GTX 1050/1050 Ti
    "GP107": {
        "boost_clock_mhz": 1392,  # GTX 1050 Ti official spec
        "max_vram_gb": 4,
        "max_cuda_cores": 768,
        "fp16_crippled": True,  # 1/64 FP32 rate
        "fp64_ratio": 1/32,
        "l2_cache_kb": 1024,
        "memory_bus_bits": 128,
    },
    
    # GP106 - GTX 1060
    "GP106": {
        "boost_clock_mhz": 1708,  # GTX 1060 official spec
        "max_vram_gb": 6,
        "max_cuda_cores": 1280,
        "fp16_crippled": True,  # 1/64 FP32 rate
        "fp64_ratio": 1/32,
        "l2_cache_kb": 1536,
        "memory_bus_bits": 192,
    },
    
    # GP104 - GTX 1080/1070
    "GP104": {
        "boost_clock_mhz": 1733,  # GTX 1080 official spec (NVIDIA)
        "max_vram_gb": 8,
        "max_cuda_cores": 2560,
        "fp16_crippled": True,  # 1/64 FP32 rate
        "fp64_ratio": 1/32,
        "l2_cache_kb": 2048,
        "memory_bus_bits": 256,
    },
    
    # GP102 - GTX 1080 Ti/Titan Xp
    "GP102": {
        "boost_clock_mhz": 1582,  # GTX 1080 Ti official spec
        "max_vram_gb": 12,
        "max_cuda_cores": 3584,
        "fp16_crippled": True,  # 1/64 FP32 rate
        "fp64_ratio": 1/32,
        "l2_cache_kb": 3072,
        "memory_bus_bits": 384,
    },
    
    # GP100 - Tesla P100
    "GP100": {
        "boost_clock_mhz": 1328,  # Tesla P100 official spec
        "max_vram_gb": 16,
        "max_cuda_cores": 3584,
        "fp16_crippled": False,  # Full FP16 rate
        "fp64_ratio": 1/2,  # Full FP64
        "l2_cache_kb": 4096,
        "memory_bus_bits": 4096,  # HBM2
    },
}

# PCI Device ID to Silicon mapping
# Source: NVIDIA PCI ID assignments (public information)
PCI_TO_SILICON = {
    # GP100 variants (highest silicon)
    "0x15f7": "GP100",  # Tesla P100 PCIe 12GB
    "0x15f8": "GP100",  # Tesla P100 PCIe 16GB
    "0x15f9": "GP100",  # Tesla P100 SMX2 16GB
    
    # GP102 variants (Titan X / 1080 Ti / Quadro P6000)
    "0x1b00": "GP102",  # Titan X Pascal
    "0x1b02": "GP102",  # Titan Xp
    "0x1b06": "GP102",  # GTX 1080 Ti
    "0x1b30": "GP102",  # Quadro P6000
    
    # GP104 variants (GTX 1080/1070 + Quadro P5000/P4000)
    "0x1b80": "GP104",  # GTX 1080
    "0x1b81": "GP104",  # GTX 1070
    "0x1b82": "GP104",  # GTX 1070 Ti
    "0x1bb0": "GP104",  # Quadro P5000
    "0x1bb1": "GP104",  # Quadro P4000
    
    # GP106 variants (GTX 1060 + Quadro P2000)
    "0x1c02": "GP106",  # GTX 1060 6GB
    "0x1c03": "GP106",  # GTX 1060 3GB
    "0x1c20": "GP106",  # GTX 1060 Mobile
    "0x1c3b": "GP106",  # Quadro P2000
    
    # GP107 variants (GTX 1050)
    "0x1c81": "GP107",  # GTX 1050
    "0x1c82": "GP107",  # GTX 1050 Ti
    "0x1c8c": "GP107",  # GTX 1050 Mobile
    "0x1c8d": "GP107",  # GTX 1050 Ti Mobile
}

# =============================================================================
# INTELLIGENCE LAYER - Interpret Raw Data
# =============================================================================

class PascalKnowledgeBase:
    """
    Left Brain - Interprets raw hardware data with intelligence
    
    Takes raw sensor data from hardware_layer.py (right brain)
    Adds physical limits, capabilities, and intelligent decisions
    Returns interpreted capabilities for action layer (CLI)
    """
    
    def __init__(self):
        self.silicon_limits = PASCAL_SILICON_LIMITS
        self.pci_map = PCI_TO_SILICON
    
    def interpret_gpu(self, raw_data: Dict) -> Dict:
        """Enrich driver facts using only known, source-backed Pascal specifications."""
        pci_id = str(raw_data.get("pci_device_id") or "").strip().lower()
        silicon_id = self.pci_map.get(pci_id, "UNKNOWN")
        limits = self.silicon_limits.get(silicon_id, {})
        major = raw_data.get("compute_capability_major")
        minor = raw_data.get("compute_capability_minor")
        compute_cap = f"{major}.{minor}" if major is not None and minor is not None else "UNKNOWN"
        cores_per_sm = self._get_cores_per_sm(compute_cap)
        sm_count = raw_data.get("multiprocessor_count")
        total_cores = (int(sm_count) * cores_per_sm) if cores_per_sm is not None and sm_count is not None else None
        known_vram_gb = limits.get("max_vram_gb")
        physical_max_mb = known_vram_gb * 1024 if known_vram_gb is not None else None
        free_mb = raw_data.get("memory_free_mb")
        # Memory planning uses measured free VRAM; never substitute a model-size default.
        available_mb = min(free_mb, physical_max_mb) if free_mb is not None and physical_max_mb is not None else free_mb
        safe_max = int(available_mb * 0.95) if available_mb is not None and available_mb > 0 else 0
        fp16_crippled = limits.get("fp16_crippled")
        return {
            **raw_data,
            "silicon_id": silicon_id,
            "compute_capability": compute_cap,
            "cuda_cores_total": total_cores,
            "cores_per_sm": cores_per_sm,
            "physical_max_vram_mb": physical_max_mb,
            "physical_max_cores": limits.get("max_cuda_cores"),
            "physical_memory_bus_bits": limits.get("memory_bus_bits"),
            "physical_l2_cache_kb": limits.get("l2_cache_kb"),
            "fp16_performance": ("crippled" if fp16_crippled else "full") if fp16_crippled is not None else "unknown",
            "fp64_ratio": limits.get("fp64_ratio"),
            "safe_test_max_mb": safe_max,
        }

    
    def _get_cores_per_sm(self, compute_cap: str) -> Optional[int]:
        """Only derive CUDA cores/SM for documented Pascal compute capabilities."""
        if compute_cap == "6.0":
            return 64
        if compute_cap in ("6.1", "6.2"):
            return 128
        return None

    
    def validate_memory_test(self, test_size_mb: int, interpreted_caps: Dict) -> Dict:
        """Check measured free VRAM and documented physical limits when available."""
        physical_max_mb = interpreted_caps.get("physical_max_vram_mb")
        free_mb = interpreted_caps.get("memory_free_mb")
        silicon_id = interpreted_caps.get("silicon_id", "UNKNOWN")
        if test_size_mb <= 0:
            return {"allowed": False, "reason": "INVALID_TEST_SIZE", "message": "Test size must be positive"}
        if physical_max_mb is not None and test_size_mb > physical_max_mb:
            return {"allowed": False, "reason": "PHYSICALLY_IMPOSSIBLE",
                    "message": f"Test requests {test_size_mb}MB but {silicon_id} physically limited to {physical_max_mb}MB maximum",
                    "suggestion": f"Test with up to {int(physical_max_mb * 0.9)}MB instead",
                    "physical_limit_mb": physical_max_mb}
        if free_mb is None:
            return {"allowed": False, "reason": "FREE_MEMORY_UNKNOWN",
                    "message": "Cannot authorize a memory allocation without measured free memory"}
        if test_size_mb > free_mb:
            return {"allowed": False, "reason": "INSUFFICIENT_FREE_MEMORY",
                    "message": f"Test requests {test_size_mb}MB but only {free_mb}MB free",
                    "suggestion": f"Test with up to {int(free_mb * 0.9)}MB instead",
                    "free_memory_mb": free_mb}
        warnings = []
        if physical_max_mb and test_size_mb > physical_max_mb * 0.95:
            warnings.append(f"High utilization: {test_size_mb}MB is {(test_size_mb/physical_max_mb)*100:.1f}% of physical maximum")
        return {"allowed": True, "reason": "OK", "warnings": warnings}

    
    def plan_memory_tests(self, interpreted_caps: Dict) -> List[int]:
        """
        Plan memory test sizes based on hardware intelligence
        
        Uses knowledge of physical limits to calculate appropriate test sizes
        NOT random - based on actual hardware capabilities
        
        Returns: List of test sizes in MB [25%, 50%, 75%, 90% of safe max]
        """
        safe_max_mb = interpreted_caps["safe_test_max_mb"]
        
        test_sizes = [
            int(safe_max_mb * 0.25),  # Light load
            int(safe_max_mb * 0.50),  # Medium load
            int(safe_max_mb * 0.75),  # Heavy load
            int(safe_max_mb * 0.90),  # Stress test
        ]
        
        return test_sizes
    
    def get_optimal_sgemm_size(self, interpreted_caps: Dict) -> Optional[int]:
        """Return an L2-based suggestion only where L2 capacity is documented."""
        import math
        l2_cache_kb = interpreted_caps.get("physical_l2_cache_kb")
        if not l2_cache_kb or l2_cache_kb <= 0:
            return None
        max_n = int(math.sqrt(l2_cache_kb * 1024 / 4))
        n = 1
        while n * 2 <= max_n:
            n *= 2
        return n

    
    def get_baseline_performance(self, interpreted_caps: Dict) -> Dict:
        """Capability discovery must not run kernels or invent measured baselines.

        Historical analyzer owns baseline estimation from verified, per-GPU run evidence.
        """
        return {
            "silicon_id": interpreted_caps.get("silicon_id", "UNKNOWN"),
            "theoretical_fp32_gflops": None,
            "realistic_baseline_gflops": None,
            "healthy_range_min_gflops": None,
            "healthy_range_max_gflops": None,
            "measured_source": "unavailable_not_measured",
            "cache_file": None,
            "note": "No measurement performed during capability discovery; use historical run analyzer",
        }

    
    def check_feature_support(self, feature: str, interpreted_caps: Dict) -> Optional[bool]:
        """False means known absent; None means not established by this Pascal KB."""
        silicon_id = interpreted_caps.get("silicon_id")
        limits = self.silicon_limits.get(silicon_id)
        if limits is None:
            return None
        feature_map = {
            "tensor_cores": False,
            "rt_cores": False,
            "nvlink": silicon_id == "GP100",
            "full_fp16": not limits["fp16_crippled"],
            "full_fp64": silicon_id == "GP100",
            "ecc": silicon_id == "GP100",
        }
        return feature_map.get(feature)



# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_silicon_from_pci_id(pci_id: str) -> str:
    """Quick lookup: PCI ID → Silicon ID"""
    return PCI_TO_SILICON.get(pci_id, "UNKNOWN")


def get_physical_limits(silicon_id: str) -> Dict:
    """Quick lookup: Silicon ID → Physical limits"""
    return PASCAL_SILICON_LIMITS.get(silicon_id, {})


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    """Test the knowledge base with simulated raw data"""
    
    print("="*60)
    print("PASCAL KNOWLEDGE BASE TEST")
    print("="*60)
    print()
    
    # Simulate raw data from hardware_layer.py (GTX 1080)
    simulated_raw_data = {
        "name": "GeForce GTX 1080",
        "pci_device_id": "0x1b80",
        "memory_total_mb": 8192,
        "memory_free_mb": 7234,
        "temperature_c": 45,
        "power_draw_w": 85,
        "clock_sm_mhz": 1733,
        "compute_capability_major": 6,
        "compute_capability_minor": 1,
        "multiprocessor_count": 20,
    }
    
    # Initialize knowledge base
    kb = PascalKnowledgeBase()
    
    # Interpret raw data (LEFT BRAIN ADDS INTELLIGENCE)
    print("Interpreting raw sensor data...")
    interpreted = kb.interpret_gpu(simulated_raw_data)
    
    print(f"  Silicon: {interpreted['silicon_id']}")
    print(f"  CUDA Cores: {interpreted['cuda_cores_total']}")
    print(f"  Physical VRAM Max: {interpreted['physical_max_vram_mb']} MB")
    print(f"  Safe Test Max: {interpreted['safe_test_max_mb']} MB")
    print(f"  FP16 Performance: {interpreted['fp16_performance']}")
    print()
    
    # Test validation logic
    print("Testing validation logic...")
    
    # Test 1: Impossible test (16GB on 8GB GPU)
    impossible_test = 16384
    result = kb.validate_memory_test(impossible_test, interpreted)
    print(f"  {impossible_test}MB test: {'✓ ALLOWED' if result['allowed'] else '✗ BLOCKED'}")
    if not result["allowed"]:
        print(f"    Reason: {result['reason']}")
        print(f"    Message: {result['message']}")
        print(f"    Suggestion: {result['suggestion']}")
    print()
    
    # Test 2: Plan appropriate tests
    print("Planning memory tests...")
    test_plan = kb.plan_memory_tests(interpreted)
    print(f"  Test sizes: {test_plan} MB")
    print()
    
    # Test 3: Get optimal SGEMM size
    print("Calculating optimal SGEMM matrix size...")
    optimal_n = kb.get_optimal_sgemm_size(interpreted)
    print(f"  Optimal size: {optimal_n}×{optimal_n}")
    print(f"  Reason: Fits in {interpreted['physical_l2_cache_kb']}KB L2 cache")
    print()
    
    # Test 4: Get performance baseline
    print("Getting realistic performance baseline...")
    baseline = kb.get_baseline_performance(interpreted)
    print(f"  Theoretical FP32: {baseline['theoretical_fp32_gflops']:.1f} GFLOPS")
    print(f"  Realistic Baseline: {baseline['realistic_baseline_gflops']:.1f} GFLOPS")
    print(f"  Healthy Range: {baseline['healthy_range_min_gflops']:.1f} - {baseline['healthy_range_max_gflops']:.1f} GFLOPS")
    print()
    
    # Test 5: Feature support
    print("Checking feature support...")
    features = ["tensor_cores", "full_fp16", "nvlink", "ecc"]
    for feature in features:
        supported = kb.check_feature_support(feature, interpreted)
        status = "✓" if supported else "✗"
        print(f"  {status} {feature}")
    
    print()
    print("="*60)
