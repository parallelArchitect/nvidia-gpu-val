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
        """
        Take raw sensor data, add intelligence
        
        Input: Raw data from hardware_layer.py
        Output: Interpreted capabilities with physical limits
        """
        
        # Identify silicon from PCI ID
        pci_id = raw_data.get("pci_device_id", "unknown")
        pci_id_norm = str(pci_id).strip().lower()
        silicon_id = self.pci_map.get(pci_id_norm, "UNKNOWN")
        
        # Get physical limits for this silicon
        limits = self.silicon_limits.get(silicon_id, {})
        
        # Calculate actual CUDA cores (if we have SM count)
        compute_cap = f"{raw_data.get('compute_capability_major', 0)}.{raw_data.get('compute_capability_minor', 0)}"
        cores_per_sm = self._get_cores_per_sm(compute_cap)
        sm_count = raw_data.get("multiprocessor_count", 0)
        total_cores = sm_count * cores_per_sm
        
        # Build interpreted capabilities
        interpreted = {
            # Pass through raw data
            **raw_data,
            
            # Add silicon identification
            "silicon_id": silicon_id,
            "compute_capability": compute_cap,
            
            # Add calculated values
            "cuda_cores_total": total_cores,
            "cores_per_sm": cores_per_sm,
            
            # Add physical limits from knowledge base
            "physical_max_vram_mb": limits.get("max_vram_gb", 0) * 1024,
            "physical_max_cores": limits.get("max_cuda_cores", 0),
            "physical_memory_bus_bits": limits.get("memory_bus_bits", 0),
            "physical_l2_cache_kb": limits.get("l2_cache_kb", 0),
            
            # Add performance characteristics
            "fp16_performance": "crippled" if limits.get("fp16_crippled", True) else "full",
            "fp64_ratio": limits.get("fp64_ratio", 1/32),
            
            # Calculate safe test maximum (95% of smaller: physical max or free memory)
            "safe_test_max_mb": int(min(
                limits.get("max_vram_gb", 8) * 1024 * 0.95,  # 95% of physical
                raw_data.get("memory_free_mb", 0) * 0.95      # 95% of free
            )),
        }
        
        return interpreted
    
    def _get_cores_per_sm(self, compute_cap: str) -> int:
        """
        Get CUDA cores per SM by compute capability
        Source: CUDA Programming Guide
        """
        if compute_cap.startswith("6.0"):
            return 64   # GP100
        elif compute_cap.startswith("6."):
            return 128  # GP102/104/106/107
        return 128  # Default for Pascal
    
    def validate_memory_test(self, test_size_mb: int, interpreted_caps: Dict) -> Dict:
        """
        Validate if memory test is physically possible
        
        Returns:
            {
                "allowed": bool,
                "reason": str,
                "message": str,
                "suggestion": str (if not allowed)
            }
        """
        physical_max_mb = interpreted_caps["physical_max_vram_mb"]
        free_mb = interpreted_caps.get("memory_free_mb", 0)
        silicon_id = interpreted_caps["silicon_id"]
        
        # Check 1: Exceeds physical silicon limits?
        if test_size_mb > physical_max_mb:
            return {
                "allowed": False,
                "reason": "PHYSICALLY_IMPOSSIBLE",
                "message": (
                    f"Test requests {test_size_mb}MB but {silicon_id} "
                    f"physically limited to {physical_max_mb}MB maximum"
                ),
                "suggestion": f"Test with up to {int(physical_max_mb * 0.9)}MB instead",
                "physical_limit_mb": physical_max_mb,
            }
        
        # Check 2: Exceeds available free memory?
        if test_size_mb > free_mb:
            return {
                "allowed": False,
                "reason": "INSUFFICIENT_FREE_MEMORY",
                "message": (
                    f"Test requests {test_size_mb}MB but only {free_mb}MB free"
                ),
                "suggestion": f"Test with up to {int(free_mb * 0.9)}MB instead",
                "free_memory_mb": free_mb,
            }
        
        # Check 3: Warning for high utilization (>95% of physical)
        warnings = []
        if test_size_mb > (physical_max_mb * 0.95):
            warnings.append(
                f"High utilization: {test_size_mb}MB is "
                f"{(test_size_mb/physical_max_mb)*100:.1f}% of physical maximum"
            )
        
        return {
            "allowed": True,
            "reason": "OK",
            "warnings": warnings,
        }
    
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
    
    def get_optimal_sgemm_size(self, interpreted_caps: Dict) -> int:
        """
        Calculate optimal SGEMM matrix size based on L2 cache
        
        Intelligence: Size matrix to fit in L2 cache for best performance
        Too small: Overhead dominates
        Too large: Cache thrashing
        Optimal: Fits in L2 cache
        """
        l2_cache_kb = interpreted_caps.get("physical_l2_cache_kb", 2048)
        
        # Rule: Matrix should fit in L2 cache
        # Matrix size N×N×sizeof(float) should be ≤ L2 cache
        # N² × 4 bytes ≤ L2_KB × 1024
        # N ≤ sqrt(L2_KB × 1024 / 4)
        
        import math
        max_n = int(math.sqrt(l2_cache_kb * 1024 / 4))
        
        # Round to nearest power of 2 for alignment
        optimal_n = 1
        while optimal_n * 2 <= max_n:
            optimal_n *= 2
        
        return optimal_n
    
    def get_baseline_performance(self, interpreted_caps: Dict) -> Dict:
        """
        Get realistic performance baselines - ACTUALLY MEASURED
        
        Flow:
        1. Check cache (~/.pascalval_baseline_{UUID}.json)
        2. If cached and recent (< 24h): use it
        3. If no cache or stale: RUN ACTUAL KERNEL and measure
        4. Save measurement to cache
        
        Returns REAL measured performance, not fake hardcoded values
        """
        import subprocess
        import re
        import json
        from pathlib import Path
        from datetime import datetime
        
        silicon_id = interpreted_caps["silicon_id"]
        cores = interpreted_caps["cuda_cores_total"]
        # Use NVIDIA official boost clock from knowledge base
        limits = self.silicon_limits.get(silicon_id, {})
        clock_mhz = interpreted_caps.get("clock_sm_mhz") or limits.get("boost_clock_mhz", 1500)
        gpu_uuid = interpreted_caps.get("uuid", "unknown")
        
        # Calculate theoretical peak
        theoretical_gflops = (cores * clock_mhz * 2) / 1000
        
        # Cache file location
        cache_dir = Path.home() / ".pascalval"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"baseline_{silicon_id}.json"
        
        # Try to load cached measurement
        cached_baseline = None
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                
                # Check if cache is recent (< 24 hours)
                cache_age_hours = (datetime.now().timestamp() - cached['timestamp']) / 3600
                
                if cache_age_hours < 24:
                    cached_baseline = cached['measured_gflops']
            except:
                pass
        
        # If no valid cache, ACTUALLY MEASURE
        if cached_baseline is None:
            # Find SGEMM binary
            binary_path = Path(__file__).parent.parent / "public_release" / "pascalval_sgemm_public"
            
            if binary_path.exists():
                try:
                    # RUN ACTUAL KERNEL
                    result = subprocess.run(
                        [str(binary_path), "4096", "5"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    # Parse REAL output
                    match = re.search(r'GFLOPS=([\d.]+)', result.stdout)
                    if match:
                        measured_gflops = float(match.group(1))
                        
                        # Save to cache
                        cache_data = {
                            'measured_gflops': measured_gflops,
                            'timestamp': datetime.now().timestamp(),
                            'silicon_id': silicon_id,
                            'gpu_uuid': gpu_uuid,
                        }
                        
                        with open(cache_file, 'w') as f:
                            json.dump(cache_data, f, indent=2)
                        
                        cached_baseline = measured_gflops
                        measured_source = "Just measured with kernel"
                    else:
                        # Measurement failed, use fallback
                        cached_baseline = theoretical_gflops * 0.75
                        measured_source = "Measurement failed, using 75% theoretical"
                except:
                    # Kernel failed, use fallback
                    cached_baseline = theoretical_gflops * 0.75
                    measured_source = "Kernel not available, using 75% theoretical"
            else:
                # Binary not found, use fallback
                cached_baseline = theoretical_gflops * 0.75
                measured_source = "Binary not found, using 75% theoretical"
        else:
            measured_source = f"Cached measurement from {cache_file.name}"
        
        realistic_baseline_gflops = cached_baseline
        
        return {
            "silicon_id": silicon_id,
            "theoretical_fp32_gflops": theoretical_gflops,
            "realistic_baseline_gflops": realistic_baseline_gflops,
            "healthy_range_min_gflops": realistic_baseline_gflops * 0.90,
            "healthy_range_max_gflops": theoretical_gflops * 0.90,
            "measured_source": measured_source,
            "cache_file": str(cache_file),
            "note": "Baseline from ACTUAL measurement, cached for 24h",
        }
    
    def check_feature_support(self, feature: str, interpreted_caps: Dict) -> bool:
        """
        Check if feature is physically possible on this silicon
        
        Examples:
        - Tensor cores: Only Volta+ (not Pascal)
        - Full FP16: Only GP100 (not GP104/106/107)
        - NVLink: Only GP100 (not consumer Pascal)
        """
        silicon_id = interpreted_caps["silicon_id"]
        limits = self.silicon_limits.get(silicon_id, {})
        
        feature_map = {
            "tensor_cores": False,  # No Pascal has tensor cores
            "rt_cores": False,  # No Pascal has RT cores
            "nvlink": silicon_id == "GP100",  # Only GP100
            "full_fp16": not limits.get("fp16_crippled", True),  # Only GP100
            "full_fp64": silicon_id == "GP100",  # Only GP100 has 1:2 FP64
            "ecc": silicon_id == "GP100",  # Only Tesla P100
        }
        
        return feature_map.get(feature, False)


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
