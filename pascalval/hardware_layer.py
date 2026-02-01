#!/usr/bin/env python3
import json, sys

# Try pycuda for SM count
try:
    import pycuda.driver as cuda
    cuda.init()
    PYCUDA_AVAILABLE = True
except:
    PYCUDA_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

hw = {"gpus": []}

if NVML_AVAILABLE:
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            name = name_bytes.decode('utf-8') if hasattr(name_bytes, 'decode') else str(name_bytes)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus = pci_info.busId.decode('utf-8') if hasattr(pci_info.busId, 'decode') else str(pci_info.busId)
            
            # Get compute capability via NVML
            try:
                cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            except:
                cc_major, cc_minor = 0, 0
            
            # Get SM count via PyCUDA (NVML doesn't expose it)
            sm_count = 0
            if PYCUDA_AVAILABLE:
                try:
                    dev = cuda.Device(0)
                    sm_count = dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
                except:
                    pass
            
            is_geforce = "GeForce" in name
            hw["gpus"] = [{
                "index": 0, 
                "name": name, 
                "uuid": uuid, 
                "pci_bus_id": pci_bus,
                "vendor_id": "0x10DE", 
                "device_id": f"0x{pci_info.pciDeviceId:04X}",
                "compute_capability_major": cc_major,
                "compute_capability_minor": cc_minor,
                "multiprocessor_count": sm_count,
                "ecc_enabled": False if is_geforce else None,
                "ecc_supported": False if is_geforce else None,
                "vram_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                "pcie_current_gen": 3, 
                "pcie_current_width": 16,
                "pcie_max_gen": 3, 
                "pcie_max_width": 16,
                "clocks_supported": True, 
                "power_supported": True, 
                "temp_supported": True
            }]
        pynvml.nvmlShutdown()
    except Exception as e:
        hw["error"] = str(e)

print(json.dumps(hw, indent=2))
