"""Which GPUs can train on this computer (asked once, in a separate process).

The website never imports PyTorch itself; a short child process asks PyTorch for CUDA
devices and reports name and memory. If there is none, the reason is explained in
plain words (no NVIDIA driver, driver installed but not running, CPU-only PyTorch).
The answer is cached; "Check again" refreshes it (e.g. after installing a driver).
"""

import json
import shutil
import subprocess
import sys
import threading

_PROBE = r"""
import json, warnings
warnings.filterwarnings("ignore")
import torch
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"id": f"cuda:{index}", "index": index, "name": props.name,
                        "memory_gb": round(props.total_memory / 1024 ** 3, 1)})
print(json.dumps({"torch": torch.__version__, "cuda_build": torch.version.cuda, "devices": devices}))
"""

_lock = threading.Lock()
_cached: dict | None = None


# Why no GPU is usable, in words a member can act on.
def _no_gpu_reason(cuda_build: str | None) -> str:
    if not cuda_build:
        return "No GPU available: this PyTorch build is CPU-only. Training runs on the CPU."
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return "No GPU available: no NVIDIA GPU driver was found. Training runs on the CPU."
    try:
        result = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is None or result.returncode != 0:
        return ("No GPU available: an NVIDIA driver is installed but not running (nvidia-smi failed). "
                "Restart the computer or reinstall the driver; until then training runs on the CPU.")
    return "No GPU available to PyTorch, although nvidia-smi lists a GPU. Training runs on the CPU."


# The detected GPUs and a message; cached until refresh=True.
def detect_gpus(refresh: bool = False) -> dict:
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        try:
            result = subprocess.run([sys.executable, "-B", "-c", _PROBE], capture_output=True, text=True,
                                    timeout=120, stdin=subprocess.DEVNULL)
            probe = json.loads(result.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
            probe = {"torch": None, "cuda_build": None, "devices": []}
        devices = probe["devices"]
        message = (f"{len(devices)} GPU{'s' if len(devices) > 1 else ''} available: "
                   + ", ".join(f"{d['name']} ({d['memory_gb']} GB)" for d in devices)) if devices \
            else _no_gpu_reason(probe.get("cuda_build"))
        _cached = {"available": bool(devices), "devices": devices, "message": message,
                   "torch": probe.get("torch"), "cuda_build": probe.get("cuda_build")}
        return _cached


# The cached answer without waiting (None if nobody asked yet).
def cached_gpus() -> dict | None:
    return _cached
