"""GPU utilities — flush models between pipeline stages."""
import gc
import subprocess

def flush_gpu():
    """Release all GPU memory between pipeline stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass  # torch not loaded yet

def vram_free_mb() -> int:
    """Return free VRAM in MB, or 0 if no GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0

def vram_total_mb() -> int:
    """Return total VRAM in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0

def wait_for_vram(min_mb: int = 1000, timeout_sec: int = 300):
    """Block until at least min_mb of VRAM is free."""
    import time
    start = time.time()
    while time.time() - start < timeout_sec:
        free = vram_free_mb()
        if free >= min_mb:
            return free
        time.sleep(2)
    raise RuntimeError(f"VRAM not freed after {timeout_sec}s (need {min_mb}MB, have {vram_free_mb()}MB)")
