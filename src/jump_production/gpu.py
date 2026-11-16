"""GPU detection utilities."""

from __future__ import annotations

from functools import lru_cache

from loguru import logger


@lru_cache(maxsize=1)
def has_gpu() -> bool:
    """Check whether a CUDA GPU is available via CuPy.

    Imports cupy and probes the CUDA runtime driver. Returns False when cupy is
    not installed *or* when the driver / device is unavailable (e.g. CPU-only
    node with cupy in the conda env).
    """
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"CuPy installed but GPU unavailable ({type(e).__name__}: {e}), falling back to CPU")
        return False
