"""GPU-accelerated HAL Lasso backend for induced-seismicity pipeline.

Core: FISTA solver in JAX with sparse-tensor support. Bridges to hal9001
via rpy2 for basis construction; the Lasso solve happens on GPU.

Entry point: `fit_hal_gpu(X, y, ...)` in gpu_hal.hal_fit.
"""
__all__ = ["fista", "cv", "backend", "hal_fit"]
