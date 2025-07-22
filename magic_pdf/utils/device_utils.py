# @Time     : 2025/7/22 15:18
# @Software : Python 3.10
# @About    :

import importlib
from functools import lru_cache

import torch


@lru_cache
def is_torch_cuda_available():
    return torch.cuda.is_available()


@lru_cache
def is_torch_mps_available():
    if hasattr(torch.backends, "mps"):
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    return False


@lru_cache
def is_torch_npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu

    try:
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


@lru_cache
def auto_device_mem_ratio(ratio):
    if is_torch_cuda_available():
        mem_free, mem_total = torch.cuda.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio
    elif is_torch_mps_available():
        mem_free, mem_total = torch.mps.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio
    elif is_torch_npu_available():
        mem_free, mem_total = torch.npu.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio
    return ratio
