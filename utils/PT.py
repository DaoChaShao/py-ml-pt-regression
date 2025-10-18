#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 12:06
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   PT.py
# @Desc     :   

from torch import cuda, backends, device


def device_checker(option: str = "") -> dict[str, device]:
    """ Check Available Device (CPU, GPU, MPS)
    Check available devices (CPU, GPU, MPS) and return as a dictionary.
    Optionally filter by `option` = ["cpu", "cuda", "mps"].

    Returns:
        dict[str, device]: A dictionary of available torch.device objects.

    :param option: filter option for device type
    :return: dictionary of available devices
    """
    available_devices: dict[str, device] = {"cpu": device("cpu")}

    # CUDA (NVIDIA GPU)
    if cuda.is_available():
        count: int = cuda.device_count()
        print(f"Detected {count} CUDA GPU(s):")
        for i in range(count):
            print(f"GPU {i}: {cuda.get_device_name(i)}")
            print(f"- Memory Usage:")
            print(f"- Allocated: {round(cuda.memory_allocated(i) / 1024 ** 3, 1)} GB")
            print(f"- Cached:    {round(cuda.memory_reserved(i) / 1024 ** 3, 1)} GB")
            available_devices[f"cuda:{i}"] = device(f"cuda:{i}")

    # MPS (Apple Silicon GPU)
    elif backends.mps.is_available():
        print("Apple MPS device detected.")
        available_devices["mps"] = device("mps")

    # Fallback: CPU
    else:
        print("⚙Using CPU (no GPU or MPS available).")

    # Filter based on "option"
    devices: dict[str, device] = {}
    option = option.lower()
    match option:
        case "cpu":
            devices = {k: v for k, v in available_devices.items() if "cpu" in k}
        case "mps":
            devices = {k: v for k, v in available_devices.items() if "mps" in k}
        case "cuda":
            devices = {k: v for k, v in available_devices.items() if "cuda" in k}
        case _:
            devices = available_devices

    print(f"Available devices: {list(devices.keys())}")
    return devices
