"""
Find the correct k2 wheel for your environment and print the install command.

Usage:
    uv run scripts/find_k2.py
"""

import requests
from bs4 import BeautifulSoup
import torch
import sys
import platform

py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
arch = platform.machine()  # e.g. x86_64, aarch64
parts = torch.__version__.split("+")
torch_ver = parts[0]
torch_device = parts[1] if len(parts) > 1 else ""
os_family = platform.system()
if os_family == "Linux":
    base_url = "https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/"
    if torch_device.startswith("cu"):
        k2_device = torch_device[: len("cu12")] + "." + torch_device[len("cu12") :]
        k2_device = k2_device.replace("cu", "cuda")
    else:
        k2_device = "cpu"
elif os_family == "Windows":
    base_url = "https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-windows/"
    k2_device = "cpu"
elif os_family == "Darwin":
    base_url = "https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-macos/"
    k2_device = "cpu"
else:
    raise NotImplementedError(f"Only support Linux, Windows or MacOS, found {os_family}")

whl_url = None
for a_el in BeautifulSoup(requests.get(base_url + torch_ver).text, "html.parser").select("a.external"):
    _whl_url = a_el.get("href")
    if k2_device in _whl_url and py_version in _whl_url and arch in _whl_url:
        whl_url = _whl_url
        break

if whl_url is None:
    raise RuntimeError(f"k2 wheel for torch {torch_ver}, {k2_device} on {py_version} not found")

print(f"Run:\n\n    uv pip install {whl_url}\n")
print("Then verify:\n\n    uv run python -c \"import k2; print(k2.__dev_version__)\"\n")
