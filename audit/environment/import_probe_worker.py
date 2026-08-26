#!/usr/bin/env python3
"""Native-import probe isolated so a crash cannot erase parent evidence."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path


OUTPUT = Path("/data/project/tzh/papers/ActMask/audit/environment/import_probe.json")


def main() -> None:
    result: dict = {"imports": {}, "observed_versions": {}, "errors": []}
    for distribution, module_name in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("sapien", "sapien"),
        ("mani_skill", "mani_skill"),
    ):
        try:
            module = importlib.import_module(module_name)
            result["imports"][module_name] = {
                "ok": True,
                "module_file": getattr(module, "__file__", None),
            }
            result["observed_versions"][distribution] = importlib.metadata.version(distribution)
        except BaseException as error:
            result["imports"][module_name] = {
                "ok": False,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            result["errors"].append(f"import {module_name}: {error!r}")
    try:
        result["observed_versions"]["opencv_python"] = importlib.metadata.version("opencv-python")
    except importlib.metadata.PackageNotFoundError:
        result["observed_versions"]["opencv_python"] = None

    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            available = bool(torch_module.cuda.is_available())
            result["torch_cuda"] = {
                "compiled_cuda": torch_module.version.cuda,
                "available": available,
                "device_count": int(torch_module.cuda.device_count()),
                "device_0": torch_module.cuda.get_device_name(0) if available else None,
            }
        except BaseException as error:
            result["torch_cuda"] = {"error": repr(error)}
            result["errors"].append(f"torch CUDA probe: {error!r}")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
