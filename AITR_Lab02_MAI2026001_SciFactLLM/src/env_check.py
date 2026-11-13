"""Environment check. Run from the project root:  python -m src.env_check"""
import importlib
import json
import platform
import sys
import time

from src.common import ROOT
from src.seed import set_seed

REQUIRED = ["numpy", "pandas", "sklearn", "yaml", "matplotlib", "torch", "transformers", "accelerate"]
OPTIONAL = ["bitsandbytes"]


def lib_version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except Exception as exc:
        return f"UNAVAILABLE ({type(exc).__name__}: {exc})"


def main():
    report = {"warnings": []}
    ok = True
    report["python"] = sys.version.split()[0]
    report["platform"] = platform.platform()
    print(f"python: {report['python']}")
    print(f"platform: {report['platform']}")

    versions = {}
    for name in REQUIRED + OPTIONAL:
        v = lib_version(name)
        versions[name] = v
        print(f"{name}: {v}")
        if v.startswith("UNAVAILABLE"):
            if name in REQUIRED:
                ok = False
            else:
                report["warnings"].append(f"optional library problem: {name}")
    report["versions"] = versions

    gpu = {"available": False}
    try:
        import torch
        gpu["available"] = torch.cuda.is_available()
        if gpu["available"]:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            gpu.update({"name": props.name, "total_gb": round(total / 1e9, 2),
                        "free_gb": round(free / 1e9, 2), "cuda_version_torch": torch.version.cuda})
            x = torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            _ = x @ x
            torch.cuda.synchronize()
            gpu["matmul_2048_seconds"] = round(time.time() - t0, 4)
        else:
            report["warnings"].append("No GPU visible to PyTorch: LLM runs will be very slow or impossible.")
    except Exception as exc:
        ok = False
        gpu["error"] = f"{type(exc).__name__}: {exc}"
    report["gpu"] = gpu
    print(f"gpu: {gpu}")

    import numpy as np
    set_seed(42)
    a = np.random.rand(3).tolist()
    set_seed(42)
    b = np.random.rand(3).tolist()
    report["seed_reproducible_numpy"] = a == b
    print(f"seed_reproducible_numpy: {a == b}")
    ok = ok and a == b

    out = ROOT / "results" / "env_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote: {out}")
    print("ENV CHECK:", "PASS" if ok else "FAIL")
    if report["warnings"]:
        print("warnings:", report["warnings"])


if __name__ == "__main__":
    main()
