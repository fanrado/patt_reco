"""Small helpers shared by the phase scripts."""
from __future__ import annotations

from dataclasses import replace

import yaml


def apply_override(cfg, spec: str):
    """Apply one `dotted.path=value` override to a nested frozen config.

    This is how every sweep in the validation plan is driven without writing a
    new YAML file for each slice:  --set event.n_objects='[5,5]'
    """
    path, sep, raw = spec.partition("=")
    if not sep:
        raise SystemExit(f"--set expects PATH=VALUE, got {spec!r}")
    value = yaml.safe_load(raw)
    if isinstance(value, list):
        value = tuple(value)

    def descend(node, keys):
        if not hasattr(node, keys[0]):
            raise SystemExit(f"--set: {type(node).__name__} has no field {keys[0]!r}")
        if len(keys) == 1:
            return replace(node, **{keys[0]: value})
        return replace(node, **{keys[0]: descend(getattr(node, keys[0]), keys[1:])})

    return descend(cfg, path.split("."))


def apply_overrides(cfg, specs):
    for spec in specs or ():
        cfg = apply_override(cfg, spec)
    return cfg


def pick_device(requested: str = "auto"):
    """Choose a torch device, and say why if the GPU is not usable."""
    import torch
    if requested not in ("auto", "cuda", "cpu"):
        raise SystemExit(f"--device must be auto/cuda/cpu, got {requested!r}")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise SystemExit("CUDA requested but not available "
                         f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})")
    print(f"  note: CUDA unavailable (torch {torch.__version__} built for CUDA "
          f"{torch.version.cuda}) -- falling back to CPU")
    return torch.device("cpu")


def describe_device(device) -> str:
    import torch
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} ({total:.1f} GB)"
