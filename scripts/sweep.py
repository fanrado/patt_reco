#!/usr/bin/env python3
"""Measure how baseline accuracy falls as the two classes are made to overlap.

    python scripts/sweep.py --levels 0,0.25,0.5,0.75,1.0 -o sweeps/overlap

One difficulty scalar `d` in [0, 1] drives four generator fields at once. At
d=0 the classes are the easy default; at d=1 a shower is a few-segment kinked
polyline and a track is always a strong arc, so neither class has a giveaway
feature the other lacks.

Each level runs the real pipeline as subprocesses -- make_dataset, train,
evaluate -- rather than reimplementing it, because the point is to measure
what those scripts actually do.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# (easy at d=0, hard at d=1)
EASY_HARD = {
    "shower.open_angle": ((0.20, 0.55), (0.02, 0.06)),
    "shower.spread": ((0.01, 0.03), (0.001, 0.004)),
    "shower.max_nodes": (200, 7),
    # track.curvature is deliberately NOT swept. Raising its floor was meant to
    # remove the "straight => track" giveaway, but alongside the shower rows it
    # only inverted it: every hard shower became near-straight and every hard
    # track an arc, so curvature stayed decisive and the curve stayed flat.
    # Overlapping two classes means moving one of them toward the other, not
    # both past each other -- so tracks keep their own [0.0, 2.2].
}


def _lerp(easy, hard, d: float):
    if isinstance(easy, tuple):
        return tuple(_lerp(e, h, d) for e, h in zip(easy, hard))
    value = easy + (hard - easy) * d
    return int(round(value)) if isinstance(easy, int) else round(value, 6)


def overrides_for(d: float) -> list[str]:
    """The --set strings that put the generator at difficulty `d`."""
    specs = []
    for field, (easy, hard) in EASY_HARD.items():
        value = _lerp(easy, hard, d)
        rendered = (f"[{', '.join(str(v) for v in value)}]"
                    if isinstance(value, tuple) else str(value))
        specs.append(f"{field}={rendered}")
    return specs


def _run(command: list[str]) -> subprocess.CompletedProcess:
    print("    $ " + " ".join(command))
    return subprocess.run(command, capture_output=True, text=True)


def _check(result: subprocess.CompletedProcess, what: str) -> str:
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {result.returncode}):\n"
                           f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return result.stdout


def run_level(d: float, out_dir: Path, args) -> dict:
    """Build, train and evaluate one difficulty level."""
    tag = f"d{int(round(d * 100)):03d}"
    data_dir = out_dir / tag / "data"
    specs = overrides_for(d)

    command = [sys.executable, "scripts/make_dataset.py", *args.configs,
               "-o", str(data_dir)]
    for spec in specs + args.set_data:
        command += ["--set", spec]
    _check(_run(command), "make_dataset")

    command = [sys.executable, "scripts/train.py", args.train_config,
               "--device", args.device,
               "--set", f"data.root={data_dir}",
               "--set", f"run.name=sweep_{tag}",
               "--set", f"run.out_dir={out_dir / tag / 'runs'}"]
    for spec in args.set_train:
        command += ["--set", spec]
    stdout = _check(_run(command), "train")

    match = re.search(r"^run:\s+(.+)$", stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("could not find the run directory in train.py output")
    run_dir = Path(match.group(1).strip())

    command = [sys.executable, "scripts/evaluate.py", str(run_dir / "best.pt"),
               "--data", str(data_dir / "test.npz"), "--device", args.device]
    _check(_run(command), "evaluate")

    report = json.loads((run_dir / "test_report.json").read_text())
    confusion = report["confusion_matrix"]
    correct = sum(confusion[i][i] for i in range(len(confusion)))
    return {
        "d": d,
        "overrides": specs,
        "accuracy": report["accuracy"],
        "auc": report["roc_auc"],
        "n_misclassified": report["n_images"] - correct,
        "n_images": report["n_images"],
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", default="0,0.25,0.5,0.75,1.0",
                        help="comma-separated difficulty values in [0, 1]")
    parser.add_argument("-o", "--out", type=Path, default=Path("sweeps/overlap"))
    parser.add_argument("--configs", nargs="+",
                        default=["configs/data/tracks.yaml", "configs/data/showers.yaml"])
    parser.add_argument("--train-config", default="configs/train/base.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--set-data", action="append", default=[], metavar="PATH=VALUE",
                        help="extra --set passed to make_dataset.py at every level")
    parser.add_argument("--set-train", action="append", default=[], metavar="PATH=VALUE",
                        help="extra --set passed to train.py at every level")
    args = parser.parse_args()

    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, d in enumerate(levels, 1):
        print(f"\n=== level {i}/{len(levels)}: d={d} ===")
        for spec in overrides_for(d):
            print(f"    {spec}")
        try:
            results.append(run_level(d, args.out, args))
        except Exception as exc:
            # one bad level should not cost the whole sweep
            print(f"  LEVEL FAILED: {exc}")
            results.append({"d": d, "overrides": overrides_for(d), "error": str(exc)})

        print(f"\n{'d':>6}  {'accuracy':>9}  {'auc':>7}  {'wrong':>7}")
        for row in results:
            if "error" in row:
                print(f"{row['d']:>6.2f}  {'FAILED':>9}  {'-':>7}  {'-':>7}")
            else:
                print(f"{row['d']:>6.2f}  {row['accuracy']:>9.4f}  "
                      f"{row['auc']:>7.4f}  {row['n_misclassified']:>7d}")

    path = args.out / "sweep_results.json"
    path.write_text(json.dumps({"levels": levels, "results": results}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
