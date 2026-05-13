"""
NDCI Calculator — reference implementation
==========================================

Computes per-sensor Normalised Diagnostic Contribution Index (NDCI) from
the published fault × sensor matrix at /data/ndci-matrix.json.

Method follows:
  Suslu, Ali, Jennions (2025). NDCI Integration to MOSOF — An ECS Case.
  Sensors 25(9), 2661. doi:10.3390/s25092661

For each candidate sensor s, NDCI(s) is the prior-weighted sum of its
per-fault contributions, max-normalised so the most informative sensor
in the pool scores 1.0:

    NDCI(s) = Σ_f p_f · ndci[f, s]   /   max_t (Σ_f p_f · ndci[f, t])

The demo runs this under three prior settings to show how NDCI
re-ranks sensors when stakeholder priorities shift.

Run:
    python ndci_demo.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data", "ndci-matrix.json"))


# ── NDCI core ────────────────────────────────────────────────────────────

def ndci_per_sensor(
    matrix: np.ndarray,
    fault_priors: np.ndarray,
) -> np.ndarray:
    """Prior-weighted, max-normalised NDCI score per sensor.

    matrix       : shape (n_faults, n_sensors), each entry in [0, 1]
    fault_priors : shape (n_faults,), non-negative, will be normalised
    """
    p = fault_priors / fault_priors.sum()
    raw = (matrix * p[:, None]).sum(axis=0)  # shape (n_sensors,)
    return raw / max(raw.max(), 1e-12)


def make_priors(n_faults: int, faults: list[dict],
                heavy: str | None = None) -> np.ndarray:
    """Build a prior over faults. 'heavy' upweights one subsystem 3:1."""
    base = np.ones(n_faults)
    if heavy is None:
        return base
    for i, f in enumerate(faults):
        if f["subsystem"] == heavy:
            base[i] *= 3.0
    return base


# ── Driver ───────────────────────────────────────────────────────────────

def main():
    with open(DATA, encoding="utf-8") as fh:
        d = json.load(fh)

    faults = d["faults"]
    sensors = d["sensors"]
    matrix = np.asarray(d["ndci"], dtype=float)   # (n_faults, n_sensors)
    n_faults, n_sensors = matrix.shape
    assert n_faults == len(faults)
    assert n_sensors == len(sensors)

    print(f"Loaded NDCI matrix: {n_faults} faults x {n_sensors} sensors")
    print(f"Source: {d['_meta']['title']}\n")

    scenarios = [
        ("Uniform",     make_priors(n_faults, faults)),
        ("Heavy: ECS",  make_priors(n_faults, faults, heavy="ECS")),
        ("Heavy: Engine", make_priors(n_faults, faults, heavy="Engine")),
    ]

    # Compute and tabulate.
    results = {}
    for name, priors in scenarios:
        scores = ndci_per_sensor(matrix, priors)
        order = np.argsort(scores)[::-1]
        results[name] = (scores, order)

    # Print top-10 under each scenario.
    for name, (scores, order) in results.items():
        bar = "=" * 8
        print(f"{bar} Top 10 sensors  -  {name} priors {bar}")
        print(f"{'Rank':<5}{'ID':<14}{'Subsystem':<10}{'NDCI':>7}")
        for rank, idx in enumerate(order[:10], 1):
            s = sensors[idx]
            print(f"{rank:<5}{s['id']:<14}{s['subsystem']:<10}{scores[idx]:>7.3f}")
        print()

    # Write the per-sensor table for the website pipeline.
    out_path = os.path.join(HERE, "ndci_per_sensor.csv")
    uniform_scores, _ = results["Uniform"]
    ecs_scores, _ = results["Heavy: ECS"]
    eng_scores, _ = results["Heavy: Engine"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor_id", "subsystem", "family",
                    "NDCI_uniform", "NDCI_ECS_heavy", "NDCI_Engine_heavy"])
        for i, s in enumerate(sensors):
            w.writerow([
                s["id"], s["subsystem"], s["family"],
                f"{uniform_scores[i]:.4f}",
                f"{ecs_scores[i]:.4f}",
                f"{eng_scores[i]:.4f}",
            ])
    print(f"Wrote {out_path}")

    # Subsystem-level coverage check.
    print("\nUniform-prior NDCI by subsystem (mean of top-3 sensors):")
    by_sub: dict[str, list[float]] = defaultdict(list)
    for i, s in enumerate(sensors):
        by_sub[s["subsystem"]].append(float(uniform_scores[i]))
    for sub, vals in by_sub.items():
        top3 = sorted(vals, reverse=True)[:3]
        print(f"  {sub:8s} {np.mean(top3):.3f}   "
              f"(over {len(vals)} candidate sensors)")


if __name__ == "__main__":
    main()
