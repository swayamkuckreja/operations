"""
Sensitivity analysis on the DAPDP MILP.

Mirrors the analysis in Section 5.5 of Mulumba & Diabat (2024). We vary the
three drone parameters that the paper highlights as most influential and
re-solve the MILP for several seed instances:

  * endurance       e            (paper Tab. 3 / Fig. 9)
  * drone speed     s'           (paper Tab. 4 / Fig. 10)
  * drone cost      alpha        (no direct paper analogue, included because
                                  the assignment guidelines explicitly ask
                                  for sensitivity to cost parameters)

For each (parameter, value) pair we solve the DAPDP and the truck-only
baseline; the figure of merit is SPDP = 1 - z_DAPDP / z_PDP, the cost
saving over the truck-only solution. Number of drone sorties is also
recorded as a secondary metric.

The output is a CSV table under results/sensitivity/ together with two
matplotlib figures (savings vs parameter, sorties vs parameter).

Instance budget. With a full Gurobi licence we use an ensemble of n in
{6, 8} on grids {10, 20} with three seeds each (12 instances). Each
parameter sweep solves DAPDP for every (value, instance) pair; PDP is
cached per instance.
"""

from __future__ import annotations

import os
import csv
import time
from dataclasses import dataclass, asdict
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from instance_generator import generate_instance, Instance
from dapdp_model import solve_dapdp
from baseline import solve_truck_only


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
@dataclass
class Run:
    parameter: str
    value: float
    instance: str
    n: int
    grid: float
    seed: int
    z_pdp: float
    z_dapdp: float
    spdp_pct: float
    n_sorties: int
    runtime_s: float


def _run_one(inst: Instance,
             param: str,
             value: float,
             pdp_cache: Dict[str, float],
             time_limit: float = 60.0) -> Run:
    """Solve DAPDP (and PDP if not cached) for a single (instance, value)."""
    # PDP cost is invariant to drone parameters, so cache it per instance.
    if inst.name not in pdp_cache:
        sol_pdp = solve_truck_only(inst, time_limit=time_limit)
        pdp_cache[inst.name] = sol_pdp.objective
    z_pdp = pdp_cache[inst.name]

    t0 = time.time()
    sol = solve_dapdp(inst, time_limit=time_limit)
    runtime = time.time() - t0

    if not np.isfinite(sol.objective) or not np.isfinite(z_pdp) or z_pdp <= 0:
        spdp = float("nan")
    else:
        spdp = 100.0 * (1.0 - sol.objective / z_pdp)

    n_sort = sum(len(s) for s in sol.drone_sorties.values())
    return Run(parameter=param, value=value, instance=inst.name,
               n=inst.n, grid=inst.grid_size, seed=inst.seed,
               z_pdp=z_pdp, z_dapdp=sol.objective,
               spdp_pct=spdp, n_sorties=n_sort, runtime_s=runtime)


# ---------------------------------------------------------------------------
def _make_ensemble() -> List[Instance]:
    """Ensemble of test instances at paper-realistic sizes. Kept modest in
    cardinality because each parameter sweep multiplies the solve count."""
    out: List[Instance] = []
    for n in [6, 8]:
        for grid in [10.0, 20.0]:
            for seed in [1, 2, 3]:
                out.append(generate_instance(n=n, grid_size=grid, seed=seed))
    return out


# ---------------------------------------------------------------------------
def sweep_endurance(instances: List[Instance],
                    values_min: List[float] = [5, 7.5, 10, 15, 30, 60],
                    time_limit: float = 300.0) -> List[Run]:
    """Vary endurance over the same set of values as the paper (Tab. 3)."""
    print("\n--- endurance sweep ---")
    results: List[Run] = []
    pdp_cache: Dict[str, float] = {}
    for v_min in values_min:
        for inst in instances:
            inst.endurance = v_min / 60.0   # hours
            r = _run_one(inst, "endurance_min", v_min, pdp_cache, time_limit)
            results.append(r)
            print(f"  e={v_min:5.1f} min  inst={inst.name:8s}  "
                  f"PDP=${r.z_pdp:6.3f}  DAPDP=${r.z_dapdp:6.3f}  "
                  f"SPDP={r.spdp_pct:5.1f}%  sorties={r.n_sorties}  "
                  f"({r.runtime_s:.1f}s)")
    # Reset to paper default
    for inst in instances:
        inst.endurance = 30.0 / 60.0
    return results


def sweep_drone_speed(instances: List[Instance],
                      values_mph: List[float] = [25, 35, 50, 75],
                      time_limit: float = 300.0) -> List[Run]:
    """Vary drone speed (paper Tab. 4)."""
    print("\n--- drone-speed sweep ---")
    results: List[Run] = []
    pdp_cache: Dict[str, float] = {}
    for v_mph in values_mph:
        for inst in instances:
            inst.drone_speed = v_mph
            r = _run_one(inst, "drone_speed_mph", v_mph, pdp_cache, time_limit)
            results.append(r)
            print(f"  s'={v_mph:4.0f} mph  inst={inst.name:8s}  "
                  f"PDP=${r.z_pdp:6.3f}  DAPDP=${r.z_dapdp:6.3f}  "
                  f"SPDP={r.spdp_pct:5.1f}%  sorties={r.n_sorties}  "
                  f"({r.runtime_s:.1f}s)")
    for inst in instances:
        inst.drone_speed = 50.0
    return results


def sweep_alpha(instances: List[Instance],
                values_alpha: List[float] = [0.05, 0.10, 0.25, 0.50, 1.00],
                time_limit: float = 300.0) -> List[Run]:
    """Vary the drone cost factor alpha (= c'/c). At alpha = 1 there is no
    cost incentive to use the drone, so SPDP should approach 0 (modulo the
    trivial saving from skipping a node visit on the truck route)."""
    print("\n--- alpha sweep ---")
    results: List[Run] = []
    pdp_cache: Dict[str, float] = {}
    for a in values_alpha:
        for inst in instances:
            inst.drone_cost_factor = a
            r = _run_one(inst, "alpha", a, pdp_cache, time_limit)
            results.append(r)
            print(f"  alpha={a:4.2f}  inst={inst.name:8s}  "
                  f"PDP=${r.z_pdp:6.3f}  DAPDP=${r.z_dapdp:6.3f}  "
                  f"SPDP={r.spdp_pct:5.1f}%  sorties={r.n_sorties}  "
                  f"({r.runtime_s:.1f}s)")
    for inst in instances:
        inst.drone_cost_factor = 0.10
    return results


# ---------------------------------------------------------------------------
def _save_csv(rows: List[Run], path: str):
    if not rows:
        return
    keys = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _aggregate(rows: List[Run], xkey: str) -> Dict[float, Dict[str, float]]:
    """Collect mean savings and mean sorties per parameter value."""
    by_x: Dict[float, List[Run]] = {}
    for r in rows:
        by_x.setdefault(r.value, []).append(r)
    out = {}
    for v, group in sorted(by_x.items()):
        out[v] = {
            "mean_spdp": float(np.nanmean([g.spdp_pct for g in group])),
            "mean_sorties": float(np.mean([g.n_sorties for g in group])),
            "n_inst": len(group),
        }
    return out


def _plot_sweep(rows: List[Run], xlabel: str, fname: str, fig_title: str):
    agg = _aggregate(rows, "value")
    xs = list(agg.keys())
    ys_savings = [agg[x]["mean_spdp"] for x in xs]
    ys_sorties = [agg[x]["mean_sorties"] for x in xs]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    color = "#1f77b4"
    ax1.plot(xs, ys_savings, "o-", color=color, label="mean SPDP")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Mean savings vs PDP (%)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(xs, ys_sorties, "s--", color=color2, label="mean sorties")
    ax2.set_ylabel("Mean number of drone sorties", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(fig_title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    instances = _make_ensemble()
    print(f"Sensitivity ensemble: {len(instances)} instances "
          f"(n in {{6,8}}, grid in {{10,20}}, seeds {{1,2,3}})")

    end_rows = sweep_endurance(instances, time_limit=300.0)
    spd_rows = sweep_drone_speed(instances, time_limit=300.0)
    alp_rows = sweep_alpha(instances, time_limit=300.0)

    _save_csv(end_rows, os.path.join(OUT_DIR, "endurance.csv"))
    _save_csv(spd_rows, os.path.join(OUT_DIR, "drone_speed.csv"))
    _save_csv(alp_rows, os.path.join(OUT_DIR, "alpha.csv"))

    _plot_sweep(end_rows, "Drone endurance (min)",
                "endurance.pdf",
                "Sensitivity to drone endurance")
    _plot_sweep(spd_rows, "Drone speed (mph)",
                "drone_speed.pdf",
                "Sensitivity to drone speed")
    _plot_sweep(alp_rows, r"Drone cost factor $\alpha$",
                "alpha.pdf",
                "Sensitivity to drone cost factor")

    # Also produce a small summary table (mean over instances per value)
    print("\n=== summary tables ===")
    for label, rows in [("endurance (min)", end_rows),
                        ("drone speed (mph)", spd_rows),
                        ("alpha", alp_rows)]:
        print(f"\n{label}")
        agg = _aggregate(rows, "value")
        print(f"  {'value':>8s}  {'mean SPDP %':>11s}  {'mean sorties':>12s}  N")
        for v, d in agg.items():
            print(f"  {v:8.3f}  {d['mean_spdp']:11.2f}  "
                  f"{d['mean_sorties']:12.2f}  {d['n_inst']}")


if __name__ == "__main__":
    main()
