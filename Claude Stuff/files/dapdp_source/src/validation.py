"""
Validation against the small-instance results of Mulumba & Diabat (2024),
Table A.5. The paper reports z_DAPDP, z_PDP, SPDP and runtime for instances
of size n in {3, 4, 5} on grids {5, 10, 20, 30}. The free Gurobi licence we
use is capped at ~2000 vars/constraints, which limits us to n <= 4. We
therefore reproduce the n=3 and n=4 rows only.

Note that the paper does not publish per-seed instance data, so we cannot
match individual rows to those tables; instead we report the mean SPDP and
mean runtime over a small ensemble of seeds and verify that they fall in
the same ballpark as the paper's reported means (SPDP ~ 25%, runtime
seconds-to-minutes for these sizes).
"""

from __future__ import annotations

import os
import time
import csv
import numpy as np

from instance_generator import generate_instance
from dapdp_model import solve_dapdp
from baseline import solve_truck_only


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "verification")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    rows = []
    for n in [3, 4]:
        for grid in [5.0, 10.0, 20.0, 30.0]:
            spdp_list = []
            t_list = []
            for seed in range(1, 6):
                inst = generate_instance(n=n, grid_size=grid, seed=seed)
                t0 = time.time()
                sol = solve_dapdp(inst, time_limit=120.0)
                t_dapdp = time.time() - t0
                sol_pdp = solve_truck_only(inst, time_limit=120.0)
                if not (np.isfinite(sol.objective) and np.isfinite(sol_pdp.objective)
                        and sol_pdp.objective > 0):
                    continue
                spdp = 100.0 * (1.0 - sol.objective / sol_pdp.objective)
                spdp_list.append(spdp)
                t_list.append(t_dapdp)
                print(f"  n={n} grid={grid:4.0f} seed={seed}: "
                      f"PDP=${sol_pdp.objective:6.3f}  DAPDP=${sol.objective:6.3f}  "
                      f"SPDP={spdp:5.1f}%  ({t_dapdp:.1f}s)")
            if spdp_list:
                rows.append({
                    "n": n,
                    "grid": grid,
                    "n_seeds": len(spdp_list),
                    "mean_spdp_pct": float(np.mean(spdp_list)),
                    "std_spdp_pct": float(np.std(spdp_list, ddof=1)) if len(spdp_list) > 1 else 0.0,
                    "mean_runtime_s": float(np.mean(t_list)),
                })

    print("\n=== validation table (mean over 5 seeds) ===")
    print(f"{'n':>3s} {'grid':>5s} {'mean SPDP %':>12s} {'std %':>7s} "
          f"{'mean t (s)':>10s} {'N seeds':>8s}")
    for r in rows:
        print(f"{r['n']:3d} {r['grid']:5.0f} {r['mean_spdp_pct']:12.2f} "
              f"{r['std_spdp_pct']:7.2f} {r['mean_runtime_s']:10.2f} {r['n_seeds']:8d}")

    out_csv = os.path.join(OUT_DIR, "validation_table.csv")
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
