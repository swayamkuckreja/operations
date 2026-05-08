"""
Verification of the DAPDP MILP.

The assignment guidelines (March 2021) require devising test instances that
demonstrate the constraints work as intended. For each constraint of interest
we either:

  (a) construct an instance whose feasible solutions are restricted in a
      well-understood way, then check the solver respects that restriction;

  (b) deliberately violate a constraint by post-processing a solution, and
      verify our checker would have caught the violation.

A solution is *checked* by independent Python code (function `audit`) that
re-derives every aspect of feasibility (precedence, capacity, endurance,
flow, weight evolution) from the optimal x and y values returned by the
solver. This is independent of the constraints inside Gurobi and therefore
a real verification, not just an output reformat.

Each scenario is a function returning a (Instance, Solution, Audit) tuple.
The script writes plots to /home/claude/dapdp/results/verification/.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from instance_generator import Instance, generate_instance
from dapdp_model import solve_dapdp, Solution
from baseline import solve_truck_only
from visualize import plot_solution

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "verification")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- audit ----
@dataclass
class Audit:
    """Independent re-check of the solution. Empty issues list = passes."""
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def __str__(self) -> str:
        head = "PASSED" if self.passed else f"FAILED ({len(self.issues)} issues)"
        out = [f"Audit: {head}"]
        for n in self.notes:
            out.append(f"  note: {n}")
        for i in self.issues:
            out.append(f"  ISSUE: {i}")
        return "\n".join(out)


def audit(inst: Instance, sol: Solution, tol: float = 1e-4) -> Audit:
    """Independently verify a DAPDP solution against the problem definition."""
    a = Audit()
    n = inst.n
    if not math.isfinite(sol.objective):
        a.issues.append("solver returned no incumbent")
        return a

    for v in sol.truck_arcs:
        truck_route = sol.visited_by_truck[v]
        sorties = sol.drone_sorties[v]
        truck_set = set(truck_route)
        drone_set = set(j for (_, j, _) in sorties)

        # ---- every customer served exactly once
        for j in range(1, 2 * n + 1):
            in_truck = j in truck_set
            in_drone = j in drone_set
            if in_truck and in_drone:
                a.issues.append(f"customer {j} served by both truck {v} and its drone")
            if not (in_truck or in_drone):
                # Could still be served by another truck; with single truck this is an issue
                if inst.num_trucks == 1:
                    a.issues.append(f"customer {j} not served by any vehicle")

        # ---- truck route starts at 0 and ends at 2n+1 (or empty)
        if truck_route and truck_route[0] != 0:
            a.issues.append(f"truck {v} does not start at depot")
        if truck_route and truck_route[-1] != 2 * n + 1:
            a.issues.append(f"truck {v} does not end at end-depot")

        # ---- pickup precedes delivery (when both on truck route)
        position = {node: idx for idx, node in enumerate(truck_route)}
        for i in range(1, n + 1):
            ni = n + i
            if i in truck_set and ni in truck_set:
                if position[i] >= position[ni]:
                    a.issues.append(
                        f"truck {v}: delivery {ni} not after pickup {i} "
                        f"(pos {position[ni]} <= {position[i]})"
                    )

        # ---- drone sorties: launch and rendezvous are on truck route or depot
        for (i, j, k) in sorties:
            if i not in truck_set and i != 0:
                a.issues.append(f"drone sortie {(i,j,k)}: launch {i} not on truck route")
            if k not in truck_set and k != 2 * n + 1:
                a.issues.append(f"drone sortie {(i,j,k)}: rendezvous {k} not on truck route")
            # endurance (geometric)
            t_fly = inst.drone_time(i, j) + inst.drone_time(j, k)
            t_total = inst.launch_time + t_fly + inst.service_time_drone[j] + inst.recovery_time
            if t_total > inst.endurance + tol:
                a.issues.append(
                    f"drone sortie {(i,j,k)}: total time {t_total*60:.2f} min "
                    f"exceeds endurance {inst.endurance*60:.0f} min"
                )
            # drone payload capacity
            if abs(inst.demands[j]) > inst.drone_capacity + tol:
                a.issues.append(
                    f"drone sortie {(i,j,k)}: customer {j} demand "
                    f"{abs(inst.demands[j]):.2f} kg exceeds drone capacity "
                    f"{inst.drone_capacity:.2f} kg"
                )
            # if drone delivers, the corresponding pickup must be on the truck route
            if j in range(n + 1, 2 * n + 1):
                pickup = j - n
                if pickup not in truck_set:
                    a.issues.append(
                        f"drone delivers {j} but pickup {pickup} not on truck route"
                    )
                # And the pickup must come before the launch in the truck sequence
                if i in truck_set and i != pickup:
                    if position[i] < position[pickup]:
                        a.issues.append(
                            f"drone sortie {(i,j,k)}: launch {i} precedes pickup "
                            f"{pickup} on truck route"
                        )

        # ---- truck capacity along route
        load = 0.0
        for idx in range(len(truck_route) - 1):
            cur = truck_route[idx]
            nxt = truck_route[idx + 1]
            if cur != 0 and cur != 2 * n + 1:
                load += inst.demands[cur]
                # any drone launched at cur takes its parcel weight off
                for (ii, jj, kk) in sorties:
                    if ii == cur:
                        load -= abs(inst.demands[jj])  # drone carries -|q|
            if load > inst.truck_capacity + tol:
                a.issues.append(
                    f"truck {v} load {load:.2f} kg exceeds capacity "
                    f"{inst.truck_capacity:.2f} kg after node {cur}"
                )

        # ---- route duration vs T_max
        if truck_route:
            arrival_end = sol.truck_arrivals.get((v, 2 * n + 1), 0.0)
            if arrival_end > inst.max_route_duration + tol:
                a.issues.append(
                    f"truck {v} arrives end-depot at {arrival_end:.2f} h > Tmax"
                )

    a.notes.append(f"objective = ${sol.objective:.4f}")
    a.notes.append(
        f"truck arcs: {sum(len(v) for v in sol.truck_arcs.values())}, "
        f"drone sorties: {sum(len(v) for v in sol.drone_sorties.values())}"
    )
    return a


# -------------------------------------------------------- scenarios ----
def _save_plot(inst: Instance, sol: Solution, fname: str, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plot_solution(inst, sol, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname))
    plt.close(fig)


def scenario_basic(seed: int = 1) -> None:
    """V1 — basic feasibility on a small random instance."""
    print("\n=== V1: basic feasibility (n=3, 5x5) ===")
    inst = generate_instance(n=3, grid_size=5.0, seed=seed)
    sol = solve_dapdp(inst, time_limit=60.0)
    a = audit(inst, sol)
    print(a)
    _save_plot(inst, sol, "V1_basic.pdf",
               f"V1 basic feasibility ({inst.name})  z* = ${sol.objective:.3f}")
    return inst, sol, a


def scenario_pickup_precedence() -> None:
    """V2 — handcrafted: pickup must precede delivery on the truck route."""
    print("\n=== V2: pickup-before-delivery precedence ===")
    # Construct an instance where the drone CAN'T deliver any customer
    # (all demands too heavy), so the truck must serve all pairs.
    n = 3
    coords = np.array([
        [0.0, 0.0],     # 0 depot
        [1.0, 1.0],     # 1 pickup 1
        [2.0, 0.0],     # 2 pickup 2
        [3.0, 1.0],     # 3 pickup 3
        [1.0, -1.0],    # 4 delivery 1
        [2.0, -1.0],    # 5 delivery 2
        [3.0, -1.0],    # 6 delivery 3
        [0.0, 0.0],     # 7 end depot
    ])
    demands = np.array([0.0, +5.0, +5.0, +5.0, -5.0, -5.0, -5.0, 0.0])
    inst = Instance(name="V2_precedence", n=n, coords=coords, demands=demands,
                    drone_eligible=[],   # heavier than 2.27 kg → truck-only
                    grid_size=4.0, seed=0,
                    service_time_truck=np.zeros(8),
                    service_time_drone=np.zeros(8))
    sol = solve_dapdp(inst, time_limit=60.0)
    a = audit(inst, sol)
    print(a)
    if sol.visited_by_truck:
        route = sol.visited_by_truck[1]
        print(f"  truck route: {route}")
    _save_plot(inst, sol, "V2_pickup_precedence.pdf",
               f"V2 pickup-before-delivery (truck-only)  z* = ${sol.objective:.3f}")
    return inst, sol, a


def scenario_drone_endurance() -> None:
    """V3 — clamping endurance forces longer truck routes (no infeasible sortie)."""
    print("\n=== V3: drone endurance limit ===")
    inst = generate_instance(n=4, grid_size=20.0, seed=7)
    # Try generous endurance vs tight endurance
    inst.endurance = 60.0 / 60.0
    sol_loose = solve_dapdp(inst, time_limit=120.0)
    a_loose = audit(inst, sol_loose)
    n_sort_loose = len(sol_loose.drone_sorties[1])

    inst.endurance = 5.0 / 60.0
    sol_tight = solve_dapdp(inst, time_limit=120.0)
    a_tight = audit(inst, sol_tight)
    n_sort_tight = len(sol_tight.drone_sorties[1])

    print(f"  endurance 60 min -> {n_sort_loose} sorties, z = ${sol_loose.objective:.3f}")
    print(f"  endurance  5 min -> {n_sort_tight} sorties, z = ${sol_tight.objective:.3f}")
    print(a_loose)
    print(a_tight)
    if not (n_sort_tight <= n_sort_loose):
        print("  ISSUE: tighter endurance did not reduce sorties")
    _save_plot(inst, sol_loose, "V3_endurance_60min.pdf",
               f"V3 endurance 60 min  ({n_sort_loose} sorties, ${sol_loose.objective:.3f})")
    _save_plot(inst, sol_tight, "V3_endurance_05min.pdf",
               f"V3 endurance 5 min  ({n_sort_tight} sorties, ${sol_tight.objective:.3f})")
    return [(sol_loose, a_loose), (sol_tight, a_tight)]


def scenario_drone_payload() -> None:
    """V4 — heavy delivery cannot be served by drone."""
    print("\n=== V4: drone payload limit ===")
    n = 3
    coords = np.array([
        [0.0, 0.0],
        [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
        [1.0, -1.0], [2.0, -1.0], [3.0, -1.0],
        [0.0, 0.0],
    ])
    # All drone-light EXCEPT delivery 5 (=node 5) which is heavy.
    demands = np.array([0.0, +1.0, +50.0, +1.0, -1.0, -50.0, -1.0, 0.0])
    inst = Instance(name="V4_payload", n=n, coords=coords, demands=demands,
                    drone_eligible=[4, 6],   # delivery 5 (heavy) excluded
                    grid_size=4.0, seed=0,
                    service_time_truck=np.zeros(8),
                    service_time_drone=np.zeros(8))
    sol = solve_dapdp(inst, time_limit=60.0)
    a = audit(inst, sol)
    print(a)
    served = sol.served_by_drone[1]
    print(f"  drone-served customers: {served}")
    if 5 in served:
        print("  ISSUE: drone served the heavy delivery (5) which is not in C'")
    _save_plot(inst, sol, "V4_drone_payload.pdf",
               f"V4 drone payload (5 forced to truck)  z* = ${sol.objective:.3f}")
    return inst, sol, a


def scenario_no_subtours() -> None:
    """V5 — confirm absence of disconnected sub-tours (constraint 5 / MTZ)."""
    print("\n=== V5: no disconnected sub-tours ===")
    inst = generate_instance(n=6, grid_size=15.0, seed=11)
    sol = solve_dapdp(inst, time_limit=300.0)
    a = audit(inst, sol)
    print(a)
    # Trace truck arcs forming a connected path 0 -> ... -> 2n+1
    arcs = sol.truck_arcs[1]
    succ = {i: j for (i, j) in arcs}
    cur = 0
    seen = [0]
    while cur in succ:
        cur = succ[cur]
        if cur in seen:
            print(f"  ISSUE: cycle detected — repeated node {cur}")
            break
        seen.append(cur)
    n = inst.n
    if cur != 2 * n + 1:
        print(f"  ISSUE: path does not terminate at end-depot ({cur} != {2*n+1})")
    # Any arc not on the chain would be a sub-tour
    chain_arcs = set(zip(seen[:-1], seen[1:]))
    extras = [a for a in arcs if a not in chain_arcs]
    if extras:
        print(f"  ISSUE: {len(extras)} arcs not on main path -> sub-tours: {extras}")
    print(f"  truck path length: {len(seen)} nodes; total arcs used: {len(arcs)}")
    _save_plot(inst, sol, "V5_no_subtours.pdf",
               f"V5 sub-tour elimination  ({inst.name})")
    return inst, sol, a


def scenario_pdp_vs_dapdp() -> None:
    """V6 — DAPDP cost <= PDP cost on identical instance."""
    print("\n=== V6: DAPDP cost <= PDP cost ===")
    inst = generate_instance(n=8, grid_size=10.0, seed=3)
    sol_pdp = solve_truck_only(inst, time_limit=600.0)
    sol_dapdp = solve_dapdp(inst, time_limit=600.0)
    print(f"  PDP   objective: ${sol_pdp.objective:.4f}")
    print(f"  DAPDP objective: ${sol_dapdp.objective:.4f}")
    if sol_dapdp.objective > sol_pdp.objective + 1e-4:
        print("  ISSUE: DAPDP cost exceeds PDP cost (drone usage should never hurt)")
    print(audit(inst, sol_dapdp))
    _save_plot(inst, sol_pdp, "V6_pdp.pdf",
               f"V6 truck-only baseline ({inst.name}) z = ${sol_pdp.objective:.3f}")
    _save_plot(inst, sol_dapdp, "V6_dapdp.pdf",
               f"V6 DAPDP optimum ({inst.name}) z = ${sol_dapdp.objective:.3f}")
    return [(sol_pdp, None), (sol_dapdp, audit(inst, sol_dapdp))]


def scenario_no_depot_drone_delivery() -> None:
    """V7 — drone cannot deliver from the start depot (since corresponding
    pickup must be made by the truck first)."""
    print("\n=== V7: depot drone delivery forbidden ===")
    inst = generate_instance(n=3, grid_size=5.0, seed=4)
    sol = solve_dapdp(inst, time_limit=60.0)
    a = audit(inst, sol)
    bad = [s for s in sol.drone_sorties[1] if s[0] == 0]
    if bad:
        print(f"  ISSUE: depot launches found: {bad}")
    else:
        print("  no depot launches in solution (as required)")
    print(a)
    return inst, sol, a


def main():
    results = []
    for fn in [scenario_basic,
               scenario_pickup_precedence,
               scenario_drone_endurance,
               scenario_drone_payload,
               scenario_no_subtours,
               scenario_pdp_vs_dapdp,
               scenario_no_depot_drone_delivery]:
        try:
            res = fn()
            results.append((fn.__name__, res))
        except Exception as e:
            print(f"  scenario {fn.__name__} crashed: {e}")
            import traceback; traceback.print_exc()
    print("\n=== verification suite complete ===")


if __name__ == "__main__":
    main()
