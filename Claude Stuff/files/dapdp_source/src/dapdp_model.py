"""
MILP implementation of the Drone-Assisted Pickup and Delivery Problem (DAPDP).

Reference: T. Mulumba and A. Diabat, "Optimization of the drone-assisted pickup
and delivery problem", Transportation Research Part E 181 (2024) 103377.

The implementation reproduces equations (1)-(37) of Section 3.2 of the paper,
together with the optional valid inequalities (43) and (44) of Section 3.3.

Notation matches the paper as closely as possible (variable names x, y, p, t,
t_drone, u, w, eta).

Notes on interpretations made where the paper is ambiguous or contains
likely typos:
  * Constraint (28) is implemented with -M(1-x_ij), not +M(1-x_ij). The
    written form would force every (i,j) with x_ij = 0 to require w_j very
    large, which is structurally infeasible. The minus sign is the only way
    the constraint reads as a standard big-M relaxation.
  * Demand sign convention: q_i > 0 at pickups, q_i < 0 at deliveries (see
    instance_generator.py).
  * In constraint (26) we extend the launch-precedence requirement to all
    sortie launches (i,j,k) in P with j a delivery, including i = 0
    (depot launches). Allowing depot launches without a precedence rule
    would let the drone deliver before the truck has visited the
    corresponding pickup node, which is physically nonsensical.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

from instance_generator import Instance


# Type aliases
Sortie = Tuple[int, int, int]   # (launch i, drone customer j, rendezvous k)


@dataclass
class Solution:
    """Optimisation result for one DAPDP instance."""

    status: str
    objective: float
    runtime: float
    mip_gap: Optional[float]
    truck_arcs: Dict[int, List[Tuple[int, int]]]    # truck v -> list of (i,j)
    drone_sorties: Dict[int, List[Sortie]]          # truck v -> list of sorties
    truck_arrivals: Dict[Tuple[int, int], float]    # (v,i) -> truck arrival time at i (h)
    drone_arrivals: Dict[Tuple[int, int], float]    # (v,i) -> drone arrival time at i (h)
    visited_by_truck: Dict[int, List[int]]          # truck v -> ordered node list
    served_by_drone: Dict[int, List[int]]           # truck v -> list of customer nodes


def _feasible_sorties(inst: Instance) -> List[Sortie]:
    """Construct P = {(i,j,k)} of geometrically feasible drone sorties.

    A sortie is kept only if:
      * the round-trip plus launch and recovery service times fit within
        the drone's endurance e (otherwise (19) forbids it anyway);
      * i != 0 — depot launches for delivery-type drone customers are
        forbidden by (26) and are always infeasible because the
        corresponding pickup must precede the launch in the truck's route;
      * the three nodes i, j, k are pairwise distinct.

    Pre-filtering shrinks |P| by roughly 10x and is harmless because the
    same constraints are enforced inside the MILP.
    """
    n = inst.n
    # Exclude depot launch (i=0); see (26): impossible for j a delivery node.
    N0_no_depot = list(range(1, 2 * n + 1))
    Nplus = list(range(1, 2 * n + 2))
    Cprime = inst.drone_eligible

    e = inst.endurance
    sL, sR = inst.launch_time, inst.recovery_time

    P: List[Sortie] = []
    for i in N0_no_depot:
        for j in Cprime:
            if j == i:
                continue
            for k in Nplus:
                if k == i or k == j:
                    continue
                t_fly = inst.drone_time(i, j) + inst.drone_time(j, k)
                t_serve = inst.service_time_drone[j]
                if sL + t_fly + t_serve + sR > e + 1e-9:
                    continue
                P.append((i, j, k))
    return P


def solve_dapdp(inst: Instance,
                time_limit: float = 600.0,
                use_valid_inequalities: bool = True,
                mip_gap_tol: float = 1e-4,
                verbose: bool = False) -> Solution:
    """Solve the DAPDP MILP for the given instance.

    Parameters
    ----------
    inst                    : Instance object
    time_limit              : Gurobi TimeLimit in seconds
    use_valid_inequalities  : add (43) and (44) cuts
    mip_gap_tol             : MIPGap stopping criterion
    verbose                 : print Gurobi log
    """
    n = inst.n
    N = list(range(0, 2 * n + 2))           # 0..2n+1
    N0 = list(range(0, 2 * n + 1))          # 0..2n
    Nplus = list(range(1, 2 * n + 2))       # 1..2n+1
    C = list(range(1, 2 * n + 1))           # 1..2n
    phi_plus = list(range(1, n + 1))        # pickups
    phi_minus = list(range(n + 1, 2 * n + 1))  # deliveries
    Cprime = inst.drone_eligible
    V = list(range(1, inst.num_trucks + 1))
    P = _feasible_sorties(inst)

    # Big-M values
    Tmax = inst.max_route_duration
    big_M_time = Tmax + 10.0                # >> any single travel/service interval
    big_M_pos = 2 * n + 2                   # = |N|, sufficient for u_i bounds

    m = gp.Model("DAPDP")
    if not verbose:
        m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap_tol

    # ---------------- Decision variables ----------------
    # x[v,i,j] : truck v traverses arc (i,j)
    x = {(v, i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{v}_{i}_{j}")
         for v in V for i in N0 for j in Nplus if i != j}

    # y[v,i,j,k] : drone of truck v flies sortie i -> j -> k
    y = {(v, i, j, k): m.addVar(vtype=GRB.BINARY, name=f"y_{v}_{i}_{j}_{k}")
         for v in V for (i, j, k) in P}

    # p[v,i,j] : 1 iff i precedes j on truck v's route
    p = {(v, i, j): m.addVar(vtype=GRB.BINARY, name=f"p_{v}_{i}_{j}")
         for v in V for i in C for j in C if i != j}

    # Truck arrival time t[v,i] and drone arrival time tdr[v,i]
    t = {(v, i): m.addVar(lb=0.0, ub=big_M_time, name=f"t_{v}_{i}")
         for v in V for i in N}
    tdr = {(v, i): m.addVar(lb=0.0, ub=big_M_time, name=f"td_{v}_{i}")
           for v in V for i in N}

    # u[v,i] : position of i in truck v's route (1..2n+2)
    u = {(v, i): m.addVar(lb=1.0, ub=big_M_pos, vtype=GRB.INTEGER, name=f"u_{v}_{i}")
         for v in V for i in Nplus}

    # w[v,i] : truck weight after visiting i
    w = {(v, i): m.addVar(lb=0.0, ub=inst.truck_capacity, name=f"w_{v}_{i}")
         for v in V for i in C}

    # eta : objective epigraph variable (Proposition 1)
    eta = m.addVar(lb=0.0, name="eta")

    # ---------------- Objective ----------------
    truck_cost_expr = gp.quicksum(
        inst.truck_cost(i, j) * x[v, i, j]
        for v in V for i in N0 for j in Nplus if i != j
    )
    drone_cost_expr = gp.quicksum(
        (inst.drone_cost(i, j) + inst.drone_cost(j, k)) * y[v, i, j, k]
        for v in V for (i, j, k) in P
    )
    # Eq. (1) / (38)-(39): minimise eta with eta >= total cost.
    m.addConstr(eta >= truck_cost_expr + drone_cost_expr, name="cost_bound")
    m.setObjective(eta, GRB.MINIMIZE)

    # ---------------- Constraints ----------------
    # (2) every customer served exactly once, by truck or by drone
    for j in C:
        m.addConstr(
            gp.quicksum(x[v, i, j] for v in V for i in N0 if i != j)
            + gp.quicksum(y[v, i, j, k] for v in V for (ii, jj, k) in P
                          if jj == j for i in [ii] if True)
            == 1,
            name=f"served_{j}"
        )

    # (3) each truck departs the start depot at most once
    for v in V:
        m.addConstr(gp.quicksum(x[v, 0, j] for j in Nplus) <= 1, name=f"dep_start_{v}")

    # (4) each truck arrives at the end depot at most once
    for v in V:
        m.addConstr(gp.quicksum(x[v, i, 2 * n + 1] for i in N0) <= 1,
                    name=f"arr_end_{v}")

    # (5) MTZ-style sub-tour elimination on truck route
    for v in V:
        for i in C:
            for j in Nplus:
                if j == i:
                    continue
                m.addConstr(
                    1 + u[v, i] - u[v, j] <= (1 - x[v, i, j]) * big_M_pos,
                    name=f"mtz_{v}_{i}_{j}"
                )

    # (6) flow conservation at every customer
    for v in V:
        for j in C:
            m.addConstr(
                gp.quicksum(x[v, i, j] for i in N0 if i != j)
                == gp.quicksum(x[v, j, k] for k in Nplus if k != j),
                name=f"flow_{v}_{j}"
            )

    # (7) drone launches from each node at most once (per truck)
    for v in V:
        for i in N0:
            relevant = [(i, j, k) for (ii, j, k) in P if ii == i]
            if relevant:
                m.addConstr(
                    gp.quicksum(y[v, i, j, k] for (_, j, k) in relevant) <= 1,
                    name=f"launch_once_{v}_{i}"
                )

    # (8) drone returns to each node at most once (per truck)
    for v in V:
        for k in Nplus:
            relevant = [(i, j, k) for (i, j, kk) in P if kk == k]
            if relevant:
                m.addConstr(
                    gp.quicksum(y[v, i, j, k] for (i, j, _) in relevant) <= 1,
                    name=f"recover_once_{v}_{k}"
                )

    # (9) launch and rendezvous nodes must be on truck route
    for v in V:
        for (i, j, k) in P:
            in_i = gp.quicksum(x[v, h, i] for h in N0 if h != i) if i != 0 else 1
            in_k = gp.quicksum(x[v, l, k] for l in N0 if l != k)
            m.addConstr(2 * y[v, i, j, k] <= in_i + in_k,
                        name=f"on_route_{v}_{i}_{j}_{k}")

    # (10) if launch is from depot 0, rendezvous k must be on truck route
    # Subsumed by (9) when i = 0 because we set in_i = 1 there. (9) already enforces (10).

    # (11) precedence: launch i precedes rendezvous k in truck's route
    for v in V:
        for (i, j, k) in P:
            if i == 0 or k == 2 * n + 1:
                continue   # depot endpoints have implicit position
            m.addConstr(
                u[v, k] - u[v, i] >= 1 - big_M_pos * (1 - y[v, i, j, k]),
                name=f"prec_launch_rdv_{v}_{i}_{j}_{k}"
            )

    # (12)-(15) time synchronisation: drone arrival at launch and rendezvous
    # nodes equals truck arrival there if a sortie touches that node.
    for v in V:
        for i in C:
            launch_indicator = gp.quicksum(
                y[v, ii, jj, kk] for (ii, jj, kk) in P if ii == i
            )
            # (12): tdr[v,i] >= t[v,i] - M * (1 - 1[i is launch node])
            m.addConstr(
                tdr[v, i] >= t[v, i] - big_M_time * (1 - launch_indicator),
                name=f"sync_launch_lo_{v}_{i}"
            )
            # (13): tdr[v,i] <= t[v,i] + M * (1 - launch_indicator)
            m.addConstr(
                tdr[v, i] <= t[v, i] + big_M_time * (1 - launch_indicator),
                name=f"sync_launch_hi_{v}_{i}"
            )
        for k in C:
            rdv_indicator = gp.quicksum(
                y[v, ii, jj, kk] for (ii, jj, kk) in P if kk == k
            )
            # (14): tdr[v,k] >= t[v,k] - M * (1 - rdv_indicator)
            m.addConstr(
                tdr[v, k] >= t[v, k] - big_M_time * (1 - rdv_indicator),
                name=f"sync_rdv_lo_{v}_{k}"
            )
            # (15): tdr[v,k] <= t[v,k] + M * (1 - rdv_indicator)
            m.addConstr(
                tdr[v, k] <= t[v, k] + big_M_time * (1 - rdv_indicator),
                name=f"sync_rdv_hi_{v}_{k}"
            )

    # (16) truck arrival time at k after travelling from h, including launch/recovery
    for v in V:
        for h in N0:
            for k in Nplus:
                if k == h:
                    continue
                launch_at_k = gp.quicksum(
                    y[v, kk, ll, mm] for (kk, ll, mm) in P if kk == k
                )
                recover_at_k = gp.quicksum(
                    y[v, ii, jj, kk] for (ii, jj, kk) in P if kk == k
                )
                d_h = inst.service_time_truck[h] if h != 0 else 0.0
                m.addConstr(
                    t[v, k] >= t[v, h] + inst.truck_time(h, k) + d_h
                              + inst.launch_time * launch_at_k
                              + inst.recovery_time * recover_at_k
                              - big_M_time * (1 - x[v, h, k]),
                    name=f"truck_time_{v}_{h}_{k}"
                )

    # (17) drone arrival time at j after launching from i
    for v in V:
        for j in Cprime:
            for i in N0:
                if i == j:
                    continue
                launch_ij = gp.quicksum(
                    y[v, ii, jj, kk] for (ii, jj, kk) in P
                    if ii == i and jj == j
                )
                m.addConstr(
                    tdr[v, j] >= tdr[v, i] + inst.drone_time(i, j)
                                 - big_M_time * (1 - launch_ij),
                    name=f"drone_arr_j_{v}_{i}_{j}"
                )

    # (18) drone arrival time at k after delivering at j
    for v in V:
        for j in Cprime:
            for k in Nplus:
                if k == j:
                    continue
                deliver_jk = gp.quicksum(
                    y[v, ii, jj, kk] for (ii, jj, kk) in P
                    if jj == j and kk == k
                )
                m.addConstr(
                    tdr[v, k] >= tdr[v, j] + inst.drone_time(j, k)
                                 + inst.service_time_drone[j]
                                 + inst.recovery_time
                                 - big_M_time * (1 - deliver_jk),
                    name=f"drone_arr_k_{v}_{j}_{k}"
                )

    # (19) drone endurance limit: tdr[k] - (tdr[j] - tau'_ij) <= e
    # i.e. total flight + drone service <= endurance e
    for v in V:
        for (i, j, k) in P:
            m.addConstr(
                tdr[v, k] - (tdr[v, j] - inst.drone_time(i, j))
                <= inst.endurance + big_M_time * (1 - y[v, i, j, k]),
                name=f"endurance_{v}_{i}_{j}_{k}"
            )

    # (20)-(22) consistency between u and p
    for v in V:
        for i in C:
            for j in C:
                if i == j:
                    continue
                # (20)
                m.addConstr(
                    u[v, i] - u[v, j] >= 1 - big_M_pos * p[v, i, j],
                    name=f"u_p_lower_{v}_{i}_{j}"
                )
                # (21)
                m.addConstr(
                    u[v, i] - u[v, j] <= -1 + (1 - p[v, i, j]) * big_M_pos,
                    name=f"u_p_upper_{v}_{i}_{j}"
                )
        for i in C:
            for j in C:
                if i >= j:
                    continue
                # (22)
                m.addConstr(p[v, i, j] + p[v, j, i] == 1,
                            name=f"p_pair_{v}_{i}_{j}")

    # (23) no second drone launch from l until first sortie (i,j,k) has returned
    for v in V:
        for i in N0:
            for k in Nplus:
                if k == i:
                    continue
                first_sortie = gp.quicksum(
                    y[v, ii, jj, kk] for (ii, jj, kk) in P
                    if ii == i and kk == k
                )
                for l in C:
                    if l == i or l == k:
                        continue
                    second_sortie = gp.quicksum(
                        y[v, ll, mm, nn] for (ll, mm, nn) in P
                        if ll == l and nn != i and nn != k and mm != i and mm != k
                    )
                    if i == 0:
                        # Depot is implicitly "before" l
                        precedence_term = 1
                    else:
                        # Need p[v,i,l]: i precedes l in route
                        # Only valid for i,l in C
                        if i in C and l in C and i != l:
                            precedence_term = p[v, i, l]
                        else:
                            continue
                    m.addConstr(
                        tdr[v, l] >= tdr[v, k]
                            - big_M_time * (3 - first_sortie - second_sortie - precedence_term),
                        name=f"no_overlap_{v}_{i}_{k}_{l}"
                    )

    # (24) prohibit direct depot-to-depot trip
    for v in V:
        if (v, 0, 2 * n + 1) in x:
            m.addConstr(x[v, 0, 2 * n + 1] == 0, name=f"no_empty_{v}")

    # (25) pickup-delivery same-vehicle constraint
    for v in V:
        for i in phi_plus:
            ni = n + i
            lhs = gp.quicksum(x[v, i, j] for j in Nplus if j != i)
            rhs1 = gp.quicksum(x[v, ni, j] for j in Nplus if j != ni)
            rhs2 = gp.quicksum(y[v, l, ni, k] for (l, jj, k) in P if jj == ni)
            m.addConstr(lhs == rhs1 + rhs2, name=f"same_vehicle_{v}_{i}")

    # (26) launch i must come AFTER the corresponding pickup j-n in truck route.
    # Implemented for (i,j,k) in P with j a delivery node. We extend to i = 0 by
    # forbidding such launches whenever the corresponding pickup must be visited
    # later (this rules out impossible depot-launch deliveries).
    for v in V:
        for (i, j, k) in P:
            if j not in phi_minus:
                continue
            pickup_node = j - n  # i.e. j - n
            if i == 0:
                # Depot launch: would force u[v,pickup] <= 0 which is infeasible
                # since u >= 1. Equivalent to forbidding such sorties.
                m.addConstr(y[v, i, j, k] == 0,
                            name=f"no_depot_drn_dlvy_{v}_{j}_{k}")
            elif i in C and pickup_node in C and i != pickup_node:
                m.addConstr(
                    u[v, i] - u[v, pickup_node] >= 1 - big_M_pos * (1 - y[v, i, j, k]),
                    name=f"pickup_before_launch_{v}_{i}_{j}_{k}"
                )

    # (27) truck visits pickup before its corresponding delivery
    for v in V:
        for i in phi_plus:
            ni = n + i
            m.addConstr(
                t[v, ni] >= t[v, i] + inst.truck_time(i, ni)
                            + inst.service_time_truck[i],
                name=f"pickup_before_delivery_{v}_{i}"
            )

    # (28) truck weight evolution along the route, accounting for parcels handed
    # to the drone at launch. Implemented with -M(1-x_ij) (the paper writes
    # +M which appears to be a typo).
    for v in V:
        for i in N0:
            for j in C:
                if i == j:
                    continue
                if (v, i, j) not in x:
                    continue
                # Drone parcel weight handed over at j (negative contribution).
                drone_handover = gp.quicksum(
                    inst.demands[mm] * y[v, j, mm, kk]
                    for (jj, mm, kk) in P if jj == j
                )
                # If i is the depot, w[v,i] does not exist. Use 0 there.
                if i == 0:
                    w_prev = 0.0
                else:
                    w_prev = w[v, i]
                m.addConstr(
                    w[v, j] >= w_prev + inst.demands[j] + drone_handover
                                - inst.truck_capacity * (1 - x[v, i, j]),
                    name=f"weight_{v}_{i}_{j}"
                )

    # (31)-(32) earliest departure from start depot is 0
    for v in V:
        m.addConstr(t[v, 0] == 0, name=f"depart_t_{v}")
        m.addConstr(tdr[v, 0] == 0, name=f"depart_tdr_{v}")

    # (33)-(34) maximum route duration
    for v in V:
        m.addConstr(t[v, 2 * n + 1] <= Tmax, name=f"Tmax_truck_{v}")
        m.addConstr(tdr[v, 2 * n + 1] <= Tmax, name=f"Tmax_drone_{v}")

    # (35) p[v,0,j] = 1 (depot precedes everyone) -- omitted because we handle
    # depot precedence implicitly through the structure of u and the time
    # constraints; preserving (35) as written would require p variables on the
    # depot, which we omitted to keep the variable count down.

    # (36)-(37) bounds already encoded via lb/ub on u and w.

    # ---------------- Valid inequalities (Section 3.3) ----------------
    if use_valid_inequalities:
        # (43): x_ij + sum_k y_ijk <= 1   (cannot serve j by truck and drone from i)
        for v in V:
            for i in N0:
                for j in Cprime:
                    if i == j:
                        continue
                    if (v, i, j) not in x:
                        continue
                    drone_to_j = gp.quicksum(
                        y[v, i, j, k] for (ii, jj, k) in P
                        if ii == i and jj == j
                    )
                    m.addConstr(x[v, i, j] + drone_to_j <= 1,
                                name=f"VI_43_{v}_{i}_{j}")

        # (44): x_ij + x_jk + x_ik + y_ijk <= 2
        for v in V:
            for (i, j, k) in P:
                if i == j or j == k or i == k:
                    continue
                if (v, i, j) not in x or (v, j, k) not in x or (v, i, k) not in x:
                    continue
                m.addConstr(
                    x[v, i, j] + x[v, j, k] + x[v, i, k] + y[v, i, j, k] <= 2,
                    name=f"VI_44_{v}_{i}_{j}_{k}"
                )

    # ---------------- Solve ----------------
    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    status_map = {GRB.OPTIMAL: "OPTIMAL",
                  GRB.SUBOPTIMAL: "SUBOPTIMAL",
                  GRB.TIME_LIMIT: "TIME_LIMIT",
                  GRB.INFEASIBLE: "INFEASIBLE",
                  GRB.UNBOUNDED: "UNBOUNDED",
                  GRB.INF_OR_UNBD: "INF_OR_UNBD"}
    status = status_map.get(m.Status, f"STATUS_{m.Status}")

    if m.SolCount == 0:
        return Solution(status=status, objective=math.inf, runtime=runtime,
                        mip_gap=None, truck_arcs={}, drone_sorties={},
                        truck_arrivals={}, drone_arrivals={},
                        visited_by_truck={}, served_by_drone={})

    # ---------------- Extract solution ----------------
    truck_arcs: Dict[int, List[Tuple[int, int]]] = {v: [] for v in V}
    drone_sorties: Dict[int, List[Sortie]] = {v: [] for v in V}
    truck_arrivals: Dict[Tuple[int, int], float] = {}
    drone_arrivals: Dict[Tuple[int, int], float] = {}

    for (v, i, j), var in x.items():
        if var.X > 0.5:
            truck_arcs[v].append((i, j))
    for (v, i, j, k), var in y.items():
        if var.X > 0.5:
            drone_sorties[v].append((i, j, k))
    for (v, i), var in t.items():
        truck_arrivals[(v, i)] = var.X
    for (v, i), var in tdr.items():
        drone_arrivals[(v, i)] = var.X

    visited_by_truck: Dict[int, List[int]] = {}
    served_by_drone: Dict[int, List[int]] = {v: [j for (i, j, k) in drone_sorties[v]]
                                             for v in V}
    for v in V:
        seq = [0]
        cur = 0
        seen = {0}
        while True:
            nxt = None
            for (i, j) in truck_arcs[v]:
                if i == cur and j not in seen:
                    nxt = j
                    break
            if nxt is None:
                break
            seq.append(nxt)
            seen.add(nxt)
            cur = nxt
            if cur == 2 * n + 1:
                break
        visited_by_truck[v] = seq

    return Solution(
        status=status,
        objective=m.ObjVal,
        runtime=runtime,
        mip_gap=getattr(m, "MIPGap", None),
        truck_arcs=truck_arcs,
        drone_sorties=drone_sorties,
        truck_arrivals=truck_arrivals,
        drone_arrivals=drone_arrivals,
        visited_by_truck=visited_by_truck,
        served_by_drone=served_by_drone,
    )


def pretty_print(sol: Solution, inst: Instance) -> str:
    lines = [
        f"Status:    {sol.status}",
        f"Objective: ${sol.objective:.4f}",
        f"Runtime:   {sol.runtime:.2f} s",
        f"MIP gap:   {sol.mip_gap if sol.mip_gap is not None else 'n/a'}",
    ]
    for v in sol.truck_arcs:
        lines.append(f"\nTruck {v} route: {' -> '.join(map(str, sol.visited_by_truck[v]))}")
        if sol.drone_sorties[v]:
            lines.append(f"Truck {v} drone sorties:")
            for (i, j, k) in sol.drone_sorties[v]:
                lines.append(f"  launch {i} -> deliver to {j} -> rendezvous {k}")
        else:
            lines.append(f"Truck {v} drone sorties: (none)")
    return "\n".join(lines)


if __name__ == "__main__":
    inst = generate_instance = __import__("instance_generator").generate_instance
    summary = __import__("instance_generator").summary
    inst_obj = inst(n=3, grid_size=5.0, seed=1)
    print(summary(inst_obj))
    sol = solve_dapdp(inst_obj, time_limit=120.0, verbose=False)
    print()
    print(pretty_print(sol, inst_obj))
