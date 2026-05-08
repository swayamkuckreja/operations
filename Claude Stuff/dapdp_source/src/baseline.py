"""
Truck-only PDP baseline.

Solves the same MILP as DAPDP but with all drone-sortie variables forced to
zero. This is the "PDP" benchmark used in the paper's large-scale
experiments (Section 5.2.3) to compute SPDP = 1 - z_DAPDP / z_PDP.

Implementation note: rather than re-deriving a separate PDP MILP, we just
disable drone usage by adding y == 0 constraints. This guarantees that any
cost difference is solely attributable to drone collaboration.
"""

from __future__ import annotations

from typing import Optional

from instance_generator import Instance
from dapdp_model import solve_dapdp, Solution


def solve_truck_only(inst: Instance,
                     time_limit: float = 600.0,
                     verbose: bool = False,
                     mip_gap_tol: float = 1e-4) -> Solution:
    """Return the optimal truck-only solution by replacing C' with empty set.

    Setting drone_eligible to [] eliminates every y variable from P, so the
    DAPDP MILP collapses to a pure pickup-and-delivery routing problem with
    a single capacitated truck.
    """
    # Clone instance with empty drone_eligible list
    truck_only_inst = Instance(
        name=inst.name + "_PDP",
        n=inst.n,
        coords=inst.coords,
        demands=inst.demands,
        drone_eligible=[],
        grid_size=inst.grid_size,
        seed=inst.seed,
        truck_speed=inst.truck_speed,
        drone_speed=inst.drone_speed,
        launch_time=inst.launch_time,
        recovery_time=inst.recovery_time,
        endurance=inst.endurance,
        truck_capacity=inst.truck_capacity,
        drone_capacity=inst.drone_capacity,
        fuel_price=inst.fuel_price,
        fuel_consumption=inst.fuel_consumption,
        miles_to_km=inst.miles_to_km,
        drone_cost_factor=inst.drone_cost_factor,
        max_route_duration=inst.max_route_duration,
        service_time_truck=inst.service_time_truck,
        service_time_drone=inst.service_time_drone,
        num_trucks=inst.num_trucks,
    )
    return solve_dapdp(truck_only_inst, time_limit=time_limit,
                       verbose=verbose, mip_gap_tol=mip_gap_tol,
                       use_valid_inequalities=True)


if __name__ == "__main__":
    from instance_generator import generate_instance, summary
    inst = generate_instance(n=3, grid_size=5.0, seed=1)
    print(summary(inst))
    print()
    sol = solve_truck_only(inst, time_limit=60.0)
    print(f"Truck-only objective: ${sol.objective:.4f}  (status: {sol.status})")
