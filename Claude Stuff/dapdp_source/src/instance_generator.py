"""
Instance generator for the Drone-Assisted Pickup and Delivery Problem (DAPDP).

Follows the protocol described in Mulumba & Diabat (2024), Section 5.2:
  - n pickup requests => 2n customer nodes (n pickups + n deliveries) on a d x d grid
  - Coordinates U(0, d) for each customer
  - Depot at the centroid of customer coordinates (start = node 0, end = node 2n+1)
  - Demand at delivery node: 86% chance U(0, 2.27) kg (drone-eligible),
                             14% chance U(2.27, 68) kg (truck-only)
  - Pickup demand is the negative of the corresponding delivery demand
  - A delivery node is drone-eligible iff its demand <= drone capacity Q'

Author: Generated for AE4441 Operations Optimisation assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class Instance:
    """Container for a single DAPDP instance.

    Node indexing convention (matches paper):
      0       : start depot
      1..n    : pickup nodes (phi+)
      n+1..2n : delivery nodes (phi-); delivery for pickup i is at node n+i
      2n+1    : end depot (same physical location as start depot)
    """

    name: str
    n: int                                # number of pickup requests
    coords: np.ndarray                    # shape (2n+2, 2): x,y for every node
    demands: np.ndarray                   # shape (2n+2,): kg, negative on pickups, positive on deliveries
    drone_eligible: List[int]             # subset of delivery node indices the drone may serve (C')
    grid_size: float
    seed: int

    # Operational parameters (defaults follow Table 1 of the paper)
    truck_speed: float = 35.0             # mph
    drone_speed: float = 50.0             # mph
    launch_time: float = 1.0 / 60.0       # hours (1 min)
    recovery_time: float = 1.0 / 60.0     # hours (1 min)
    endurance: float = 30.0 / 60.0        # hours (30 min)
    truck_capacity: float = 1300.0        # kg (with drone on board)
    drone_capacity: float = 2.27          # kg
    fuel_price: float = 0.840             # $/L
    fuel_consumption: float = 0.07        # L/km
    miles_to_km: float = 1.61
    drone_cost_factor: float = 0.10       # alpha
    max_route_duration: float = 8.0       # hours
    service_time_truck: np.ndarray = field(default_factory=lambda: np.array([]))   # per node, hours
    service_time_drone: np.ndarray = field(default_factory=lambda: np.array([]))   # per node, hours
    num_trucks: int = 1                   # |V|

    @property
    def num_nodes(self) -> int:
        return 2 * self.n + 2

    def euclidean_distance(self, i: int, j: int) -> float:
        """Distance between two nodes in miles (coords are stored in miles)."""
        return float(np.linalg.norm(self.coords[i] - self.coords[j]))

    def truck_time(self, i: int, j: int) -> float:
        """Truck travel time in hours."""
        return self.euclidean_distance(i, j) / self.truck_speed

    def drone_time(self, i: int, j: int) -> float:
        """Drone travel time in hours (Euclidean)."""
        return self.euclidean_distance(i, j) / self.drone_speed

    def truck_cost(self, i: int, j: int) -> float:
        """Truck travel cost in $: c_ij = f_p * f_c * d_ij_in_km."""
        return self.fuel_price * self.fuel_consumption * \
               self.euclidean_distance(i, j) * self.miles_to_km

    def drone_cost(self, i: int, j: int) -> float:
        """Drone travel cost in $: c'_ij = alpha * c_ij (paper's convention)."""
        return self.drone_cost_factor * self.truck_cost(i, j)


def generate_instance(n: int,
                      grid_size: float,
                      seed: int = 0,
                      name: str | None = None,
                      num_trucks: int = 1) -> Instance:
    """Generate a random DAPDP instance.

    Parameters
    ----------
    n           : number of pickup requests (=> 2n customers + 2 depot copies)
    grid_size   : side length d of the square service area in miles
    seed        : RNG seed for reproducibility
    name        : optional instance name (default "n.d.seed")
    num_trucks  : number of homogeneous trucks |V|
    """
    rng = np.random.default_rng(seed)

    # Customer coordinates
    cust_coords = rng.uniform(0.0, grid_size, size=(2 * n, 2))
    # Depot at centroid of customers
    depot = cust_coords.mean(axis=0)

    # Full coordinate array: index 0 = start depot, 2n+1 = end depot (same place)
    coords = np.vstack([depot, cust_coords, depot])

    # Demand generation: 86% drone-eligible (<=2.27 kg), 14% heavier.
    # Sign convention used here matches constraint (28) of the MILP:
    #   q > 0 at pickup nodes  (truck load INCREASES on arrival)
    #   q < 0 at delivery nodes (truck load DECREASES on arrival)
    # NB: this is the opposite of the textual description in §5.2 of the
    # paper. We keep it consistent with the MILP itself, which is what the
    # solver actually sees.
    demands = np.zeros(2 * n + 2)
    drone_eligible: List[int] = []
    for i in range(1, n + 1):
        delivery_node = n + i
        if rng.random() < 0.86:
            q = rng.uniform(0.0, 2.27)
            drone_eligible.append(delivery_node)
        else:
            q = rng.uniform(2.27, 68.0)
        demands[i] = +q              # pickup: load gained
        demands[delivery_node] = -q  # delivery: load shed

    # Service times: U[1,6] min for truck, U[1,3] min for drone, depot has 0
    svc_truck = rng.uniform(1, 6, size=2 * n + 2) / 60.0   # to hours
    svc_drone = rng.uniform(1, 3, size=2 * n + 2) / 60.0
    svc_truck[0] = svc_truck[2 * n + 1] = 0.0
    svc_drone[0] = svc_drone[2 * n + 1] = 0.0

    if name is None:
        name = f"{2*n}.{int(grid_size)}.{seed}"

    return Instance(
        name=name,
        n=n,
        coords=coords,
        demands=demands,
        drone_eligible=drone_eligible,
        grid_size=float(grid_size),
        seed=seed,
        service_time_truck=svc_truck,
        service_time_drone=svc_drone,
        num_trucks=num_trucks,
    )


def summary(inst: Instance) -> str:
    """Return a one-block textual description of an instance."""
    lines = [
        f"Instance '{inst.name}'",
        f"  Pickup requests n = {inst.n}  (=> {2*inst.n} customers, {inst.num_nodes} total nodes)",
        f"  Grid size = {inst.grid_size} x {inst.grid_size} miles",
        f"  |C'| (drone-eligible deliveries) = {len(inst.drone_eligible)} / {inst.n}",
        f"  Trucks |V| = {inst.num_trucks}",
        f"  Drone speed/endurance = {inst.drone_speed} mph / {inst.endurance*60:.0f} min",
        f"  Drone capacity Q' = {inst.drone_capacity} kg",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    inst = generate_instance(n=4, grid_size=10.0, seed=42)
    print(summary(inst))
    print("\nCoordinates:")
    for i, c in enumerate(inst.coords):
        tag = "depot-start" if i == 0 else \
              "depot-end" if i == inst.num_nodes - 1 else \
              "pickup" if 1 <= i <= inst.n else "delivery"
        elig = " (drone-eligible)" if i in inst.drone_eligible else ""
        print(f"  node {i:2d} [{tag:11s}]: ({c[0]:6.3f}, {c[1]:6.3f}), "
              f"demand={inst.demands[i]:+7.3f} kg{elig}")
