"""
Visualisation helpers for DAPDP instances and solutions.

Produces matplotlib figures that show:
  * the depot, pickup nodes, delivery nodes (drone-eligible vs not)
  * the truck route (solid arrows)
  * the drone sorties (dashed arrows in three legs each)

Designed to be exported as PDF for direct inclusion in the LaTeX report.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from instance_generator import Instance
from dapdp_model import Solution


_TRUCK_COLOR = "#1f77b4"
_DRONE_COLOR = "#d62728"
_PICKUP_COLOR = "#2ca02c"
_DELIVERY_COLOR_TRUCK = "#7f7f7f"
_DELIVERY_COLOR_DRONE = "#ff7f0e"
_DEPOT_COLOR = "black"


def plot_instance(inst: Instance, ax=None, show_labels: bool = True):
    """Draw nodes for an instance (no routes)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Depot (start coincides with end)
    ax.scatter(inst.coords[0, 0], inst.coords[0, 1],
               s=180, c=_DEPOT_COLOR, marker="s",
               edgecolor="white", linewidth=1.5, label="depot", zorder=5)

    # Pickup nodes
    pickup_idx = list(range(1, inst.n + 1))
    ax.scatter(inst.coords[pickup_idx, 0], inst.coords[pickup_idx, 1],
               s=120, c=_PICKUP_COLOR, marker="^",
               edgecolor="white", linewidth=1.0, label="pickup", zorder=4)

    # Drone-eligible deliveries
    drone_dlvy = [j for j in range(inst.n + 1, 2 * inst.n + 1)
                  if j in inst.drone_eligible]
    truck_dlvy = [j for j in range(inst.n + 1, 2 * inst.n + 1)
                  if j not in inst.drone_eligible]
    if drone_dlvy:
        ax.scatter(inst.coords[drone_dlvy, 0], inst.coords[drone_dlvy, 1],
                   s=120, c=_DELIVERY_COLOR_DRONE, marker="o",
                   edgecolor="white", linewidth=1.0,
                   label="delivery (drone-eligible)", zorder=4)
    if truck_dlvy:
        ax.scatter(inst.coords[truck_dlvy, 0], inst.coords[truck_dlvy, 1],
                   s=120, c=_DELIVERY_COLOR_TRUCK, marker="o",
                   edgecolor="white", linewidth=1.0,
                   label="delivery (truck-only)", zorder=4)

    if show_labels:
        for i in range(inst.num_nodes - 1):  # skip end-depot duplicate
            ax.annotate(str(i), inst.coords[i],
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8)

    ax.set_xlabel("x (miles)")
    ax.set_ylabel("y (miles)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_solution(inst: Instance, sol: Solution,
                  ax=None, title: Optional[str] = None,
                  show_labels: bool = True):
    """Draw nodes plus truck route and drone sorties."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    plot_instance(inst, ax=ax, show_labels=show_labels)

    # Truck arcs
    for v, arcs in sol.truck_arcs.items():
        for (i, j) in arcs:
            xi, yi = inst.coords[i]
            xj, yj = inst.coords[j]
            ax.annotate("",
                        xy=(xj, yj), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->",
                                        color=_TRUCK_COLOR,
                                        lw=1.8,
                                        shrinkA=10, shrinkB=10),
                        zorder=3)

    # Drone sorties (3 dashed segments per sortie)
    for v, sorties in sol.drone_sorties.items():
        for (i, j, k) in sorties:
            for (a, b) in [(i, j), (j, k)]:
                xa, ya = inst.coords[a]
                xb, yb = inst.coords[b]
                ax.annotate("",
                            xy=(xb, yb), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="->",
                                            color=_DRONE_COLOR,
                                            lw=1.0, ls="--",
                                            shrinkA=10, shrinkB=10),
                            zorder=3)

    if title:
        ax.set_title(title)

    # Custom legend element to distinguish truck/drone arrows
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color=_TRUCK_COLOR, lw=2.0, label="truck"),
        Line2D([0], [0], color=_DRONE_COLOR, lw=1.5, ls="--", label="drone"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, loc="best", fontsize=8)
    return ax


if __name__ == "__main__":
    from instance_generator import generate_instance
    from dapdp_model import solve_dapdp
    inst = generate_instance(n=4, grid_size=10.0, seed=2)
    sol = solve_dapdp(inst, time_limit=120.0, verbose=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_solution(inst, sol, ax=ax,
                  title=f"DAPDP optimal — {inst.name}, ${sol.objective:.3f}")
    fig.tight_layout()
    fig.savefig("/tmp/example_solution.pdf")
    print("Saved /tmp/example_solution.pdf")
