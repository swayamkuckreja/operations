# DAPDP Codebase Walkthrough

A complete explanation of every file, what it does, how it maps to
Mulumba & Diabat (2024), and why each implementation choice was made.

This is meant to be read in order: each section builds on the last.

---

## 0. Reading order and prerequisites

You need to know:
- the **paper**: Sections 3 (model), 5.2 (instance protocol), 5.5
  (sensitivity)
- **Python dataclasses**, type hints, dictionaries
- **Gurobi/MILP basics**: variables, constraints, objective, big-M

The repository is structured as:

```
dapdp_source/
├── src/
│   ├── instance_generator.py    # build test instances
│   ├── dapdp_model.py           # the MILP itself
│   ├── baseline.py              # truck-only PDP
│   ├── visualize.py             # matplotlib plots
│   ├── verification.py          # independent audits
│   ├── sensitivity.py           # parameter sweeps
│   └── validation.py            # vs paper Table A.5
├── results/                     # CSVs and PDFs (generated)
├── report/                      # main.tex and figures
├── run_all.sh                   # pipeline runner
└── requirements.txt
```

The dependency graph among modules:

```
instance_generator.py
        │
        ▼
   dapdp_model.py ──────────► visualize.py
        │                          │
        ▼                          │
   baseline.py                     │
        │                          │
        ▼                          │
        ├──► verification.py ◄─────┘
        ├──► sensitivity.py
        └──► validation.py
```

Every other file imports `Instance` from `instance_generator.py` and
`solve_dapdp`/`Solution` from `dapdp_model.py`. Nothing else depends on
the analysis scripts (`verification`, `sensitivity`, `validation`); they
are leaves in the graph.

---

## 1. The big picture

The DAPDP is a **mixed-integer linear program (MILP)**: a set of
linear inequalities over a mix of continuous and integer/binary
variables, with a linear objective. The paper writes 37 numbered
constraints; the code reproduces every one of them and submits the
whole thing to **Gurobi** for branch-and-cut to proven optimality.

There are no metaheuristics anywhere. The solver runs branch-and-bound
exhaustively until either (a) it proves optimality, (b) the MIP gap
falls below a tolerance, or (c) the time limit is hit.

The paper *also* includes a metaheuristic (ALNS, Section 4) for solving
much larger instances where exact solving is intractable. We do not
implement it.

The pipeline `run_all.sh` does:

```
1. verification.py    → 7 small handcrafted/random scenarios, each
                         double-checked by an independent auditor
2. validation.py      → reproduce paper Table A.5 (n ∈ {6,8,10})
3. sensitivity.py     → sweep endurance, drone speed, drone cost
4. pdflatex main.tex  → compile the report
```

---

## 2. Mathematical model — the paper, restated

This section is a tight summary of paper Section 3.2 so the rest of
the walkthrough can refer back.

### Sets

| Symbol | Meaning |
|---|---|
| `N⁰ = {0,…,2n}` | all nodes except end-depot |
| `N⁺ = {1,…,2n+1}` | all nodes except start-depot |
| `C = {1,…,2n}` | customer nodes |
| `Φ⁺ = {1,…,n}` | pickup nodes |
| `Φ⁻ = {n+1,…,2n}` | delivery nodes (delivery for pickup `i` is `n+i`) |
| `C'` ⊆ `Φ⁻` | drone-eligible deliveries (demand ≤ Q′) |
| `V` | trucks (we use \|V\|=1 throughout) |
| `P` | feasible sorties `(i,j,k)`: launch `i`, drone delivery `j`, rendezvous `k` |

### Parameters (paper Table 1)

| Symbol | Default | Meaning |
|---|---|---|
| `s` | 35 mph | truck speed |
| `s'` | 50 mph | drone speed |
| `e` | 30 min | drone endurance |
| `Q` | 1300 kg | truck capacity |
| `Q'` | 2.27 kg | drone capacity |
| `α` | 0.10 | drone cost factor `c'/c` |
| `Tmax` | 8 h | maximum route duration |
| `sL, sR` | 1 min each | launch / recovery service time |
| `q_i` | varies | demand at node `i` (signed) |
| `c_ij` | `f_p · f_c · d_ij_km` | truck travel cost ($) |
| `c'_ij` | `α · c_ij` | drone travel cost ($) |
| `τ_ij = d_ij/s` | hours | truck travel time |
| `τ'_ij = d_ij/s'` | hours | drone travel time |

### Variables

| Variable | Type | Index | Meaning |
|---|---|---|---|
| `x_v_ij` | binary | v∈V, i∈N⁰, j∈N⁺, i≠j | truck v traverses arc (i,j) |
| `y_v_ijk` | binary | v∈V, (i,j,k)∈P | drone of truck v flies sortie i→j→k |
| `p_v_ij` | binary | v∈V, i,j∈C, i≠j | i precedes j on truck v's route |
| `t_v_i` | continuous ≥0 | v∈V, i∈N | truck arrival time at i |
| `t'_v_i` (`tdr`) | continuous ≥0 | v∈V, i∈N | drone arrival time at i |
| `u_v_i` | integer ≥1 | v∈V, i∈N⁺ | position of i in truck v's sequence |
| `w_v_i` | continuous ≥0 | v∈V, i∈C | truck weight after visiting i |
| `η` | continuous ≥0 | — | epigraph variable: `η ≥ total cost` |

### Objective

Minimise total fuel cost. The paper uses the epigraph trick
(Proposition 1) to keep things linear:

```
min η  subject to  η ≥ Σ c_ij · x_v_ij + Σ (c'_ij + c'_jk) · y_v_ijk
```

### Constraints, grouped

| Eqs. | Purpose |
|---|---|
| (2) | each customer served exactly once |
| (3)-(4) | depot start/end at most once per truck |
| (5) | MTZ sub-tour elimination |
| (6) | flow conservation |
| (7)-(8) | drone launches/returns at most once per node |
| (9)-(10) | launch/rendezvous nodes lie on truck route |
| (11) | launch precedes rendezvous in route order |
| (12)-(15) | drone arrival = truck arrival at launch & rendezvous |
| (16) | truck arrival recursion (with launch & recovery service times) |
| (17)-(18) | drone arrival recursion |
| (19) | drone endurance: total flight + drone service ≤ e |
| (20)-(22) | consistency between u and p |
| (23) | no second sortie before the first one returns |
| (24) | no empty depot-to-depot trip |
| (25) | pickup and delivery on same vehicle |
| (26) | pickup precedes launch (when drone delivers a delivery node) |
| (27) | pickup precedes delivery on the truck |
| (28) | truck weight evolution along the route |
| (31)-(34) | departure time = 0; arrival ≤ Tmax |
| (35) | depot precedence (we omit; see §4.4 below) |
| (36)-(37) | bounds on u and w (encoded as variable bounds) |
| (43)-(44) | valid inequalities |

---

## 3. `instance_generator.py` — building test problems

### 3.1 The `Instance` dataclass (lines 23-58)

A pure data container holding everything the model needs to know about
a single test problem. The fields are:

```python
name           : str                 # human-readable identifier
n              : int                 # number of pickup requests
coords         : np.ndarray (2n+2,2) # x,y for every node
demands        : np.ndarray (2n+2,)  # signed demand in kg
drone_eligible : List[int]           # = C'
grid_size      : float               # side length of square service area
seed           : int                 # for reproducibility
# operational parameters with paper Table 1 defaults
truck_speed, drone_speed, launch_time, recovery_time, endurance,
truck_capacity, drone_capacity, fuel_price, fuel_consumption,
miles_to_km, drone_cost_factor (α), max_route_duration (Tmax),
service_time_truck, service_time_drone, num_trucks
```

The defaults match paper Table 1 exactly. Each parameter is exposed so
sensitivity.py can sweep over it.

**Node indexing (lines 27-32):**
- `0`        : start depot
- `1..n`     : pickups (Φ⁺)
- `n+1..2n`  : deliveries (Φ⁻); delivery for pickup `i` is at `n+i`
- `2n+1`     : end depot (same coordinates as start depot)

This 1-indexed node scheme is exactly what the paper uses and keeps
constraints like "pickup `i` paired with delivery `n+i`" trivial to
write.

### 3.2 Helper methods (lines 59-83)

```python
num_nodes       → 2n+2
euclidean_distance(i,j)  → ‖coords_i − coords_j‖₂   in miles
truck_time(i,j)          → distance / truck_speed   in hours
drone_time(i,j)          → distance / drone_speed   in hours
truck_cost(i,j)          → f_p · f_c · distance_in_km
drone_cost(i,j)          → α · truck_cost(i,j)
```

The cost formula `c_ij = f_p · f_c · d_km` is the paper's, with
`miles_to_km = 1.61` to convert.

### 3.3 `generate_instance` (lines 85-149)

Reproduces the paper's protocol from §5.2:

```python
rng = np.random.default_rng(seed)
cust_coords = rng.uniform(0.0, grid_size, size=(2 * n, 2))
depot       = cust_coords.mean(axis=0)
coords      = np.vstack([depot, cust_coords, depot])
```

**Customer coords** are uniform on `[0, d]²`. **Depot** sits at the
centroid of the customers, with both start (index 0) and end (index
`2n+1`) at the same place — an artefact of the indexing scheme that
lets the model treat depot-leaving and depot-returning as different
arcs.

**Demand generation (lines 117-127):**

```python
for i in range(1, n + 1):
    delivery_node = n + i
    if rng.random() < 0.86:
        q = rng.uniform(0.0, 2.27)        # drone-eligible
        drone_eligible.append(delivery_node)
    else:
        q = rng.uniform(2.27, 68.0)       # truck-only
    demands[i]            = +q     # pickup: load gained
    demands[delivery_node] = -q    # delivery: load shed
```

Two things to flag:

1. **86% drone-eligible** matches the paper's distribution.
2. **Sign convention** (`+q` at pickup, `−q` at delivery) is the
   *opposite* of what the paper text in §5.2 says, but the *only*
   convention consistent with constraint (28) of the MILP. We chose to
   match the math, not the prose. This is documented in the report's
   Interpretation Decisions section.

**Service times (lines 130-133):**

```python
svc_truck = rng.uniform(1, 6, size=2 * n + 2) / 60.0   # 1-6 min → hours
svc_drone = rng.uniform(1, 3, size=2 * n + 2) / 60.0   # 1-3 min → hours
svc_truck[0] = svc_truck[2*n+1] = 0.0                   # zero at depot
svc_drone[0] = svc_drone[2*n+1] = 0.0
```

Per the paper. Depot service is zero because there's nothing to load
at the depot.

### 3.4 `summary()` and `__main__` (lines 152-176)

Just diagnostic helpers. Run `python instance_generator.py` to print
an example instance.

---

## 4. `dapdp_model.py` — the core MILP

This is the file. Everything else is scaffolding around it. ~628
lines. We'll go section by section.

### 4.1 Module docstring and imports (lines 1-43)

The docstring spells out the three places we deviate from the paper:

1. **(28) sign correction** — the paper writes `+ M(1−x_ij)` which
   would force every unused arc to require `w_j` arbitrarily large
   (structurally infeasible). We use `−M(1−x_ij)` (standard big-M
   relaxation that deactivates the inequality when the arc isn't used).
2. **Demand sign convention** — already discussed.
3. **(26) extension** — we forbid depot launches for delivery-type
   sorties because they would let the drone deliver before the truck
   has performed the pickup.

### 4.2 The `Solution` dataclass (lines 45-58)

```python
@dataclass
class Solution:
    status         : str    # OPTIMAL, TIME_LIMIT, INFEASIBLE, …
    objective      : float  # η*
    runtime        : float  # seconds
    mip_gap        : Optional[float]
    truck_arcs     : Dict[int, List[(i,j)]]      # per truck
    drone_sorties  : Dict[int, List[(i,j,k)]]
    truck_arrivals : Dict[(v,i), float]          # hours
    drone_arrivals : Dict[(v,i), float]
    visited_by_truck : Dict[int, List[int]]      # ordered route
    served_by_drone  : Dict[int, List[int]]
```

A pure data-out container — produced once at the end of `solve_dapdp`
and consumed everywhere else (visualize, audit, sensitivity, etc).

### 4.3 `_feasible_sorties` (lines 61-97)

Builds the index set `P` of geometrically feasible sorties. Without
this filter, `|P| = O(n³)` and the model balloons.

```python
for i in N0_no_depot:           # i ≠ 0  (depot launches forbidden by (26))
    for j in Cprime:            # only drone-eligible deliveries
        if j == i: continue
        for k in Nplus:         # any rendezvous
            if k == i or k == j: continue
            t_fly = drone_time(i,j) + drone_time(j,k)
            t_serve = service_time_drone[j]
            if sL + t_fly + t_serve + sR > e + 1e-9: continue
            P.append((i,j,k))
```

Three filters:
1. `i ≠ 0` — depot launches always forbidden by (26)
2. `j ∈ C'` — by definition only drone-eligible deliveries
3. **Endurance pre-check** — drop sorties whose round-trip can't fit

Filter 3 is critical: at e=5 min basically no sortie geometry fits, so
`|P|` collapses to nearly empty. This is what makes the small-endurance
verification scenario solve in 0.1 s.

This pre-filtering is *equivalent* to the model — every sortie excluded
here is also forbidden by constraint (19) — but pre-filtering shrinks
`|y|` by ~10x, which speeds up the LP relaxation considerably.

### 4.4 `solve_dapdp` (lines 100-600)

The main entry point. Its structure is:

```
1. Build index sets   (lines 115-124)
2. Set big-M values   (lines 127-129)
3. Create Gurobi model + parameters  (lines 131-135)
4. Add decision variables            (lines 138-165)
5. Set objective                     (lines 168-178)
6. Add constraints (2)–(37)          (lines 181-499)
7. Add valid inequalities            (lines 502-529)
8. Solve                             (lines 531-534)
9. Extract solution into a Solution  (lines 536-599)
```

#### 4.4.1 Index sets (lines 115-124)

Direct translation of paper notation:

```python
N      = list(range(0, 2*n+2))    # 0..2n+1
N0     = list(range(0, 2*n+1))    # 0..2n
Nplus  = list(range(1, 2*n+2))    # 1..2n+1
C      = list(range(1, 2*n+1))    # 1..2n
phi_plus  = list(range(1, n+1))         # pickups
phi_minus = list(range(n+1, 2*n+1))     # deliveries
Cprime = inst.drone_eligible
V      = list(range(1, inst.num_trucks+1))
P      = _feasible_sorties(inst)
```

#### 4.4.2 Big-M values (lines 127-129)

```python
big_M_time = Tmax + 10.0         # bigger than any single arrival time
big_M_pos  = 2*n + 2 = |N|       # bigger than any position u_i
```

The constants are deliberately tight — the smaller `M` is, the
stronger the LP relaxation, which is critical for solver speed.
`big_M_time` is just `Tmax` plus a small buffer. `big_M_pos = |N|` is
the smallest valid value: `u_i ∈ {1,…,|N|}` so the difference of two
position variables can never exceed `|N|−1`.

#### 4.4.3 Gurobi parameters (lines 131-135)

```python
m = gp.Model("DAPDP")
if not verbose: m.Params.OutputFlag = 0
m.Params.TimeLimit = time_limit
m.Params.MIPGap    = mip_gap_tol      # 1e-4 relative gap
```

If you want fully reproducible solves (no parallel non-determinism), add:

```python
m.Params.Threads = 1
m.Params.Seed    = 0
```

#### 4.4.4 Variables (lines 138-165)

Direct one-to-one with the paper:

```python
x[v,i,j]   binary    over v∈V, i∈N⁰, j∈N⁺, i≠j
y[v,i,j,k] binary    over v∈V, (i,j,k)∈P
p[v,i,j]   binary    over v∈V, i,j∈C, i≠j
t[v,i]     ≥0, ≤Tmax+10   over v∈V, i∈N
tdr[v,i]   ≥0, ≤Tmax+10   over v∈V, i∈N
u[v,i]     int [1, |N|]   over v∈V, i∈N⁺
w[v,i]     ≥0, ≤Q          over v∈V, i∈C
eta        ≥0
```

Note: `u` only over `N⁺`, not `N⁰`, because position 0 (= depot start)
is implicit. Similarly `w` only over `C`, not depot.

#### 4.4.5 Objective (lines 168-178)

Epigraph form (Proposition 1 in the paper):

```python
truck_cost_expr = Σ c_ij · x_v_ij
drone_cost_expr = Σ (c'_ij + c'_jk) · y_v_ijk
m.addConstr(eta >= truck_cost_expr + drone_cost_expr)
m.setObjective(eta, GRB.MINIMIZE)
```

The drone cost is `c'_ij + c'_jk` because a sortie traverses both
legs `i→j` and `j→k`. There's no return leg `k→i` because the truck
moves from `i` to `k` while the drone is out, and the drone meets the
truck at `k`.

#### 4.4.6 Constraints — group by group

I'll go through every numbered constraint that the code adds.

##### Eq. (2) — coverage (lines 182-189)

> Every customer must be served exactly once, by truck or by drone.

```python
for j in C:
    addConstr(
        Σ_v Σ_{i≠j} x[v,i,j]                           # truck arrivals
      + Σ_v Σ_{(i,j,k)∈P, jj==j} y[v,i,j,k]            # drone deliveries
      == 1
    )
```

The truck term counts arrivals at `j`; the drone term counts sorties
that *deliver* to `j` (not pass through `j` as launch or rendezvous).

##### Eq. (3)-(4) — depot start/end (lines 192-198)

> Each truck leaves the start depot at most once and arrives at the
> end depot at most once.

```python
for v in V:
    addConstr(Σ_j x[v,0,j] <= 1)              # (3)
    addConstr(Σ_i x[v,i,2n+1] <= 1)           # (4)
```

`<= 1` not `== 1` because the model allows a truck to be unused (e.g.
in multi-truck instances). With `|V|=1` and the coverage constraint
above forcing every customer to be served, we always end up with `=1`.

##### Eq. (5) — MTZ sub-tour elimination (lines 201-209)

> If `x_ij = 1`, then `u_j ≥ u_i + 1` (positions strictly increase
> along the route).

```python
for v in V:
    for i in C:
        for j in Nplus, j != i:
            addConstr(1 + u[v,i] - u[v,j] <= (1 - x[v,i,j]) * big_M_pos)
```

When `x_ij = 1`, the right-hand side becomes 0, so `u_j ≥ u_i + 1`.
When `x_ij = 0`, the right-hand side becomes `M = |N|`, deactivating
the inequality. Standard Miller-Tucker-Zemlin formulation.

This single constraint kills all sub-tours: any cycle in `C`
disconnected from the depot would force `u_i ≥ u_i + 1` somewhere,
contradiction.

##### Eq. (6) — flow conservation (lines 212-218)

> Number of arcs into `j` = number of arcs out of `j`, for every
> customer.

```python
for v in V:
    for j in C:
        addConstr(Σ_i x[v,i,j] == Σ_k x[v,j,k])
```

With (3)/(4) bounding depot flow, this forces a path from start-depot
to end-depot through served customers.

##### Eq. (7)-(8) — launch/return cardinality (lines 221-238)

> Drone launches at most once per node (per truck), and returns at
> most once per node.

```python
for v in V:
    for i in N0:
        addConstr(Σ_(j,k): (i,j,k)∈P  y[v,i,j,k] <= 1)     # (7)
    for k in Nplus:
        addConstr(Σ_(i,j): (i,j,k)∈P  y[v,i,j,k] <= 1)     # (8)
```

The drone is a single physical aircraft, so it can't launch twice
from the same node before returning, nor can it be recovered twice at
the same node.

##### Eq. (9) — on-route requirement (lines 241-246)

> Both launch and rendezvous nodes must be on the truck's route.

```python
for v in V:
    for (i,j,k) in P:
        in_i = (Σ_h x[v,h,i])  if i != 0  else 1
        in_k =  Σ_l x[v,l,k]
        addConstr(2 * y[v,i,j,k] <= in_i + in_k)
```

If `y_ijk = 1`, then `in_i + in_k ≥ 2`, forcing both to be 1, i.e.
both `i` and `k` are visited. The `if i != 0 else 1` handles the depot
launch case: the depot is always visited so `in_0 = 1` trivially.

##### Eq. (10) — subsumed (line 248-249)

> If launch is from depot 0, rendezvous k must be on truck route.

This is a special case of (9) with `i=0`. The code notes the
subsumption and skips it.

##### Eq. (11) — launch precedes rendezvous (lines 252-259)

> If `y_ijk = 1`, then `u_i < u_k` in the truck's sequence.

```python
for v in V:
    for (i,j,k) in P:
        if i == 0 or k == 2n+1: continue
        addConstr(u[v,k] - u[v,i] >= 1 - big_M_pos * (1 - y[v,i,j,k]))
```

We skip cases where `i=0` (depot, position implicitly 0) or `k=2n+1`
(end-depot, position implicitly maximum) because their position
variables don't exist.

##### Eq. (12)-(15) — time synchronisation (lines 263-291)

> Drone arrival time at the launch and rendezvous nodes must equal the
> truck arrival time there.

This is what makes the rendezvous physical: the drone has to wait at
`k` until the truck arrives (or vice versa).

```python
for i in C:
    launch_indicator = Σ_(jj,kk) y[v,i,jj,kk]   # 1 if i is a launch
    addConstr(tdr[v,i] >= t[v,i] - M * (1 - launch_indicator))   # (12)
    addConstr(tdr[v,i] <= t[v,i] + M * (1 - launch_indicator))   # (13)
for k in C:
    rdv_indicator = Σ_(ii,jj) y[v,ii,jj,k]      # 1 if k is a rendezvous
    addConstr(tdr[v,k] >= t[v,k] - M * (1 - rdv_indicator))      # (14)
    addConstr(tdr[v,k] <= t[v,k] + M * (1 - rdv_indicator))      # (15)
```

When `i` is a launch, both inequalities collapse to `tdr_i = t_i`.
When `i` isn't a launch, both inequalities are deactivated and `tdr_i`
floats freely.

##### Eq. (16) — truck arrival recursion (lines 294-312)

> If `x_hk = 1`, then `t_k ≥ t_h + d_h + τ_hk + sL · 1[h is a launch]
> + sR · 1[k is a rendezvous]`.

```python
for h in N0:
    for k in Nplus, k != h:
        launch_at_h_or_k = ...   # actually launch_at_k below
        recover_at_k     = ...
        d_h = service_time_truck[h]  # 0 at depot
        addConstr(
            t[v,k] >= t[v,h] + truck_time(h,k) + d_h
                    + launch_time * launch_at_h_or_k
                    + recovery_time * recover_at_k
                    - big_M_time * (1 - x[v,h,k])
        )
```

When `x_hk = 0` the inequality is deactivated. When it's 1, `t_k`
must be at least the time the truck spent travelling `h→k` plus any
service-on-arrival at `h`, plus any drone launch/recovery overhead.

##### Eq. (17)-(18) — drone arrival recursion (lines 315-346)

> When the drone leaves `i` for `j`, then arrives at `k` after
> delivering at `j`:
>
> (17) `tdr_j ≥ tdr_i + τ'_ij` if drone goes `i→j`
>
> (18) `tdr_k ≥ tdr_j + τ'_jk + d'_j + sR` if drone goes `j→k`

The launch service time `sL` is already included in `t_i = tdr_i` via
constraint (16); the recovery service time `sR` is added in (18) at
the rendezvous.

##### Eq. (19) — endurance (lines 350-356)

> Total drone time on a sortie ≤ endurance `e`.

```python
for (i,j,k) in P:
    addConstr(
        tdr[v,k] - (tdr[v,j] - drone_time(i,j))
        <= endurance + big_M_time * (1 - y[v,i,j,k])
    )
```

Read this: `tdr_k − tdr_j + τ'_ij ≤ e`. The `−τ'_ij` term backs out
the time it took to fly `i→j`, leaving everything from launch to
rendezvous. When `y_ijk = 0` the big-M deactivates.

##### Eq. (20)-(22) — u/p consistency (lines 359-380)

> p_ij = 1 iff i precedes j in the truck's sequence.

```python
for i,j in C, i != j:
    addConstr(u_i - u_j >= 1 - M·p_ij)            # (20): if p=1, u_i > u_j? No: re-read.
    addConstr(u_i - u_j <= -1 + (1-p_ij)·M)       # (21)
for i<j:
    addConstr(p_ij + p_ji == 1)                    # (22)
```

Note the directional convention: `p_ij = 1` means **i precedes j** on
the route — but in the paper's notation, that means `u_i < u_j`, so
"i precedes j" iff `u_i < u_j`. Re-reading the constraints with that:

- (20): `u_i − u_j ≥ 1 − M · p_ij`. If `p_ij = 0` (i does NOT precede
  j), then `u_i − u_j ≥ 1`, i.e. `u_i > u_j`. ✓
- (21): `u_i − u_j ≤ −1 + (1−p_ij)·M`. If `p_ij = 1` (i precedes j),
  then `u_i − u_j ≤ −1`, i.e. `u_i < u_j`. ✓
- (22): exactly one direction holds.

##### Eq. (23) — non-overlapping sorties (lines 383-413)

> No second drone sortie can begin (at any later launch node `l`)
> before the first one has returned to its rendezvous `k`.

This is the most complex constraint. It says: if the first sortie is
`(i,j,k)` and a second sortie launches from `l` later, then `tdr_l ≥
tdr_k`. The "later" condition is captured by the precedence variable
`p_il`.

```python
for i in N0, k in Nplus:
    first_sortie = Σ_jj y[v,i,jj,k]     # 1 if some sortie i→·→k exists
    for l in C:
        second_sortie = Σ_mm,nn y[v,l,mm,nn]   # any sortie launching from l
        precedence_term = 1 if i==0 else p[v,i,l]   # i precedes l
        addConstr(tdr_l >= tdr_k - M·(3 − first − second − precedence))
```

When all three indicators are 1, the inequality activates and forces
`tdr_l ≥ tdr_k`. Otherwise it's slack.

The code carefully handles `i=0` (depot is implicitly before
everything) and skips degenerate cases.

##### Eq. (24) — no empty trip (lines 416-418)

> The truck must not go directly from start-depot to end-depot
> without serving anyone.

```python
addConstr(x[v,0,2n+1] == 0)
```

##### Eq. (25) — same vehicle (lines 421-427)

> Pickup `i` and delivery `n+i` must be served by the same vehicle.

```python
for v in V:
    for i in phi_plus:
        ni = n + i
        # arrivals at i = arrivals at n+i (by truck) + drone deliveries to n+i
        addConstr(Σ_j x[v,i,j] == Σ_j x[v,ni,j] + Σ_(l,k) y[v,l,ni,k])
```

The left side is "truck v leaves pickup i" (which by flow conservation
equals "truck v arrives at pickup i"). The right side is "truck v
visits delivery n+i" plus "the drone of truck v delivers n+i". The two
must match because if v handles the pickup, it also handles the
delivery (by truck or by its own drone).

With `|V|=1` this is trivially satisfied, but for multi-truck
instances it's the key pairing constraint.

##### Eq. (26) — pickup before launch (lines 433-447)

> If the drone delivers a delivery node `j ∈ Φ⁻` via sortie `(i,j,k)`,
> then in the truck route, the corresponding pickup `j−n` must come
> before the launch `i`.

This is the constraint we strengthened. The paper writes it for `i ∈
C`, leaving `i = 0` (depot launch) out, which would let the drone
deliver before the truck has done the pickup.

```python
for (i,j,k) in P:
    if j not in phi_minus: continue
    pickup_node = j - n
    if i == 0:
        addConstr(y[v,0,j,k] == 0)        # forbid depot launches
    elif i in C and pickup_node in C and i != pickup_node:
        addConstr(u[v,i] - u[v,pickup_node] >= 1 - M·(1 - y[v,i,j,k]))
```

##### Eq. (27) — pickup before delivery on truck (lines 450-457)

> If both pickup and delivery are on the truck's route, the truck
> visits the pickup first.

```python
for i in phi_plus:
    ni = n + i
    addConstr(t[v,ni] >= t[v,i] + truck_time(i,ni) + service_time_truck[i])
```

Note the constraint is on **time**, not on position. The paper does it
this way (and it's what (27) actually says), and it works because
arrival times monotonically increase along the route.

##### Eq. (28) — truck weight (lines 462-483)

> If `x_ij = 1`, then `w_j ≥ w_i + q_j + Σ q_m · y_jmk`.

```python
for i in N0, j in C, i != j:
    drone_handover = Σ_(jj,mm,kk)∈P, jj==j  q_mm · y[v,j,mm,kk]
    w_prev = 0.0 if i == 0 else w[v,i]
    addConstr(
        w[v,j] >= w_prev + q_j + drone_handover
                  - truck_capacity * (1 - x[v,i,j])
    )
```

Three things to flag:
1. **Sign of big-M** is `−Q` (the paper has `+Q`, which is a typo).
2. `drone_handover` is `q_m · y_jmk`, which with our sign convention
   is *negative* (delivery demand is negative). So the truck weight
   *decreases* at `j` if a drone parcel is launched there, as it
   should: the drone takes some weight off the truck.
3. `w_prev = 0` at the depot because there's no `w[v,0]` variable.

##### Eq. (31)-(32) — departure (lines 487-488)

> Truck and drone both depart the start depot at time 0.

```python
addConstr(t[v,0] == 0)
addConstr(tdr[v,0] == 0)
```

##### Eq. (33)-(34) — Tmax (lines 491-493)

> Total route duration ≤ Tmax.

```python
addConstr(t[v,2n+1] <= Tmax)
addConstr(tdr[v,2n+1] <= Tmax)
```

##### Eq. (35) — omitted

> p_v_0_j = 1 for all j ∈ C (depot precedes everyone).

We omit this. Including it would require `p` variables on the depot
(which would inflate the variable count by `|C|`); we don't, because
all our other depot-linked constraints (MTZ, time/position bounds,
(3)-(4)) implicitly place the depot first.

##### Eq. (36)-(37) — bounds

> `1 ≤ u_i ≤ |N|` and `0 ≤ w_i ≤ Q`.

These are encoded directly as variable bounds in the variable
definitions (lines 157, 161). Nothing to add as constraints.

##### Valid inequalities (43)-(44) (lines 503-529)

> (43): `x_ij + Σ_k y_ijk ≤ 1` — can't serve `j` by truck and drone
> from `i` simultaneously.
>
> (44): `x_ij + x_jk + x_ik + y_ijk ≤ 2` — at most two of the four
> "edges" of the sortie triangle can be active.

These are *redundant* (they're implied by the rest of the model) but
**tighten the LP relaxation**, often dramatically. With them on, the
solver explores fewer nodes. Default: enabled.

#### 4.4.7 Solving and extracting (lines 531-599)

```python
t0 = time.time()
m.optimize()
runtime = time.time() - t0
```

After solve, we map Gurobi status codes to readable strings, then —
if any incumbent exists — extract:

```python
truck_arcs[v]    = [(i,j) : x[v,i,j].X > 0.5]
drone_sorties[v] = [(i,j,k) : y[v,i,j,k].X > 0.5]
truck_arrivals[(v,i)] = t[v,i].X
drone_arrivals[(v,i)] = tdr[v,i].X
served_by_drone[v]   = [j for (i,j,k) in drone_sorties[v]]
```

The final loop (lines 567-587) reconstructs the truck route as an
ordered list `[0, …, 2n+1]` by traversing the arcs:

```python
seq = [0]
cur = 0
while True:
    nxt = next j s.t. (cur, j) in truck_arcs[v] and j not yet seen
    if nxt is None: break
    seq.append(nxt)
    cur = nxt
    if cur == 2n+1: break
```

This is purely a presentation step — the model itself uses arc
incidence, not an ordered list.

### 4.5 `pretty_print` (lines 603-618)

Diagnostic helper. Prints status, objective, runtime, route, and
sorties.

---

## 5. `baseline.py` — the PDP comparator

The truck-only baseline used for SPDP computation.

```python
def solve_truck_only(inst):
    truck_only_inst = Instance(... , drone_eligible=[], ...)
    return solve_dapdp(truck_only_inst, ...)
```

Setting `drone_eligible = []` empties `C'`, which empties `P` (no
feasible sorties), which empties the `y` dictionary. Every `y_ijk`
disappears from the model and the DAPDP collapses to a pure
pickup-and-delivery routing problem with a single capacitated truck.

This is **the right way** to construct the PDP baseline because:

1. Any cost difference between PDP and DAPDP is *guaranteed* to be due
   to drone usage, not to formulation differences.
2. We don't need to write a separate PDP MILP — same code path, same
   constraints, same solver behaviour.

The clone is a *deep* copy in spirit: every operational parameter is
copied, but the coordinates and demands are shared (NumPy arrays
shared by reference). That's safe because nothing in `solve_dapdp`
mutates them.

---

## 6. `visualize.py` — sanity checking

### 6.1 Colour scheme (lines 23-28)

```python
TRUCK = blue   DRONE = red    PICKUP = green
DELIVERY_TRUCK = grey   DELIVERY_DRONE = orange   DEPOT = black
```

Mnemonics: trucks are work-vehicle blue, drones are red because they're
fast and dangerous, pickups green because they add load, drone-eligible
deliveries orange because they're "special".

### 6.2 `plot_instance` (lines 31-74)

Just nodes, no routes. Used as a starting point for both
`plot_solution` and `plot_pairing_only`.

```python
ax.scatter(coords[0], …, marker="s", color=DEPOT)         # depot
ax.scatter(coords[pickups], marker="^", color=PICKUP)     # pickups
ax.scatter(coords[drone_dlvy], marker="o", color=ORANGE)  # drone-eligible
ax.scatter(coords[truck_dlvy], marker="o", color=GREY)    # truck-only
```

Labels each non-end-depot node with its index. Equal aspect ratio so
miles are square.

### 6.3 `plot_solution` (lines 86-138)

Adds:

- **Truck arcs**: solid blue arrows between consecutive nodes
- **Drone sorties**: dashed red arrows in two segments (`i→j` and
  `j→k`)
- **Pickup-delivery pairing** (optional, with `show_pairing=True`):
  thin grey dotted arrows from each pickup to its delivery, drawn
  *behind* everything else

The dotted pairing overlay is the visual sanity check: when you look
at the route, every grey arrow should be honoured (pickup visited
before delivery) by the actual blue/red routing.

### 6.4 `plot_pairing_only` (lines 140-160)

Same as `plot_instance` but adds curved grey arrows for each pairing.
No routes. Useful for inspecting the instance topology without route
clutter.

---

## 7. `verification.py` — independent feasibility audits

### 7.1 The `Audit` dataclass (lines 44-61)

Container for a list of `issues` (failures) and `notes`
(informational). `passed` is True iff `issues` is empty.

### 7.2 `audit()` (lines 64-172) — the heart of verification

For each truck `v`, this function independently re-checks every
feasibility condition by inspecting only the returned `truck_arcs`,
`drone_sorties`, `truck_arrivals`, `drone_arrivals`, **without**
querying any Gurobi state. If the model returned a solution that
violates a constraint, this function will catch it.

The checks (in order, lines 78-165):

1. **Coverage** (lines 79-87): every customer either on the truck
   route or in the drone-served set, but not both.
2. **Depot endpoints** (lines 89-93): truck route starts at 0 and
   ends at 2n+1.
3. **Pickup-before-delivery** (lines 95-104): for each pair `(i, n+i)`
   on the truck route, position of pickup < position of delivery.
4. **Drone sortie integrity** (lines 106-140): for each `(i,j,k)`:
   - launch `i` is on truck route (or = 0)
   - rendezvous `k` is on truck route (or = 2n+1)
   - geometric endurance: `sL + τ'_ij + τ'_jk + d'_j + sR ≤ e`
   - drone payload: `|q_j| ≤ Q'`
   - the corresponding pickup is on the truck route
   - the launch comes after the pickup in route order
5. **Truck capacity** (lines 142-157): walk the route accumulating
   `q_i` and subtracting `|q_j|` at each launch; load must never
   exceed `Q`.
6. **Tmax** (lines 159-165): truck arrival at end-depot ≤ Tmax.

The fact that this is **independent code** is what makes the
verification meaningful. If `solve_dapdp` had a bug, the audit would
catch it.

### 7.3 The seven scenarios

#### V1 — `scenario_basic` (lines 184-193)

Random instance, n=3. Just runs the model and audits. The point: prove
the model produces feasible solutions on a clean random input.

#### V2 — `scenario_pickup_precedence` (lines 196-226)

Handcrafted instance with all demands set to 5 kg (> Q' = 2.27), so
`drone_eligible = []` and the truck must serve every pair. The 8-node
geometry is chosen so the optimal route visits pickups 1,2,3 (top
row) then deliveries 4,5,6 (bottom row) in order, terminating at
end-depot 7.

The audit's pickup-before-delivery check is the relevant one here.
This isolates constraint (27) from drone-related complications.

#### V3 — `scenario_drone_endurance` (lines 229-254)

Same instance solved twice with different endurance values:
- e = 60 min → drone is generously useful, multiple sorties
- e = 5 min → no sortie geometrically fits, zero sorties

The number of sorties at the tight endurance must be ≤ the number at
the loose endurance. This tests constraint (19) and the
`_feasible_sorties` pre-filter.

#### V4 — `scenario_drone_payload` (lines 257-283)

Three deliveries with demands `{1, 50, 1}` kg. Only deliveries 4 and 6
are in `C'`. Delivery 5 (50 kg, way over Q'=2.27) is forced onto the
truck route. The audit verifies that:
- drone never serves customer 5
- truck route does include 5

#### V5 — `scenario_no_subtours` (lines 286-315)

Random n=6 instance. Beyond the standard audit, this scenario walks
the arc graph from the start-depot to the end-depot and checks:
- the chain reaches `2n+1`
- there are no extra arcs not on the chain (which would be sub-tours)

This is the direct test of constraint (5).

#### V6 — `scenario_pdp_vs_dapdp` (lines 318-333)

Same instance solved twice: PDP and DAPDP. Checks that DAPDP cost ≤
PDP cost (which it must by construction, since DAPDP includes PDP as
a special case). A negative SPDP would indicate either:
- non-determinism in parallel solving (fix: `Threads=1, Seed=0`)
- a model bug

#### V7 — `scenario_no_depot_drone_delivery` (lines 336-349)

Confirms that no drone sortie has launch node = 0. This tests our
extension of constraint (26).

### 7.4 `main()` (lines 352-368)

Runs all 7 scenarios and prints results. Crashes are caught and
reported but don't abort the suite.

---

## 8. `sensitivity.py` — parameter sweeps

### 8.1 The `Run` dataclass (lines 49-61)

Container for one (parameter, value, instance) triple's result:

```python
parameter, value, instance, n, grid, seed,
z_pdp, z_dapdp, spdp_pct, n_sorties, runtime_s
```

### 8.2 `_run_one` (lines 64-89)

Solves DAPDP for one (instance, parameter value), with the PDP cost
cached per instance (since it doesn't depend on drone parameters).

### 8.3 `_make_ensemble` (lines 93-101)

The default ensemble: `n ∈ {6, 8}`, `grid ∈ {10, 20}`, `seed ∈ {1,2,3}`
= 12 instances.

To shrink for faster runs: change to `for n in [6]:` (cuts time ~75%).

### 8.4 The three sweeps (lines 105-168)

```python
sweep_endurance     : e ∈ {5, 7.5, 10, 15, 30, 60} min   (paper Table 3)
sweep_drone_speed   : s' ∈ {25, 35, 50, 75} mph          (paper Table 4)
sweep_alpha         : α ∈ {0.05, 0.10, 0.25, 0.50, 1.00} (no paper analogue)
```

Each sweep mutates `inst.<parameter>` in place, calls `_run_one`, and
resets to the default at the end. Since `_make_ensemble` instances are
shared across sweeps, the reset is important.

### 8.5 Plotting (lines 198-221)

Each sweep produces a dual-axis line plot:
- left axis (blue circles): mean SPDP %
- right axis (red squares): mean number of drone sorties

The expected qualitative shape:
- **Endurance**: SPDP rises monotonically, saturates around 30 min.
- **Drone speed**: SPDP rises, flattens above 50 mph.
- **Alpha**: SPDP falls monotonically, → 0 at α = 1.

### 8.6 CSV output

Every individual `Run` is written to a CSV (one per parameter), so
you can re-aggregate or re-plot in Excel/pandas if needed.

---

## 9. `validation.py` — comparison with paper Table A.5

The paper reports mean SPDP and runtime for `n ∈ {6,8,10,12}` on
grids `{5,10,20,30}`. This script reproduces a slimmed version:

```python
seeds_per_n = {6: 5, 8: 3, 10: 2}      # fewer seeds at larger n
time_limit_per_n = {6: 120, 8: 300, 10: 600}    # seconds
```

For each `(n, grid)` cell it solves 2-5 random seeds, both DAPDP and
PDP, and reports mean SPDP, std SPDP, and mean runtime. Output goes to
`results/verification/validation_table.csv`.

The expected ballpark:
- mean SPDP ≈ 25% on small/medium grids
- much lower on 30-mile grids (endurance limit kicks in)

---

## 10. `run_all.sh` — pipeline runner

```bash
1. cd src && python verification.py    → V1-V7
2. python validation.py                 → vs paper Table A.5
3. python sensitivity.py                → 3 parameter sweeps
4. cp results/*/V*.pdf report/figures/
5. cp results/sensitivity/*.pdf report/figures/
6. cd report && pdflatex main.tex (×2)
```

Single-command regeneration of every artifact in the project.

---

## 11. Paper-to-code reference table

If you only want to know "where in the code is constraint X?", here:

| Paper | Where | Note |
|---|---|---|
| Eq. (1) objective | `dapdp_model.py` 168-178 | epigraph form |
| Eq. (2) coverage | 182-189 |  |
| Eq. (3) start depot | 192-193 |  |
| Eq. (4) end depot | 196-198 |  |
| Eq. (5) MTZ | 201-209 |  |
| Eq. (6) flow | 212-218 |  |
| Eq. (7) launch ≤1 | 221-228 |  |
| Eq. (8) recover ≤1 | 231-238 |  |
| Eq. (9) on-route | 241-246 |  |
| Eq. (10) | 248-249 | subsumed by (9) |
| Eq. (11) launch precedes rdv | 252-259 |  |
| Eq. (12)-(15) time sync | 263-291 |  |
| Eq. (16) truck arrival | 294-312 |  |
| Eq. (17) drone i→j | 315-328 |  |
| Eq. (18) drone j→k | 331-346 |  |
| Eq. (19) endurance | 350-356 | also pre-filtered into P |
| Eq. (20)-(22) u/p | 359-380 |  |
| Eq. (23) no overlap | 383-413 |  |
| Eq. (24) no empty trip | 416-418 |  |
| Eq. (25) same vehicle | 421-427 |  |
| Eq. (26) pickup before launch | 433-447 | extended to forbid depot launches |
| Eq. (27) pickup before delivery | 450-457 |  |
| Eq. (28) weight | 462-483 | sign correction; uses −M not +M |
| Eq. (31)-(32) departure | 487-488 |  |
| Eq. (33)-(34) Tmax | 491-493 |  |
| Eq. (35) | 495-498 | omitted |
| Eq. (36)-(37) bounds | implicit | encoded as variable bounds |
| Eq. (43)-(44) VI | 503-529 |  |
| §5.2 instance protocol | `instance_generator.py` 100-149 |  |
| §5.5 sensitivity | `sensitivity.py` 105-168 | three sweeps |
| Tab. A.5 validation | `validation.py` 29-87 |  |

---

## 12. Things you should be ready to defend in the report or viva

1. **Why epigraph?** Because the objective `min Σ c·x + Σ c'·y` is
   already linear, so the epigraph form `min η, η ≥ Σ…` is
   mathematically equivalent. The paper does it for symmetry with
   their stochastic-cost extension; we keep it for fidelity.

2. **Why pre-filter P?** The MILP enforces (19) anyway, so any sortie
   we drop from P is also forbidden by the model. But pre-filtering
   shrinks `|y|` ~10x, which helps the LP relaxation and the branching.

3. **Why Threads=1?** Reproducibility. Parallel branch-and-bound is
   non-deterministic in tie-breaking. Multiple runs of the same
   instance can return different (but equally optimal) solutions.

4. **Why is V6 important?** It's a structural sanity check:
   `z_DAPDP ≤ z_PDP` must hold by construction (DAPDP is a relaxation
   of PDP). If V6 ever fails, something is wrong with the formulation.

5. **Why is the (28) sign correction needed?** The paper writes
   `+M(1−x_ij)`. With `x_ij = 0`, this would force `w_j` to be
   bounded above by something `M − Q` huge, which combined with the
   `Q` upper bound on `w_j` makes the system infeasible whenever
   `M > 2Q`. The correct big-M relaxation pattern is to deactivate the
   inequality, which requires the minus sign.

6. **Why omit (35)?** It says `p_v_0_j = 1` for every customer,
   meaning the depot precedes everyone. That's already captured by
   the structure of `u` and the time constraints. Adding `p_v_0_j`
   variables would inflate `|p|` by `|C|` for no information gain.

7. **Why 18-25% mean SPDP?** Because (a) only ~86% of deliveries are
   drone-eligible, (b) only sorties whose round-trip fits in 30 min
   are usable, (c) the drone only saves cost when its route is shorter
   than the truck detour avoided. The intersection of these three is
   the savings ceiling, and 18-25% is in that range.

8. **Why does SPDP collapse on 30-mile grids?** A 30 min drone flight
   at 50 mph covers 25 miles; round-trip + service must fit in 30 min
   so the drone can only reach customers within roughly 10 miles of
   any potential launch point. On a 30×30 grid, the depot-to-customer
   distance often exceeds that, so most candidate sorties fail the
   geometric pre-filter.

---

## 13. Common edits you might make

| Want to … | Edit |
|---|---|
| Run smaller/faster | `sensitivity.py` `_make_ensemble`: `[6, 8]` → `[6]` |
| Change paper params | `instance_generator.py` `Instance` defaults |
| Change ensemble | `validation.py` `seeds_per_n` and grid loop |
| Add a new constraint | `dapdp_model.py` after the existing constraint blocks |
| Add a new audit check | `verification.py` `audit()` |
| Add a new sortie pre-filter | `dapdp_model.py` `_feasible_sorties` |
| Make solves deterministic | Add `Threads=1, Seed=0` to `dapdp_model.py` and `baseline.py` after `m = gp.Model(...)` |
