#DONT USE THIS
# 
# 
# 
# 
# 
# ITS AN OLD IMPLEMENTATION - NONE OF THIS CODE IS USED FOR THE REPORT
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# import sys
import math
import random
import matplotlib.pyplot as plt
from docplex.mp.model import Model

# ==============================================================================
# PART 1: DATA GENERATION & INSTANCE CLASS
# ==============================================================================
class Instance:
    """
    Represents a specific DAPDP scenario.
    Generates data based on Mulumba & Diabat (2024) parameters.
    """
    def __init__(self, n_requests, grid_size=10, seed=42):
        self.n = n_requests
        self.grid_size = grid_size
        self.num_nodes = 2 * n_requests + 2
        
        # Node Indices
        self.depot_start = 0
        self.depot_end = 2 * n_requests + 1
        
        # Sets
        self.P = list(range(1, self.n + 1)) # Pickups
        self.D = list(range(self.n + 1, 2 * self.n + 1)) # Deliveries
        self.C = self.P + self.D # All customers
        self.C_prime = self.D # Drone eligible nodes (Strictly deliveries)
        
        self.N_0 = [self.depot_start] + self.C # Departure nodes
        self.N_plus = self.C + [self.depot_end] # Arrival nodes
        self.N = [self.depot_start] + self.C + [self.depot_end] # All nodes
        self.V = [1] # Single Truck

        # --- Parameters (Table 1) ---
        self.s_truck = 35 / 60.0  # miles/min
        self.s_drone = 50 / 60.0  # miles/min
        self.sL = 1 # Launch time (min)
        self.sR = 1 # Recovery time (min)
        self.endurance = 40 # min (Default, can be changed for sensitivity)
        self.Q = 1000 # Truck Capacity (kg)
        self.Q_drone = 2.27 # Drone Capacity (kg) ~5lbs
        
        # Cost Parameters (Section 5.1)
        # Cost ($) = f_p * f_c * distance
        self.f_p = 0.840 # Fuel Price ($/liter)
        self.f_c = 0.07 * 1.61 # Fuel Consumption (liter/mile) -> 0.07 l/km * 1.61 km/mi
        self.cost_per_mile = self.f_p * self.f_c 
        self.alpha = 0.1 # Drone cost is 10% of truck cost
        
        self.service_time_truck = {i: 2 for i in self.N}
        self.big_M = 10000 

        # --- Coordinate Generation ---
        random.seed(seed) 
        self.coords = {}
        # Depot at center
        self.coords[self.depot_start] = (grid_size/2, grid_size/2)
        self.coords[self.depot_end] = (grid_size/2, grid_size/2)
        
        for i in self.C:
            self.coords[i] = (random.uniform(0, grid_size), random.uniform(0, grid_size))
            
        # --- Demands & Weight ---
        self.q = {i: 0 for i in self.N}
        for i in self.P:
            # 86% chance to be drone eligible (< 2.27kg)
            if random.random() < 0.86:
                weight = random.uniform(0.5, 2.2)
            else:
                weight = random.uniform(2.3, 10.0)
            
            del_node = i + self.n
            self.q[i] = weight
            self.q[del_node] = -weight
            
        # --- Distances & Time Matrices ---
        self.tau = {}       
        self.tau_prime = {} 
        self.c_truck = {}   
        
        for i in self.N:
            for j in self.N:
                dist = math.sqrt((self.coords[i][0]-self.coords[j][0])**2 + 
                                 (self.coords[i][1]-self.coords[j][1])**2)
                
                self.tau[i, j] = dist / self.s_truck
                self.tau_prime[i, j] = dist / self.s_drone
                self.c_truck[i, j] = dist * self.cost_per_mile

    def get_drone_cost(self, i, j, k):
        """Calculates cost for sortie i -> j -> k"""
        dist = math.sqrt((self.coords[i][0]-self.coords[j][0])**2 + (self.coords[i][1]-self.coords[j][1])**2) + \
               math.sqrt((self.coords[j][0]-self.coords[k][0])**2 + (self.coords[j][1]-self.coords[k][1])**2)
        return (dist * self.cost_per_mile) * self.alpha

# ==============================================================================
# PART 2: EXACT MILP SOLVER
# ==============================================================================
def solve_dapdp(inst: Instance, verbose=True, time_limit=120):
    if verbose: print(f"Solving DAPDP (n={inst.n}, Nodes={inst.num_nodes}, Endurance={inst.endurance}m)...")
    mdl = Model(name='DAPDP')
    
    # --- Decision Variables ---
    x_keys = [(v, i, j) for v in inst.V for i in inst.N_0 for j in inst.N_plus if i != j]
    x = mdl.binary_var_dict(x_keys, name='x')
    
    # Drone Sorties (Pre-filtered by endurance and weight)
    y_keys = []
    for v in inst.V:
        for i in inst.N_0:
            for j in inst.C_prime:
                # Drone Weight Check:
                pickup_node = j - inst.n
                if inst.q[pickup_node] > inst.Q_drone: continue 
                
                for k in inst.N_plus:
                    if i != j and j != k and i != k:
                        # Endurance Check:
                        if (inst.tau_prime[i, j] + inst.tau_prime[j, k] + inst.sL + inst.sR) <= inst.endurance:
                            y_keys.append((v, i, j, k))
    y = mdl.binary_var_dict(y_keys, name='y')
    
    p = mdl.binary_var_dict([(v, i, j) for v in inst.V for i in inst.C for j in inst.C if i != j], name='p')
    t_truck = mdl.continuous_var_dict([(v, j) for v in inst.V for j in inst.N], name='t_truck', lb=0)
    t_drone = mdl.continuous_var_dict([(v, j) for v in inst.V for j in inst.N_plus], name='t_drone', lb=0)
    u = mdl.continuous_var_dict([(v, i) for v in inst.V for i in inst.N_plus], name='u', lb=1, ub=2*inst.n + 2)
    w = mdl.continuous_var_dict([(v, i) for v in inst.V for i in inst.N], name='w', lb=0, ub=inst.Q)

    # --- Objective ---
    obj = mdl.sum(inst.c_truck[i, j] * x[v, i, j] for v, i, j in x) + \
          mdl.sum(inst.get_drone_cost(i, j, k) * y[v, i, j, k] for v, i, j, k in y)
    mdl.minimize(obj)
    
    # --- Constraints ---
    
    # (2) Coverage
    for j in inst.C:
        mdl.add_constraint(
            mdl.sum(x[v, i, j] for v in inst.V for i in inst.N_0 if (v, i, j) in x) + 
            mdl.sum(y[v, i, j, k] for v in inst.V for i in inst.N_0 for k in inst.N_plus if (v, i, j, k) in y) == 1
        )

    # (3-6) Flow & Depots
    for v in inst.V:
        mdl.add_constraint(mdl.sum(x[v, inst.depot_start, j] for j in inst.N_plus if (v, inst.depot_start, j) in x) == 1)
        mdl.add_constraint(mdl.sum(x[v, i, inst.depot_end] for i in inst.N_0 if (v, i, inst.depot_end) in x) == 1)
        for j in inst.C:
            mdl.add_constraint(mdl.sum(x[v, i, j] for i in inst.N_0 if (v, i, j) in x) == 
                               mdl.sum(x[v, j, k] for k in inst.N_plus if (v, j, k) in x))

    # (7-8) Drone Launch/Recover Limits
    for v in inst.V:
        for i in inst.N_0:
            mdl.add_constraint(mdl.sum(y[v, i, j, k] for j in inst.C_prime for k in inst.N_plus if (v, i, j, k) in y) <= 1)
        for k in inst.N_plus:
            mdl.add_constraint(mdl.sum(y[v, i, j, k] for i in inst.N_0 for j in inst.C_prime if (v, i, j, k) in y) <= 1)

    # (9) Sync
    for v in inst.V:
        for i in inst.N_0:
            for k in inst.N_plus:
                rel = [y[v, i, j, k] for j in inst.C_prime if (v, i, j, k) in y]
                if rel:
                    mdl.add_constraint(2 * mdl.sum(rel) <= mdl.sum(x[v, i, h] for h in inst.N_plus if (v, i, h) in x) + mdl.sum(x[v, l, k] for l in inst.N_0 if (v, l, k) in x))

    # (11) Order (u variables)
    for v in inst.V:
        for i in inst.C:
            for k in inst.C:
                if i != k:
                    rel = [y[v, i, j, k] for j in inst.C_prime if (v, i, j, k) in y]
                    for yv in rel: mdl.add_constraint(u[v, k] - u[v, i] >= 1 - (2*inst.n+2)*(1-yv))

    # (16) Truck Time Propagation
    for v in inst.V:
        mdl.add_constraint(t_truck[v, inst.depot_start] == 0)
        for h in inst.N_0:
            for k in inst.N_plus:
                if h!=k and (v, h, k) in x:
                    launch = mdl.sum(y[v, h, j, m] for j in inst.C_prime for m in inst.N_plus if (v, h, j, m) in y)
                    recover = mdl.sum(y[v, l, j, k] for l in inst.N_0 for j in inst.C_prime if (v, l, j, k) in y)
                    mdl.add_constraint(t_truck[v, k] >= t_truck[v, h] + inst.tau[h, k] + inst.service_time_truck[h] + 
                                       (inst.sL * launch) + (inst.sR * recover) - inst.big_M*(1 - x[v, h, k]))

    # (17-18) Drone Time Sync
    for v in inst.V:
        for j in inst.C_prime:
            # Drone Arrival
            for i in inst.N_0:
                rel = [y[v, i, j, k] for k in inst.N_plus if (v, i, j, k) in y]
                if rel: mdl.add_constraint(t_drone[v, j] >= t_truck[v, i] + inst.tau_prime[i, j] + inst.sL - inst.big_M*(1 - mdl.sum(rel)))
            # Drone Return (Truck must wait)
            for k in inst.N_plus:
                rel = [y[v, i, j, k] for i in inst.N_0 if (v, i, j, k) in y]
                if rel: mdl.add_constraint(t_truck[v, k] >= t_drone[v, j] + inst.tau_prime[j, k] + inst.sR - inst.big_M*(1 - mdl.sum(rel)))

    # (19) Endurance
    for v in inst.V:
        for i in inst.N_0:
            for j in inst.C_prime:
                for k in inst.N_plus:
                    if (v, i, j, k) in y:
                        req = inst.tau_prime[i, j] + inst.tau_prime[j, k] + inst.sL + inst.sR
                        mdl.add_constraint(y[v, i, j, k] * req <= inst.endurance)

    # (20-22) MTZ Order
    for v in inst.V:
        for i in inst.C:
            for j in inst.C:
                if i!=j:
                    mdl.add_constraint(u[v, i] - u[v, j] >= 1 - (2*inst.n+2)*p[v, i, j])
                    mdl.add_constraint(u[v, i] - u[v, j] <= -1 + (2*inst.n+2)*(1 - p[v, i, j]))
                    mdl.add_constraint(p[v, i, j] + p[v, j, i] == 1)

    # (25) Pickup & Delivery Pairing & Strict Precedence
    for i in inst.P:
        del_node = i + inst.n
        for v in inst.V:
            # Consistency
            mdl.add_constraint(mdl.sum(x[v, h, i] for h in inst.N_0 if (v, h, i) in x) == 
                               mdl.sum(x[v, h, del_node] for h in inst.N_0 if (v, h, del_node) in x) + 
                               mdl.sum(y[v, l, del_node, m] for l in inst.N_0 for m in inst.N_plus if (v, l, del_node, m) in y))
            
            # Strict Precedence (Prevent Time Machine)
            visited_i = mdl.sum(x[v, h, i] for h in inst.N_0 if (v, h, i) in x)
            visited_j = mdl.sum(x[v, h, del_node] for h in inst.N_0 if (v, h, del_node) in x)
            
            # Case A: Both on Truck
            if (v, i, del_node) in p:
                mdl.add_constraint(p[v, i, del_node] >= visited_i + visited_j - 1)
            
            # Case B: Delivery by Drone (launch l)
            if del_node in inst.C_prime:
                for l in inst.N_0:
                    if l != i:
                        for m in inst.N_plus:
                            if (v, l, del_node, m) in y and (v, i, l) in p:
                                # If launch at l, truck must visit i before l
                                mdl.add_constraint(p[v, i, l] >= y[v, l, del_node, m] + visited_i - 1)

    # (28) Load
    for v in inst.V:
        mdl.add_constraint(w[v, inst.depot_start] == 0)
        for i in inst.N_0:
            for j in inst.N_plus:
                if i != j and (v, i, j) in x:
                    mdl.add_constraint(w[v, j] >= w[v, i] + inst.q[j] - inst.big_M*(1 - x[v, i, j]))

    # Solve
    mdl.parameters.timelimit = time_limit
    mdl.parameters.mip.tolerances.mipgap = 0.05
    sol = mdl.solve(log_output=verbose)
    
    return sol, x, y

# ==============================================================================
# PART 3: VISUALIZATION
# ==============================================================================
def visualize_route(inst, x, y, sol, title_suffix=""):
    if not sol: return
    plt.figure(figsize=(10, 8))
    
    # Nodes
    for i, (cx, cy) in inst.coords.items():
        if i == inst.depot_start or i == inst.depot_end:
            c, m, lbl = 'k', 's', 'Depot'
        elif i in inst.P:
            c, m, lbl = 'b', 'o', f'P{i}'
        else:
            c, m, lbl = 'r', '^', f'D{i}'
        plt.scatter(cx, cy, c=c, marker=m, s=150, zorder=5)
        plt.text(cx+0.3, cy+0.3, lbl, fontsize=9)

    # Truck
    for v, i, j in x:
        if sol.get_value(x[v, i, j]) > 0.9:
            p1 = inst.coords[i]
            p2 = inst.coords[j]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2, label='Truck' if i==inst.depot_start else "")
            # Arrow
            plt.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.8, (p2[1]-p1[1])*0.8, head_width=0.3, fc='k', ec='k')

    # Drone
    drone_plotted = False
    for v, i, j, k in y:
        if sol.get_value(y[v, i, j, k]) > 0.9:
            p1 = inst.coords[i]; p2 = inst.coords[j]; p3 = inst.coords[k]
            lbl = 'Drone' if not drone_plotted else ""
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=2, label=lbl)
            plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'g--', linewidth=2)
            drone_plotted = True

    plt.title(f"DAPDP Solution: {title_suffix}\nObj: {sol.objective_value:.4f}")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def visualize_sensitivity(x_vals, y_vals, param_name):
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.title(f"Sensitivity Analysis: {param_name}")
    plt.xlabel(f"{param_name} Value")
    plt.ylabel("Total Operational Cost ($)")
    plt.grid(True)
    plt.show()

# ==============================================================================
# PART 4: MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
if __name__ == "__main__":
    print("========================================================")
    print("   DRONE-ASSISTED PICKUP AND DELIVERY PROBLEM (DAPDP)   ")
    print("========================================================")
    
    # -------------------------------------------------------
    # STEP 1: VALIDATION (Small scale checking against paper)
    # -------------------------------------------------------
    print("\n--- STEP 1: VALIDATION (Small Instance n=3) ---")
    val_inst = Instance(n_requests=3, grid_size=5, seed=55)
    
    # Solve Truck Only (Force endurance = 0 to kill drone)
    original_endurance = val_inst.endurance
    val_inst.endurance = 0
    sol_truck, _, _ = solve_dapdp(val_inst, verbose=False)
    cost_truck = sol_truck.objective_value if sol_truck else 9999
    
    # Solve DAPDP
    val_inst.endurance = original_endurance # Restore
    sol_drone, _, _ = solve_dapdp(val_inst, verbose=False)
    cost_drone = sol_drone.objective_value if sol_drone else 9999
    
    print(f"Truck Only Cost: ${cost_truck:.4f}")
    print(f"DAPDP Cost:      ${cost_drone:.4f}")
    if cost_drone < cost_truck:
        print(f"[PASS] Savings found: {(1 - cost_drone/cost_truck)*100:.2f}%")
    else:
        print("[FAIL] No savings found (Check constraints)")

    # -------------------------------------------------------
    # STEP 2: LARGE SCENARIO VISUALIZATION
    # -------------------------------------------------------
    print("\n--- STEP 2: LARGE SCENARIO (n=6, 30x30 grid) ---")
    large_inst = Instance(n_requests=4, grid_size=30, seed=42)
    sol, x_vars, y_vars = solve_dapdp(large_inst, verbose=True, time_limit=180)
    
    if sol:
        print(f"Optimal Objective Found: {sol.objective_value:.4f}")
        visualize_route(large_inst, x_vars, y_vars, sol, title_suffix="n=6, 30x30 Grid")
    else:
        print("No solution found for large scenario within time limit.")

# -------------------------------------------------------
    # STEP 3: SENSITIVITY ANALYSIS (FIXED)
    # -------------------------------------------------------
    print("\n--- STEP 3: SENSITIVITY ANALYSIS (Endurance) ---")
    
    # CHANGE 1: Grid Size increased to 50 (makes battery life critical)
    # CHANGE 2: Seed set to 42 (guarantees drone-eligible light packages)
    sens_inst = Instance(n_requests=4, grid_size=50, seed=42)
    
    endurance_values = [20, 30, 40, 50, 60, 80]
    results = []
    
    print(f"{'Endurance':<10} | {'Cost'}")
    print("-" * 20)
    
    for e in endurance_values:
        sens_inst.endurance = e
        # Relax time limit slightly for the larger grid
        s_sol, _, _ = solve_dapdp(sens_inst, verbose=False, time_limit=60)
        
        if s_sol:
            val = s_sol.objective_value
            results.append(val)
            print(f"{e:<10} | {val:.4f}")
        else:
            results.append(None)
            print(f"{e:<10} | No Feasible Sol")
            
    # Filter and Plot
    valid_x = [x for i, x in enumerate(endurance_values) if results[i] is not None]
    valid_y = [y for y in results if y is not None]
    
    if valid_y:
        visualize_sensitivity(valid_x, valid_y, "Drone Endurance (min)")
    
    print("\nExecution Complete.")
