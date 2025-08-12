"""
two_body_blackhole_sim.py

- RK4 two-body gravitational simulation.
- Collision detection and merge -> black hole creation.
- Saves cache files: blackhole.json and trajectories.json (if collision occurs).
- Produces a simple Matplotlib animation.
"""

import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Simulation Parameters
# -----------------------
G = 6.67430e-11        # gravitational constant (SI)
DT = 100.0             # time step (seconds)
STEPS = 5000           # max simulation steps
COLLISION_DISTANCE_FACTOR = 1.0  # collision when distance < (r1 + r2) * factor

# Bodies: [mass(kg), radius(m), position(vec3), velocity(vec3)]
# Example: star (body 0) and planet (body 1)
bodies = [
    {
        "name": "body0",
        "mass": 2.0e30,                    # ~1 solar mass
        "radius": 7.0e8,                   # arbitrary radius (m)
        "pos": np.array([0.0, 0.0, 0.0]),
        "vel": np.array([0.0, 0.0, 0.0])
    },
    {
        "name": "body1",
        "mass": 6.0e24,                    # ~Earth mass
        "radius": 6.4e6,
        "pos": np.array([1.5e11, 0.0, 0.0]), # 1 AU
        "vel": np.array([0.0, 29780.0, 0.0]) # orbital speed ~29.8 km/s
    }
]

# Output filenames (cache)
CACHE_DIR = "cache"
BLACKHOLE_JSON = os.path.join(CACHE_DIR, "blackhole.json")
TRAJECTORIES_JSON = os.path.join(CACHE_DIR, "trajectories.json")

# -----------------------
# Utilities & RK4
# -----------------------
def gravitational_acceleration(target_index, bodies_state):
    """Compute acceleration on bodies_state[target_index] from others.
    bodies_state: list of dicts with mass and pos arrays.
    Returns np.array(3).
    """
    acc = np.zeros(3)
    p = bodies_state[target_index]["pos"]
    for j, other in enumerate(bodies_state):
        if j == target_index:
            continue
        r_vec = other["pos"] - p
        r = np.linalg.norm(r_vec) + 1e-12
        acc += G * other["mass"] * (r_vec / (r**3))
    return acc

def rk4_step(bodies_state, dt):
    """Advance all bodies by dt using RK4 (coupled)."""
    # Save original state
    n = len(bodies_state)
    # Prepare arrays
    pos0 = np.array([b["pos"] for b in bodies_state])
    vel0 = np.array([b["vel"] for b in bodies_state])
    mass = np.array([b["mass"] for b in bodies_state])

    # helper to compute accel array
    def acc_array(positions):
        accs = np.zeros_like(positions)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec) + 1e-12
                accs[i] += G * mass[j] * (r_vec / (r**3))
        return accs

    k1_v = acc_array(pos0)
    k1_p = vel0

    k2_p = vel0 + 0.5 * dt * k1_v
    pos_k2 = pos0 + 0.5 * dt * k1_p
    k2_v = acc_array(pos_k2)

    k3_p = vel0 + 0.5 * dt * k2_v
    pos_k3 = pos0 + 0.5 * dt * k2_p
    k3_v = acc_array(pos_k3)

    k4_p = vel0 + dt * k3_v
    pos_k4 = pos0 + dt * k3_p
    k4_v = acc_array(pos_k4)

    pos_new = pos0 + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    vel_new = vel0 + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    # write back
    for i, b in enumerate(bodies_state):
        b["pos"] = pos_new[i]
        b["vel"] = vel_new[i]

# -----------------------
# Collision detection & merge
# -----------------------
def check_collision_and_merge(bodies_state):
    """Detect collision (pairwise). If collision occurs, merge bodies and return (True, merged_body_index)
       Otherwise return (False, None).
    """
    n = len(bodies_state)
    for i in range(n):
        for j in range(i+1, n):
            pi = bodies_state[i]["pos"]
            pj = bodies_state[j]["pos"]
            dist = np.linalg.norm(pi - pj)
            # collision if distance less than sum of radii * factor
            if dist <= (bodies_state[i]["radius"] + bodies_state[j]["radius"]) * COLLISION_DISTANCE_FACTOR:
                # merge j into i (create new merged body)
                mi = bodies_state[i]["mass"]
                mj = bodies_state[j]["mass"]
                new_mass = mi + mj
                # conserve momentum: v_new = (mi*vi + mj*vj) / (mi+mj)
                v_new = (mi * bodies_state[i]["vel"] + mj * bodies_state[j]["vel"]) / new_mass
                # position at center of mass
                pos_new = (mi * bodies_state[i]["pos"] + mj * bodies_state[j]["pos"]) / new_mass
                # approximate radius of merged object by volume conservation (assume uniform density)
                ri = bodies_state[i]["radius"]
                rj = bodies_state[j]["radius"]
                vol = (4/3) * math.pi * (ri**3 + rj**3)
                r_new = ((3.0 * vol) / (4.0 * math.pi))**(1.0/3.0)
                merged = {
                    "name": f"merged_{i}_{j}",
                    "mass": new_mass,
                    "radius": r_new,
                    "pos": pos_new,
                    "vel": v_new
                }
                # remove j and i, append merged
                new_bodies = []
                for k, b in enumerate(bodies_state):
                    if k == i or k == j:
                        continue
                    new_bodies.append(b)
                new_bodies.append(merged)
                return True, new_bodies, {"merged_from": (i, j), "mass": new_mass, "pos": pos_new.tolist(), "time": None}
    return False, bodies_state, None

# -----------------------
# Run simulation
# -----------------------
def run_simulation():
    state = [ {**b, "pos": b["pos"].astype(float), "vel": b["vel"].astype(float)} for b in bodies ]
    traj = [ [] for _ in range(len(state)) ]
    collision_info = None

    for step in range(STEPS):
        # record positions
        for i, b in enumerate(state):
            traj[i].append(b["pos"].tolist())

        # advance
        rk4_step(state, DT)

        # check collision
        collided, new_state, info = check_collision_and_merge(state)
        if collided:
            info["time"] = step * DT
            collision_info = info
            state = new_state
            # continue sim a bit after merge (optional)
            # expand traj arrays to fit new state size
            # convert traj to new length
            new_traj = []
            for i in range(len(state)):
                # for merged one, create combined trajectory from previous two
                if state[i]["name"].startswith("merged_"):
                    a, bidx = info["merged_from"]
                    # combine previous two trajectories up to collision step
                    comb = traj[a][:] + traj[bidx][:]
                    new_traj.append(comb)
                else:
                    # find original index by name mapping
                    # best effort: if name matches existing old bodies
                    new_traj.append(traj[min(i, len(traj)-1)])
            traj = new_traj
            # break or continue? We'll continue to simulate for a bit to let black hole settle
            # We'll run a short post-merge phase:
            for post_step in range(500):
                for i, b in enumerate(state):
                    traj[i].append(b["pos"].tolist())
                rk4_step(state, DT)
            break

    # Save cache if collision happened
    if collision_info:
        os.makedirs(CACHE_DIR, exist_ok=True)
        # blackhole data
        blackhole_data = {
            "merged_mass": collision_info["mass"],
            "merged_pos": collision_info["pos"],
            "formation_time_s": collision_info["time"],
            "note": "This is a merged object — treat as black hole candidate (visual demo)."
        }
        with open(BLACKHOLE_JSON, "w") as f:
            json.dump(blackhole_data, f, indent=2)

        # trajectories (may be big — use caution)
        save_traj = {
            "trajectories": traj,
            "names": [b.get("name", "") for b in state]
        }
        with open(TRAJECTORIES_JSON, "w") as f:
            json.dump(save_traj, f)
        print(f"Collision/merge occurred at t={collision_info['time']} s — cache written to {CACHE_DIR}/")

    else:
        print("No collision occurred in simulation.")

    return traj, state, collision_info

# -----------------------
# Visualization (2D plot of xy)
# -----------------------
def animate_simulation(traj, state, collision_info):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_facecolor("black")
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    # determine limits
    all_points = np.concatenate([np.array(t) for t in traj], axis=0)
    xmin, ymin = np.min(all_points[:,0]), np.min(all_points[:,1])
    xmax, ymax = np.max(all_points[:,0]), np.max(all_points[:,1])
    margin = 0.2 * max(xmax-xmin, ymax-ymin, 1.0)
    ax.set_xlim(xmin-margin, xmax+margin)
    ax.set_ylim(ymin-margin, ymax+margin)
    ax.set_title("Two-body simulation (XY plane)")

    lines = []
    points = []
    colors = ['cyan','yellow','magenta','orange','white']
    for i in range(len(traj)):
        ln, = ax.plot([], [], color=colors[i % len(colors)], lw=1)
        pt, = ax.plot([], [], 'o', color=colors[i % len(colors)], markersize=5)
        lines.append(ln)
        points.append(pt)

    def init():
        for ln, pt in zip(lines, points):
            ln.set_data([], [])
            pt.set_data([], [])
        return lines + points

    max_frame = max(len(t) for t in traj)
    def update(frame):
        for i, t in enumerate(traj):
            upto = min(frame, len(t)-1)
            arr = np.array(t[:upto+1])
            lines[i].set_data(arr[:,0], arr[:,1])
            points[i].set_data(arr[-1,0], arr[-1,1])
        return lines + points

    ani = FuncAnimation(fig, update, frames=max_frame, init_func=init, interval=20, blit=True)
    plt.show()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    traj, final_state, collision_info = run_simulation()
    animate_simulation(traj, final_state, collision_info)
