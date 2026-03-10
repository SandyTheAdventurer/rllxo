#!/usr/bin/env python3
"""Resistor network evolution with learned crossover.

Simple benchmark: evolve resistor topology to achieve target resistance.
Outputs topology diagram and I-V curve as PDFs.

Usage:
    python resistor_evolve.py                    # Random crossover
    python resistor_evolve.py --learned          # RL-guided crossover
"""
from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from main import SubtreeSelector, set_global_seed, DEVICE

# =============================================================================
# Topology representation
# =============================================================================

@dataclass
class Resistor:
    n1: int
    n2: int
    value: float  # Ohms

@dataclass
class Topology:
    resistors: list[Resistor]
    num_nodes: int

    def clone(self) -> Topology:
        return copy.deepcopy(self)
    
    def __str__(self) -> str:
        r = ", ".join(f"R({b.n1}-{b.n2}:{b.value:.1f}Ω)" for b in self.resistors)
        return f"Topology(n={self.num_nodes}; {r})"

def _sig(top: Topology) -> tuple:
    return (top.num_nodes,
            tuple(sorted((r.n1, r.n2, round(r.value, 1)) for r in top.resistors)))

# =============================================================================
# DC Analysis (simple nodal analysis)
# =============================================================================

def solve_dc(top: Topology, v_in: float = 1.0) -> float | None:
    """
    Apply v_in between node 0 (V+) and node 1 (GND).
    Return equivalent resistance seen from input, or None if failed.
    
    Node 0 = V+ (voltage source)
    Node 1 = GND (reference)
    """
    if top.num_nodes < 2:
        return None
    
    # Build conductance matrix for nodes 2..N-1 (node 0 is V+, node 1 is GND)
    unknowns = list(range(2, top.num_nodes))
    if not unknowns:
        # Only 2 nodes: direct connection
        g_total = sum(1.0 / max(r.value, 1e-9) for r in top.resistors 
                     if {r.n1, r.n2} == {0, 1})
        if g_total < 1e-12:
            return None
        return 1.0 / g_total
    
    nmap = {n: i for i, n in enumerate(unknowns)}
    nvar = len(unknowns)
    
    G = np.zeros((nvar, nvar))
    I = np.zeros(nvar)
    
    # Fixed voltages
    v_fixed = {0: v_in, 1: 0.0}
    
    for r in top.resistors:
        g = 1.0 / max(r.value, 1e-9)
        n1, n2 = r.n1, r.n2
        
        i1 = nmap.get(n1)
        i2 = nmap.get(n2)
        
        if i1 is not None:
            G[i1, i1] += g
            if i2 is not None:
                G[i1, i2] -= g
            elif n2 in v_fixed:
                I[i1] += g * v_fixed[n2]
        
        if i2 is not None:
            G[i2, i2] += g
            if i1 is not None:
                G[i2, i1] -= g
            elif n1 in v_fixed:
                I[i2] += g * v_fixed[n1]
    
    # Solve for node voltages
    try:
        v_nodes = np.linalg.solve(G, I)
    except np.linalg.LinAlgError:
        return None
    
    if not np.all(np.isfinite(v_nodes)):
        return None
    
    # Calculate current from V+ node
    v_all = np.zeros(top.num_nodes)
    v_all[0] = v_in
    v_all[1] = 0.0
    for n, i in nmap.items():
        v_all[n] = v_nodes[i]
    
    i_total = 0.0
    for r in top.resistors:
        if r.n1 == 0:
            i_total += (v_all[0] - v_all[r.n2]) / max(r.value, 1e-9)
        elif r.n2 == 0:
            i_total += (v_all[0] - v_all[r.n1]) / max(r.value, 1e-9)
    
    if abs(i_total) < 1e-12:
        return None
    
    return v_in / i_total

# =============================================================================
# Fitness (Pareto: error vs complexity)
# =============================================================================

TARGET_RESISTANCE = 100.0  # Target: 100 ohms

def prune_unused(top: Topology) -> Topology:
    """Remove resistors not on any V+ to GND path."""
    # BFS to find nodes reachable from V+ (node 0)
    adj = {i: set() for i in range(top.num_nodes)}
    for r in top.resistors:
        adj[r.n1].add(r.n2)
        adj[r.n2].add(r.n1)
    
    def reachable(start: int) -> set:
        visited = {start}
        queue = [start]
        while queue:
            n = queue.pop(0)
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited
    
    from_vplus = reachable(0)
    from_gnd = reachable(1)
    useful_nodes = from_vplus & from_gnd
    
    # Keep only resistors where both nodes are useful
    kept = [r for r in top.resistors if r.n1 in useful_nodes and r.n2 in useful_nodes]
    if not kept:
        return top  # Can't prune everything
    
    # Renumber nodes
    old_to_new = {0: 0, 1: 1}
    next_id = 2
    for r in kept:
        for n in (r.n1, r.n2):
            if n not in old_to_new:
                old_to_new[n] = next_id
                next_id += 1
    
    pruned = Topology(
        [Resistor(old_to_new[r.n1], old_to_new[r.n2], r.value) for r in kept],
        next_id
    )
    return pruned

def evaluate_pareto(top: Topology, target: float = TARGET_RESISTANCE) -> tuple[float, int]:
    """Return (error, num_resistors) for Pareto ranking."""
    pruned = prune_unused(top)
    r_eq = solve_dc(pruned)
    if r_eq is None or r_eq <= 0:
        return (1e6, len(top.resistors))
    error = abs(r_eq - target) / target
    return (error, len(pruned.resistors))

def dominates(a: tuple[float, int], b: tuple[float, int]) -> bool:
    """True if a dominates b (better or equal in all, strictly better in one)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def pareto_rank(objs: list[tuple[float, int]]) -> list[int]:
    """Simple non-dominated sorting. Returns rank for each individual (0 = best)."""
    n = len(objs)
    dom_count = [sum(1 for j in range(n) if i != j and dominates(objs[j], objs[i])) for i in range(n)]
    
    ranks = [0] * n
    assigned = [False] * n
    rank = 0
    remaining = n
    
    while remaining > 0:
        for i in range(n):
            if not assigned[i] and dom_count[i] == 0:
                ranks[i] = rank
                assigned[i] = True
                remaining -= 1
        # Decrease counts for next round
        for i in range(n):
            if assigned[i] and ranks[i] == rank:
                for j in range(n):
                    if not assigned[j] and dominates(objs[i], objs[j]):
                        dom_count[j] -= 1
        rank += 1
    
    return ranks

def crowding_distance(objs: list[tuple[float, int]], indices: list[int]) -> dict[int, float]:
    """Crowding distance for diversity preservation."""
    if len(indices) <= 2:
        return {i: float('inf') for i in indices}
    
    dist = {i: 0.0 for i in indices}
    for dim in range(2):
        s = sorted(indices, key=lambda i: objs[i][dim])
        dist[s[0]] = dist[s[-1]] = float('inf')
        rng = objs[s[-1]][dim] - objs[s[0]][dim]
        if rng > 1e-12:
            for k in range(1, len(s) - 1):
                dist[s[k]] += (objs[s[k+1]][dim] - objs[s[k-1]][dim]) / rng
    return dist

def evaluate(top: Topology, target: float = TARGET_RESISTANCE) -> float:
    """Single objective for backward compat (weighted sum)."""
    err, comp = evaluate_pareto(top, target)
    return err + 0.01 * comp

# =============================================================================
# Genetic operators
# =============================================================================

R_VALUES = [10, 22, 47, 100]  # Standard resistor values

def _rand_pair(n: int) -> tuple[int, int]:
    a = random.randrange(n)
    b = random.randrange(n - 1)
    return (a, b + 1) if b >= a else (a, b)

def _repair(top: Topology):
    for r in top.resistors:
        if r.n1 == r.n2:
            r.n1, r.n2 = _rand_pair(top.num_nodes)
        if r.n1 > r.n2:
            r.n1, r.n2 = r.n2, r.n1

def random_topology(nodes: int = 4, nr: int = 4) -> Topology:
    top = Topology(
        [Resistor(*_rand_pair(nodes), random.choice(R_VALUES)) for _ in range(nr)],
        nodes,
    )
    _repair(top)
    return top

def mutate(top: Topology, rate: float = 0.3) -> Topology:
    out = top.clone()
    for r in out.resistors:
        if random.random() < rate:
            # Mutate connection
            r.n1, r.n2 = _rand_pair(out.num_nodes)
        if random.random() < rate:
            # Mutate value
            r.value = random.choice(R_VALUES)
    _repair(out)
    return out

def crossover(a: Topology, b: Topology) -> tuple[Topology, Topology]:
    c1, c2 = a.clone(), b.clone()
    for i in range(min(len(c1.resistors), len(c2.resistors))):
        if random.random() < 0.5:
            c1.resistors[i], c2.resistors[i] = c2.resistors[i], c1.resistors[i]
    _repair(c1)
    _repair(c2)
    return c1, c2

# =============================================================================
# RL crossover
# =============================================================================

FEATURE_DIM = 8

def _resistor_features(top: Topology, r: Resistor, idx: int) -> torch.Tensor:
    deg = np.bincount([res.n1 for res in top.resistors] + [res.n2 for res in top.resistors], 
                      minlength=top.num_nodes).astype(float)
    maxd = max(deg.max(), 1)
    return torch.tensor([
        r.n1 / max(top.num_nodes - 1, 1), r.n2 / max(top.num_nodes - 1, 1),
        deg[r.n1] / maxd, deg[r.n2] / maxd,
        r.n1 in {0, 1} or r.n2 in {0, 1},
        np.log10(r.value) / 3, idx / max(len(top.resistors) - 1, 1), len(top.resistors) / 10,
    ], dtype=torch.float32, device=DEVICE)

def learned_crossover(a: Topology, b: Topology, selector: SubtreeSelector, temp: float,
                      history: list | None = None) -> tuple[Topology, Topology]:
    c1, c2 = a.clone(), b.clone()
    if not c1.resistors or not c2.resistors:
        return c1, c2
    
    feat_a = torch.stack([_resistor_features(c1, r, i) for i, r in enumerate(c1.resistors)])
    feat_b = torch.stack([_resistor_features(c2, r, i) for i, r in enumerate(c2.resistors)])
    
    with torch.no_grad():
        logits_a, _ = selector(feat_a, return_value=True)
        logits_b, _ = selector(feat_b, return_value=True)
        pair = torch.clamp((logits_a.unsqueeze(1) + logits_b.unsqueeze(0)) / max(temp, 1e-6), -10, 10)
        idx = int(torch.multinomial(torch.softmax(pair.flatten(), 0), 1).item())
        ai, bi = idx // len(c2.resistors), idx % len(c2.resistors)
    
    c1.resistors[ai], c2.resistors[bi] = c2.resistors[bi], c1.resistors[ai]
    _repair(c1); _repair(c2)
    
    if history is not None:
        history.append({"feat_a": feat_a, "feat_b": feat_b, "ai": ai, "bi": bi,
                        "c1": c1.clone(), "c2": c2.clone()})
    return c1, c2

def update_selector(selector: SubtreeSelector, optimizer: optim.Optimizer,
                    history: list, eval_fn) -> None:
    if not history:
        return
    
    # Compute rewards
    base = eval_fn(history[0]["c1"])
    rewards = torch.tensor([base - min(eval_fn(r["c1"]), eval_fn(r["c2"])) for r in history],
                          dtype=torch.float32, device=DEVICE)
    if rewards.std() < 1e-12:
        history.clear()
        return
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Compute log probs
    log_probs = []
    for rec in history:
        logits_a, _ = selector(rec["feat_a"].to(DEVICE), return_value=True)
        logits_b, _ = selector(rec["feat_b"].to(DEVICE), return_value=True)
        pair = (logits_a.unsqueeze(1) + logits_b.unsqueeze(0)).flatten()
        idx = rec["ai"] * rec["feat_b"].shape[0] + rec["bi"]
        log_probs.append(torch.log_softmax(pair, dim=0)[idx])
    
    loss = -(torch.stack(log_probs) * rewards).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
    optimizer.step()
    history.clear()

# =============================================================================
# Evolution
# =============================================================================

def evolve(
    target: float = TARGET_RESISTANCE,
    pop_size: int = 100,
    gens: int = 30,
    nodes: int = 4,
    nr: int = 4,
    cx_rate: float = 0.85,
    mut_rate: float = 0.3,
    tour_k: int = 3,
    elite: int = 2,
    learned: bool = False,
    sel_lr: float = 3e-4,
    sel_freq: int = 3,
    temp: float = 0.6,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[Topology, float, list[Topology]]:
    """Returns (best, best_score, pareto_front)."""
    set_global_seed(seed)
    cache: dict[tuple, tuple[float, int]] = {}
    
    def eval_cached(top: Topology) -> tuple[float, int]:
        key = _sig(top)
        if key not in cache:
            cache[key] = evaluate_pareto(top, target)
        return cache[key]
    
    def scalar_score(top: Topology) -> float:
        err, comp = eval_cached(top)
        return err + 0.01 * comp
    
    pop = [random_topology(nodes, nr) for _ in range(pop_size)]
    
    selector = history = optimizer = None
    if learned:
        selector = SubtreeSelector(feature_dim=FEATURE_DIM, hidden_dim=32,
                                   mlp_layers=2, use_critic=True).to(DEVICE)
        optimizer = optim.Adam(selector.parameters(), lr=sel_lr)
        history = []
    
    best, best_score = pop[0], float("inf")
    
    for gen in range(gens):
        objs = [eval_cached(t) for t in pop]
        ranks = pareto_rank(objs)
        
        # Track best
        for i, t in enumerate(pop):
            score = objs[i][0] + 0.01 * objs[i][1]
            if score < best_score:
                best_score, best = score, t.clone()
        
        if verbose and (gen == 0 or (gen + 1) % 5 == 0):
            pruned = prune_unused(best)
            print(f"Gen {gen+1:3d} | error={eval_cached(best)[0]:.4f} | R={len(pruned.resistors)} | R_eq={solve_dc(pruned):.1f}Ω | front={ranks.count(0)}")
        
        if learned and history and gen % sel_freq == 0:
            update_selector(selector, optimizer, history, scalar_score)
        
        # Crowding for diversity
        crowding = {}
        for r in range(max(ranks) + 1):
            crowding.update(crowding_distance(objs, [i for i in range(pop_size) if ranks[i] == r]))
        
        # Selection key: rank first, then crowding (higher = better)
        key = lambda i: (ranks[i], -crowding.get(i, 0))
        next_pop = [pop[i].clone() for i in sorted(range(pop_size), key=key)[:elite]]
        
        def select():
            cands = random.sample(range(pop_size), min(tour_k, pop_size))
            return pop[min(cands, key=key)].clone()
        
        while len(next_pop) < pop_size:
            p1, p2 = select(), select()
            if random.random() < cx_rate:
                c1, c2 = learned_crossover(p1, p2, selector, temp, history) if learned else crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            next_pop.append(mutate(c1, mut_rate))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2, mut_rate))
        pop = next_pop
    
    # Final Pareto front
    final_objs = [eval_cached(t) for t in pop]
    final_ranks = pareto_rank(final_objs)
    front = {_sig(prune_unused(pop[i])): prune_unused(pop[i]) for i in range(pop_size) if final_ranks[i] == 0}
    
    return prune_unused(best), best_score, list(front.values())

# =============================================================================
# Plotting
# =============================================================================

def plot_topology(top: Topology, filename: str = "topology.pdf"):
    """Draw resistor network as a clean graph with values in side panel."""
    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(12, 7), 
                                         gridspec_kw={'width_ratios': [3, 1]})
    
    # Circular layout for nodes
    n = top.num_nodes
    pos = {}
    # Place V+ (0) on left, GND (1) on right, others in arc above
    pos[0] = (-1.0, 0.0)  # V+
    pos[1] = (1.0, 0.0)   # GND
    
    if n > 2:
        # Internal nodes in arc above
        for i in range(2, n):
            angle = np.pi * (0.3 + 0.4 * (i - 2) / max(1, n - 3))
            pos[i] = (np.cos(angle) * 0.8, np.sin(angle) * 0.8 + 0.2)
    
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.8, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Evolved Resistor Network", fontsize=16, fontweight='bold', pad=10)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A', '#228B22']
    
    # Group resistors by edge for parallel offset
    edge_counts = {}
    for r in top.resistors:
        key = (min(r.n1, r.n2), max(r.n1, r.n2))
        edge_counts[key] = edge_counts.get(key, 0) + 1
    edge_idx = {}
    
    # Draw edges (resistors)
    for i, r in enumerate(top.resistors):
        key = (min(r.n1, r.n2), max(r.n1, r.n2))
        idx = edge_idx.get(key, 0)
        edge_idx[key] = idx + 1
        
        x1, y1 = pos[r.n1]
        x2, y2 = pos[r.n2]
        
        # Curve parallel resistors
        rad = 0
        if edge_counts[key] > 1:
            rad = 0.3 * (idx - (edge_counts[key] - 1) / 2)
        
        color = colors[i % len(colors)]
        
        # Draw curved edge
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=f"arc3,rad={rad}",
                                color=color, lw=3, zorder=1)
        ax.add_patch(arrow)
        
        # Small label on edge (just R0, R1, etc.)
        mid_x = (x1 + x2) / 2 + rad * 0.3 * (y2 - y1) / max(0.01, np.sqrt((x2-x1)**2 + (y2-y1)**2))
        mid_y = (y1 + y2) / 2 - rad * 0.3 * (x2 - x1) / max(0.01, np.sqrt((x2-x1)**2 + (y2-y1)**2))
        ax.text(mid_x, mid_y, f"R{i}", fontsize=9, ha='center', va='center',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                         edgecolor=color, alpha=0.9, lw=1))
    
    # Draw nodes
    node_styles = {
        0: ('V+', 'red', 700),
        1: ('GND', 'black', 700),
    }
    for node, (x, y) in pos.items():
        if node in node_styles:
            label, color, size = node_styles[node]
        else:
            label, color, size = f'N{node}', '#4169E1', 500
        
        ax.scatter(x, y, s=size, c=color, zorder=10, edgecolors='white', linewidths=2)
        
        # Label position
        if node == 0:
            ax.text(x - 0.15, y, label, fontsize=12, fontweight='bold', 
                   ha='right', va='center', color=color)
        elif node == 1:
            ax.text(x + 0.15, y, label, fontsize=12, fontweight='bold',
                   ha='left', va='center', color=color)
        else:
            ax.text(x, y + 0.18, label, fontsize=11, fontweight='bold',
                   ha='center', va='bottom', color=color)
    
    # === Side panel: Resistor values ===
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    
    # Title
    ax_legend.text(0.5, 0.95, "Components", fontsize=14, fontweight='bold',
                   ha='center', va='top', transform=ax_legend.transAxes)
    
    # Draw color-coded legend
    y_pos = 0.85
    for i, r in enumerate(top.resistors):
        color = colors[i % len(colors)]
        ax_legend.plot([0.05, 0.2], [y_pos, y_pos], color=color, lw=4, 
                       transform=ax_legend.transAxes)
        ax_legend.text(0.25, y_pos, f"R{i}: {int(r.value)}Ω", fontsize=11,
                       va='center', ha='left', transform=ax_legend.transAxes,
                       color='black')
        ax_legend.text(0.25, y_pos - 0.04, f"   ({r.n1}→{r.n2})", fontsize=9,
                       va='center', ha='left', transform=ax_legend.transAxes,
                       color='gray')
        y_pos -= 0.12
    
    # Equivalent resistance at bottom
    r_eq = solve_dc(top)
    if r_eq is not None:
        ax_legend.text(0.5, 0.05, f"R_eq = {r_eq:.1f}Ω", fontsize=13, 
                       fontweight='bold', ha='center', va='bottom',
                       transform=ax_legend.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                edgecolor='orange', lw=2))
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_iv_curve(top: Topology, filename: str = "iv_curve.pdf"):
    """Plot I-V characteristic and save as PDF."""
    r_eq = solve_dc(top)
    if r_eq is None:
        print("Cannot solve circuit")
        return
    
    v_range = np.linspace(0, 10, 100)
    i_range = v_range / r_eq
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v_range, i_range * 1000, 'b-', lw=2)
    ax.set_xlabel('Voltage (V)', fontsize=12)
    ax.set_ylabel('Current (mA)', fontsize=12)
    ax.set_title(f'I-V Characteristic (R_eq = {r_eq:.1f}Ω)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(i_range) * 1000 * 1.1)
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve resistor network")
    parser.add_argument("--learned", action="store_true", help="Use RL-guided crossover")
    parser.add_argument("--target", type=float, default=53.9, help="Target resistance (Ω)")
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    mode = "learned" if args.learned else "random"
    print(f"Evolving resistor network ({mode} crossover, target={args.target}Ω)...")
    
    best, score, front = evolve(target=args.target, gens=args.gens, pop_size=args.pop,
                                learned=args.learned, seed=args.seed, verbose=True)
    
    r_eq = solve_dc(best)
    print(f"\nBest: R_eq = {r_eq:.2f}Ω (target: {args.target}Ω)")
    print(f"Topology: {best}")
    
    print(f"\nPareto front ({len(front)} solutions):")
    for i, t in enumerate(sorted(front, key=lambda x: len(x.resistors))):
        req = solve_dc(t)
        err = abs(req - args.target) / args.target if req else float('inf')
        print(f"  {i+1}. R={len(t.resistors)}, R_eq={req:.1f}Ω, error={err:.2%}")
    
    plot_topology(best, "topology.pdf")
    plot_iv_curve(best, "iv_curve.pdf")
