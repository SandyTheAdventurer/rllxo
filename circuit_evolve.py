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

import matplotlib
matplotlib.use("Agg")
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

def _connected_to_vplus(top: Topology) -> set:
    """Nodes connected to V+ (node 0) through any resistors."""
    adj = {i: set() for i in range(top.num_nodes)}
    for r in top.resistors:
        adj[r.n1].add(r.n2)
        adj[r.n2].add(r.n1)
    seen, queue = {0}, [0]
    while queue:
        for nb in adj[queue.pop()]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return seen


def branch_currents(top: Topology, v_in: float = 1.0) -> list[float] | None:
    """Current through each resistor with v_in across V+ (node 0) and GND (node 1), or None if they are not
    connected. Resistors outside the V+/GND component carry no current."""
    comp = _connected_to_vplus(top)
    if 1 not in comp:
        return None
    unknowns = sorted(comp - {0, 1})
    nmap = {n: i for i, n in enumerate(unknowns)}
    G, I = np.zeros((len(unknowns), len(unknowns))), np.zeros(len(unknowns))
    fixed = {0: v_in, 1: 0.0}
    for r in top.resistors:
        if r.n1 not in comp:
            continue
        g = 1.0 / max(r.value, 1e-9)
        for a, b in ((r.n1, r.n2), (r.n2, r.n1)):
            if a in nmap:
                G[nmap[a], nmap[a]] += g
                if b in nmap:
                    G[nmap[a], nmap[b]] -= g
                else:
                    I[nmap[a]] += g * fixed[b]
    try:
        v = np.linalg.solve(G, I) if unknowns else np.zeros(0)
    except np.linalg.LinAlgError:
        return None
    volt = {**fixed, **{n: v[i] for n, i in nmap.items()}}
    return [(volt[r.n1] - volt[r.n2]) / max(r.value, 1e-9) if r.n1 in comp else 0.0 for r in top.resistors]


def prune_unused(top: Topology) -> Topology:
    """Remove resistors that carry no current between V+ and GND: dangling branches, floating parts and
    resistors not connected to the terminals. (The previous reachability test kept a dangling branch off
    V+, because GND reaches it through V+.)"""
    cur = branch_currents(top)
    if cur is None:
        return top  # V+ and GND are not connected; solve_dc reports the failure
    peak = max((abs(c) for c in cur), default=0.0)
    kept = [r for r, c in zip(top.resistors, cur) if abs(c) > 1e-9 * peak]
    if not kept:
        return top
    old_to_new = {0: 0, 1: 1}
    for r in kept:
        for n in (r.n1, r.n2):
            if n not in old_to_new:
                old_to_new[n] = len(old_to_new)
    return Topology([Resistor(old_to_new[r.n1], old_to_new[r.n2], r.value) for r in kept], len(old_to_new))


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

def plot_topology(top, filename: str = "topology.pdf", target: float = TARGET_RESISTANCE):
    """Draw resistor network as a dark blueprint graph with glowing edges and a clean side panel."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    top = prune_unused(top)  # Ensure only nodes on V+→GND paths are drawn

    # ── Palette ───────────────────────────────────────────────────────────
    BG_DARK  = "#FFFFFF"
    BG_PANEL = "#F8F9FA"
    GRID_COL = "#E5E9EF"
    ACCENT   = "#1A6EBD"
    GOLD     = "#C8850A"
    TEXT_LT  = "#1A1A2E"
    TEXT_DIM = "#6B7280"
    RESISTOR_COLORS = ["#1A6EBD", "#D94F3D", "#2E8B57", "#C8850A", "#7B3FA0", "#2976A8"]
    NODE_COLORS = {0: "#D94F3D", 1: "#374151"}  # V+, GND
    tnr_candidates = [
        "/usr/share/fonts/TTF/Times.TTF",
        "/usr/share/fonts/TTF/Timesi.TTF",
        "/usr/local/share/fonts/Times New Roman.ttf",
        os.path.expanduser("~/.local/share/fonts/Times New Roman.ttf"),
    ]
    tnr_path = next((p for p in tnr_candidates if os.path.exists(p)), None)
    if tnr_path:
        font_prop = font_manager.FontProperties(fname=tnr_path)
    else:
        font_prop = font_manager.FontProperties(family="serif")
    font_scale = 14.0 / 10.0
    fs = lambda size: size * font_scale

    # ── Figure & axes ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.8, 1], wspace=0.04,
                           left=0.02, right=0.98, top=0.93, bottom=0.0)
    ax       = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])

    ax.set_facecolor(BG_DARK)
    ax.axis('off')
    ax_panel.set_facecolor(BG_PANEL)
    ax_panel.axis('off')
    for spine in ax_panel.spines.values():
        spine.set_color(GRID_COL); spine.set_linewidth(1.5)

    XLIM, YLIM = (-1.7, 1.7), (-0.6, 1.5)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect('equal')

    # Grid intentionally removed for publication-style clean layout.

    # ── Node positions ────────────────────────────────────────────────────
    n   = top.num_nodes
    pos = {0: (-1.15, 0.0), 1: (1.15, 0.0)}
    for i in range(2, n):
        angle  = np.pi * (0.2 + 0.6 * (i - 2) / max(1, n - 3))
        pos[i] = (np.cos(angle) * 0.85, np.sin(angle) * 0.85 + 0.2)

    # ── Resistor edges ────────────────────────────────────────────────────
    edge_counts, edge_idx = {}, {}
    for r in top.resistors:
        key = (min(r.n1, r.n2), max(r.n1, r.n2))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    label_positions = []
    for i, r in enumerate(top.resistors):
        key   = (min(r.n1, r.n2), max(r.n1, r.n2))
        idx   = edge_idx.get(key, 0); edge_idx[key] = idx + 1
        x1, y1 = pos[r.n1]; x2, y2 = pos[r.n2]
        color = RESISTOR_COLORS[i % len(RESISTOR_COLORS)]
        rad   = 0.6 * (idx - (edge_counts[key] - 1) / 2) if edge_counts[key] > 1 else 0.0

        # Clean, rounded resistor segment (single stroke)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            color=color, lw=12.0, alpha=1.0, zorder=2, arrowstyle="-",
            capstyle="round", joinstyle="round"
        ))

        norm = max(0.01, np.hypot(x2 - x1, y2 - y1))
        mx   = (x1 + x2) / 2 + rad * 0.35 * (y2 - y1) / norm
        my   = (y1 + y2) / 2 - rad * 0.35 * (x2 - x1) / norm
        label_positions.append((mx, my, f"R{i}", color))

    for mx, my, lbl, col in label_positions:
        ax.text(mx, my, lbl, fontsize=fs(10.0), ha='center', va='center',
            color=col, fontweight='bold', zorder=15, fontproperties=font_prop,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=col, alpha=0.92, lw=1.2))

    # ── Nodes ─────────────────────────────────────────────────────────────
    for node, (x, y) in pos.items():
        color = NODE_COLORS.get(node, "#000000")
        label = {0: "V+", 1: "GND"}.get(node, f"N{node}")

        # Solid endpoint marker
        ax.scatter(x, y, s=260, c=color, alpha=1.0, zorder=9,
                   linewidths=1.5, edgecolors="white")

        ox = -0.18 if node == 0 else (0.18 if node == 1 else 0)
        oy = 0 if node in (0, 1) else 0.20
        ax.text(x + ox, y + oy, label, fontsize=fs(11), fontweight='bold',
                ha='right' if node == 0 else ('left' if node == 1 else 'center'),
                va='center' if node in (0, 1) else 'bottom',
                color=color, fontproperties=font_prop,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    row_h = min(0.10, 0.68 / max(1, len(top.resistors)))
    list_h = row_h * max(1, len(top.resistors))
    y_pos = min(0.92, 0.5 + list_h / 2)
    for i, r in enumerate(top.resistors):
        color = RESISTOR_COLORS[i % len(RESISTOR_COLORS)]
        ty    = y_pos - 0.01
        ax_panel.add_patch(FancyBboxPatch(
            (0.04, ty - 0.018), 0.10, 0.036, boxstyle="round,pad=0.005",
            facecolor=color, edgecolor='none',
            transform=ax_panel.transAxes, zorder=5))
        ax_panel.text(0.20, ty + 0.010, f"R{i}  {int(r.value):,}\u03a9",
                  fontsize=fs(10), va='center', ha='left', color=TEXT_LT,
                      fontproperties=font_prop, fontweight='bold',
                      transform=ax_panel.transAxes)
        ax_panel.text(0.20, ty - 0.018, f"node {r.n1} \u2192 {r.n2}",
                  fontsize=fs(7.5), va='center', ha='left', color=TEXT_DIM,
                      fontproperties=font_prop, transform=ax_panel.transAxes)
        y_pos -= row_h

    # ── Resistance summary ────────────────────────────────────────────────
    r_eq = solve_dc(top)
    summary_top = max(0.10, y_pos - 0.03)
    ax_panel.axhline(summary_top + 0.04, color=GRID_COL, lw=0.8, xmin=0.05, xmax=0.95)
    ax_panel.text(0.08, summary_top, "Target",  fontsize=fs(8),  va='center', color=TEXT_DIM,
                  fontproperties=font_prop, transform=ax_panel.transAxes)
    ax_panel.text(0.92, summary_top, f"{target:.1f} \u03a9", fontsize=fs(9), fontweight='bold',
                  va='center', ha='right', color=ACCENT,
                  fontproperties=font_prop, transform=ax_panel.transAxes)
    r_eq_str = f"{r_eq:.1f} \u03a9" if r_eq is not None else "N/A"
    r_eq_col  = GOLD if r_eq is not None else "#D94F3D"
    ax_panel.text(0.08, summary_top - 0.06, "Evolved", fontsize=fs(8),  va='center', color=TEXT_DIM,
                  fontproperties=font_prop, transform=ax_panel.transAxes)
    ax_panel.text(0.92, summary_top - 0.06, r_eq_str, fontsize=fs(9), fontweight='bold',
                  va='center', ha='right', color=r_eq_col,
                  fontproperties=font_prop, transform=ax_panel.transAxes)

    plt.savefig(filename, format='pdf', bbox_inches='tight',
                facecolor='white', dpi=180)
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
    
    plot_topology(best, "topology.pdf", target=args.target)