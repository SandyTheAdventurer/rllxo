#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random
import concurrent.futures as cf
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(**_kw):
        def _passthrough(fn): return fn
        return _passthrough
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from main import SubtreeSelector
try:
    from tqdm.auto import tqdm
except:
    def tqdm(iterable, **kwargs): return iterable

V_RMS = 230.0
FREQ_HZ = 50.0
V_PEAK = V_RMS * math.sqrt(2.0)
BASE_NODES = ["ac_p", "ac_n", "out_p", "out_n"]
INTERNAL_NODES = ["n1", "n2", "n3", "n4", "n5", "n6"]
COMP_TYPES = ("diode", "resistor", "capacitor")

# GA hyperparameters (edit here for quick tuning)
# Keep the defaults modest to avoid long simulation stalls; raise via CLI if stable.
POP_SIZE = 1000
GENS = 50
MIN_PARTS = 4
MAX_PARTS = 20
CROSSOVER_RATE = 0.85
ADD_RATE = 0.35
REMOVE_RATE = 0.25
NODE_MUTATION_RATE = 0.35
VALUE_JITTER = 0.25
TOURNAMENT_SIZE = 3
ELITISM_SIZE = 5
DC_WEIGHT = 0.5
TARGET_DC = 0.9 * V_PEAK
LOAD_RESISTANCE = 1000.0
SIZE_PENALTY = 1.0
SEED = 1024
SIM_PERIODS = 5
SIM_STEPS_PER_PERIOD = 400
DIODE_IS = 4.352e-9      # saturation current (A)
DIODE_N  = 1.906         # ideality factor
DIODE_VT = 0.02585       # thermal voltage at 25 °C
NR_TOL   = 1e-6          # Newton-Raphson tolerance
NR_MAX   = 40            # Newton-Raphson max iterations

# Selector hyperparameters
USE_SELECTOR = True
SELECTOR_HIDDEN = 128
SELECTOR_ACTIVATION = "leaky_relu"
SELECTOR_DROPOUT = 0.1
SELECTOR_LAYERS = 8
SELECTOR_TEMPERATURE = 1.0


# ---------------------------------------------------------------------------
#  Fast in-process MNA transient simulator  (Numba-accelerated)
# ---------------------------------------------------------------------------

def _build_node_map(genome: "CircuitGenome") -> Dict[str, int]:
    """Assign an integer index to every node (0 = ground = ac_n)."""
    nodes: set[str] = set()
    for c in genome.components:
        nodes.add(c.node1)
        nodes.add(c.node2)
    nodes.discard("ac_n")  # ground node, not in matrix
    node_list = sorted(nodes)
    return {n: i for i, n in enumerate(node_list)}


@njit(nogil=True, cache=True)
def _mna_transient(G_base, cap_nodes, cap_gc, diode_nodes, src_idx, out_p, out_n,
                   v_peak, freq_hz, dt, n_steps, steps_per_period,
                   diode_is, diode_nVt, nr_tol, nr_max):
    """Backward-Euler MNA transient with Newton-Raphson (Numba-accelerated)."""
    M = G_base.shape[0]
    v = np.zeros(M, dtype=np.float64)
    v_out = np.empty(n_steps, dtype=np.float64)
    n_caps = cap_nodes.shape[0]
    n_diodes = diode_nodes.shape[0]
    two_pi_f = 2.0 * math.pi * freq_hz

    for step in range(n_steps):
        Vs = v_peak * math.sin(two_pi_f * step * dt)

        for _nr in range(nr_max):
            G_nr = G_base.copy()
            rhs = np.zeros(M, dtype=np.float64)
            rhs[src_idx] = Vs

            # Capacitor companion currents (backward Euler)
            for ci in range(n_caps):
                n1 = cap_nodes[ci, 0]
                n2 = cap_nodes[ci, 1]
                gc = cap_gc[ci]
                v1 = v[n1] if n1 >= 0 else 0.0
                v2 = v[n2] if n2 >= 0 else 0.0
                Icap = gc * (v1 - v2)
                if n1 >= 0:
                    rhs[n1] += Icap
                if n2 >= 0:
                    rhs[n2] -= Icap

            # Diode stamps (Shockley model, linearised companion)
            for di in range(n_diodes):
                anode = diode_nodes[di, 0]
                cathode = diode_nodes[di, 1]
                va = v[anode] if anode >= 0 else 0.0
                vc = v[cathode] if cathode >= 0 else 0.0
                vd = va - vc
                x = vd / diode_nVt
                if x > 80.0:
                    x = 80.0
                elif x < -80.0:
                    x = -80.0
                exp_val = math.exp(x)
                Id = diode_is * (exp_val - 1.0)
                gd = (diode_is / diode_nVt) * exp_val
                Ieq = Id - gd * vd
                if anode >= 0:
                    G_nr[anode, anode] += gd
                    rhs[anode] -= Ieq
                if cathode >= 0:
                    G_nr[cathode, cathode] += gd
                    rhs[cathode] += Ieq
                if anode >= 0 and cathode >= 0:
                    G_nr[anode, cathode] -= gd
                    G_nr[cathode, anode] -= gd

            # Small diagonal regularisation to prevent singular matrices
            for i in range(M):
                G_nr[i, i] += 1e-12

            v_new = np.linalg.solve(G_nr, rhs)

            max_diff = 0.0
            for i in range(M):
                d = abs(v_new[i] - v[i])
                if d > max_diff:
                    max_diff = d

            v = v_new
            if max_diff < nr_tol:
                break

        vp = v[out_p] if out_p >= 0 else 0.0
        vn = v[out_n] if out_n >= 0 else 0.0
        v_out[step] = vp - vn

    return v_out


def _simulate_numpy(
    genome: "CircuitGenome",
    load_resistance: float,
    periods: int = SIM_PERIODS,
    steps_per_period: int = SIM_STEPS_PER_PERIOD,
) -> Tuple[float, Dict[str, float]]:
    """Prepare arrays from genome and dispatch to the Numba kernel."""
    node_map = _build_node_map(genome)
    N = len(node_map)
    if N == 0:
        raise ValueError("No non-ground nodes")
    if "ac_p" not in node_map:
        raise ValueError("ac_p node missing")
    src_node = node_map["ac_p"]
    M = N + 1
    src_idx = N

    def _idx(name: str) -> int:
        return node_map.get(name, -1)

    # Classify components
    res_list: List[Tuple[int, int, float]] = []
    cap_list: List[Tuple[int, int, float]] = []
    diode_list: List[Tuple[int, int]] = []
    for comp in genome.components:
        n1, n2 = _idx(comp.node1), _idx(comp.node2)
        if comp.type == "resistor" and comp.value and comp.value > 0:
            res_list.append((n1, n2, 1.0 / comp.value))
        elif comp.type == "capacitor" and comp.value and comp.value > 0:
            cap_list.append((n1, n2, comp.value))
        elif comp.type == "diode":
            diode_list.append((n1, n2))
    out_p, out_n = _idx("out_p"), _idx("out_n")
    res_list.append((out_p, out_n, 1.0 / load_resistance))

    period = 1.0 / FREQ_HZ
    dt = period / steps_per_period
    n_steps = periods * steps_per_period

    # Build static G matrix (resistors + capacitor companion conductances)
    G_base = np.zeros((M, M), dtype=np.float64)
    for n1, n2, g in res_list:
        if n1 >= 0: G_base[n1, n1] += g
        if n2 >= 0: G_base[n2, n2] += g
        if n1 >= 0 and n2 >= 0:
            G_base[n1, n2] -= g; G_base[n2, n1] -= g

    cap_nodes = np.empty((len(cap_list), 2), dtype=np.int64)
    cap_gc = np.empty(len(cap_list), dtype=np.float64)
    for i, (n1, n2, cap) in enumerate(cap_list):
        gc = cap / dt
        cap_nodes[i] = (n1, n2)
        cap_gc[i] = gc
        if n1 >= 0: G_base[n1, n1] += gc
        if n2 >= 0: G_base[n2, n2] += gc
        if n1 >= 0 and n2 >= 0:
            G_base[n1, n2] -= gc; G_base[n2, n1] -= gc

    # Voltage source stamps
    G_base[src_node, src_idx] += 1.0
    G_base[src_idx, src_node] += 1.0

    # Diode array
    diode_arr = np.array(diode_list, dtype=np.int64).reshape(-1, 2) if diode_list else np.empty((0, 2), dtype=np.int64)

    # Dispatch to Numba kernel
    v_out = _mna_transient(
        G_base, cap_nodes, cap_gc, diode_arr,
        src_idx, out_p, out_n,
        V_PEAK, FREQ_HZ, dt, n_steps, steps_per_period,
        DIODE_IS, DIODE_N * DIODE_VT, NR_TOL, NR_MAX,
    )

    # Metrics from last period
    last = v_out[-steps_per_period:]
    mean_out = float(np.mean(last))
    ripple = float(np.max(last) - np.min(last))
    ripple_factor = ripple / max(abs(mean_out), 1e-6)

    # Shape error: MSE against ideal rectified waveform |Vin(t)|
    t_last = np.arange(n_steps - steps_per_period, n_steps) * dt
    v_ideal = V_PEAK * np.abs(np.sin(2.0 * np.pi * FREQ_HZ * t_last))
    shape_error = float(np.mean((last - v_ideal) ** 2))

    metrics = {
        "mean_out": mean_out,
        "ripple": ripple,
        "ripple_factor": ripple_factor,
        "shape_error": shape_error,
        "diodes": len(diode_list),
        "resistors": len(res_list) - 1,
        "capacitors": len(cap_list),
    }
    return mean_out, metrics


@dataclass
class CircuitComponent:
    type: str; node1: str; node2: str; value: float | None = None
    def to_json(self) -> Dict[str, object]:
        data = asdict(self)
        if self.type == "diode": data.pop("value", None)
        return data

@dataclass
class CircuitGenome:
    components: List[CircuitComponent]
    def json_dict(self) -> Dict[str, object]:
        nodes = sorted({n for c in self.components for n in (c.node1, c.node2)} | set(BASE_NODES))
        return {"nodes": nodes, "components": [c.to_json() for c in self.components],
                "meta": {"source": "evolve_rectifier", "v_rms": V_RMS, "freq_hz": FREQ_HZ}}


def random_node() -> str: return random.choice(BASE_NODES + INTERNAL_NODES)

def random_component() -> CircuitComponent:
    ctype = random.choice(COMP_TYPES)
    n1, n2 = random_node(), random_node()
    while n2 == n1: n2 = random_node()
    value = (float(10 ** random.uniform(2.5, 4.5)) if ctype == "resistor" else
             float(10 ** random.uniform(-7, -3)) if ctype == "capacitor" else None)
    return CircuitComponent(ctype, n1, n2, value)


def _genome_key(genome: CircuitGenome) -> Tuple[Tuple[str, str, str, float | None], ...]:
    """Hashable representation for memoizing simulations."""
    return tuple((c.type, c.node1, c.node2, None if c.value is None else float(round(c.value, 9)))
                 for c in genome.components)


def _component_features(comp: CircuitComponent, idx: int, total: int) -> List[float]:
    type_oh = [1.0 if comp.type == t else 0.0 for t in COMP_TYPES]
    val_log = float(np.clip(math.log10(comp.value), -9, 6)) if comp.value and comp.value > 0 else 0.0
    n1_base, n2_base = (1.0 if comp.node1 in BASE_NODES else 0.0, 1.0 if comp.node2 in BASE_NODES else 0.0)
    same = 1.0 if comp.node1 == comp.node2 else 0.0
    internal = 1.0 if comp.node1 in INTERNAL_NODES or comp.node2 in INTERNAL_NODES else 0.0
    return type_oh + [val_log, n1_base, n2_base, same, internal, idx / max(total - 1, 1), float(total)]


def make_initial_genome(min_parts: int, max_parts: int) -> CircuitGenome:
    count = random.randint(min_parts, max_parts)
    comps = [random_component() for _ in range(count)]
    comps.append(CircuitComponent("resistor", "out_p", "out_n", value=1000.0))
    while sum(1 for c in comps if c.type == "diode") < 4:
        comps.append(CircuitComponent("diode", "ac_p", "out_p"))
    return CircuitGenome(comps)


def mutate_component(comp: CircuitComponent, node_mutation_rate: float, value_jitter: float) -> CircuitComponent:
    c = CircuitComponent(comp.type, comp.node1, comp.node2, comp.value)
    if random.random() < node_mutation_rate: c.node1 = random_node()
    if random.random() < node_mutation_rate: c.node2 = random_node()
    if c.node1 == c.node2: c.node2 = random_node()
    if c.value is not None:
        c.value *= float(np.exp(np.random.normal(0.0, value_jitter)))
        c.value = float(np.clip(c.value, 100.0 if c.type == "resistor" else 1e-8,
                                100000.0 if c.type == "resistor" else 1e-2))
    return c


def _selector_choice(genome: CircuitGenome, selector: SubtreeSelector, temperature: float = 1.0) -> int:
    feats = [_component_features(c, i, len(genome.components)) for i, c in enumerate(genome.components)]
    device = next(selector.parameters()).device
    feats_tensor = torch.tensor(feats, dtype=torch.float32, device=device)
    logits = selector(feats_tensor) / max(temperature, 1e-8)
    probs = F.softmax(logits, dim=0)
    return int(torch.multinomial(probs, 1).item()) if probs.numel() > 1 else 0


def crossover(parent_a: CircuitGenome, parent_b: CircuitGenome, selector: SubtreeSelector | None = None,
              temperature: float = 1.0, crossover_rate: float = 1.0) -> Tuple[CircuitGenome, CircuitGenome]:
    if not parent_a.components or not parent_b.components or random.random() >= crossover_rate:
        return parent_a, parent_b
    if selector is None:
        cut_a, cut_b = random.randint(1, len(parent_a.components)), random.randint(1, len(parent_b.components))
    else:
        cut_a, cut_b = _selector_choice(parent_a, selector, temperature) or 1, _selector_choice(parent_b, selector, temperature) or 1
    return (CircuitGenome(parent_a.components[:cut_a] + parent_b.components[cut_b:]),
            CircuitGenome(parent_b.components[:cut_b] + parent_a.components[cut_a:]))


def mutate(genome: CircuitGenome, add_rate: float, remove_rate: float, node_mutation_rate: float, value_jitter: float,
           min_parts: int, max_parts: int) -> CircuitGenome:
    comps: List[CircuitComponent] = []
    for comp in genome.components:
        if random.random() < remove_rate and len(comps) > min_parts:
            continue
        comps.append(mutate_component(comp, node_mutation_rate, value_jitter))
    if random.random() < add_rate and len(comps) < max_parts:
        comps.append(random_component())
    if not any(c.type == "resistor" and {c.node1, c.node2} == {"out_p", "out_n"} for c in comps):
        comps.append(CircuitComponent("resistor", "out_p", "out_n", value=1000.0))
    return CircuitGenome(comps)


_sim_cache: Dict[Tuple[Tuple[str, str, str, float | None], ...], Tuple[float, Dict[str, float]]] = {}


def simulate(genome: CircuitGenome, load_resistance: float,
             periods: int = SIM_PERIODS) -> Tuple[float, Dict[str, float]]:
    """Cached entry-point for the fast NumPy simulator."""
    cache_key = _genome_key(genome)
    cached = _sim_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _simulate_numpy(genome, load_resistance, periods)
    _sim_cache[cache_key] = result
    return result


def optimize_component_values(genome: CircuitGenome, target_dc: float, dc_weight: float,
                             load_resistance: float, method: str = 'L-BFGS-B', 
                             maxiter: int = 50) -> CircuitGenome:
    """Optimize resistor and capacitor values using L-BFGS-B while keeping topology fixed."""
    # Collect components with optimizable values
    value_components = [(i, comp) for i, comp in enumerate(genome.components) 
                       if comp.value is not None and comp.type in ('resistor', 'capacitor')]
    
    if not value_components:
        return genome  # No values to optimize
    
    # Extract initial values (in log space for better optimization)
    initial_log_values = np.array([math.log10(comp.value) for _, comp in value_components])
    
    # Define bounds in log space
    bounds = []
    for _, comp in value_components:
        if comp.type == 'resistor':
            bounds.append((2.0, 5.0))  # 100Ω to 100kΩ
        else:  # capacitor
            bounds.append((-8.0, -2.0))  # 10nF to 10mF
    
    # Define objective function
    def objective(log_values):
        # Create temporary genome with updated values
        temp_genome = CircuitGenome([
            CircuitComponent(c.type, c.node1, c.node2, c.value)
            for c in genome.components
        ])
        
        # Update values from log space
        for (idx, _), log_val in zip(value_components, log_values):
            temp_genome.components[idx].value = float(10 ** log_val)
        
        # Evaluate fitness (same hierarchical objective as GA)
        try:
            mean_out, metrics = simulate(temp_genome, load_resistance)
            j_shape = metrics["shape_error"]
            j_dc = (target_dc - mean_out) ** 2
            return j_shape + dc_weight * j_dc
        except Exception:
            return 1e10  # Large penalty for failed simulations
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_log_values,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-6}
        )
        
        # Update genome with optimized values if successful
        if result.success and result.fun < objective(initial_log_values):
            for (idx, _), log_val in zip(value_components, result.x):
                genome.components[idx].value = float(10 ** log_val)
    except Exception:
        pass  # If optimization fails, keep original values
    
    return genome


def fitness(genome: CircuitGenome, target_dc: float, dc_weight: float,
            load_resistance: float, size_penalty: float = SIZE_PENALTY) -> Tuple[float, Dict[str, float]]:
    """Hierarchical fitness: J = J_shape + λ·J_dc + μ·size.

    J_shape  = MSE(V_out(t), |V_in(t)|)  — dominates early, kills capacitor traps
    J_dc     = (⟨V_out⟩ − V_target)²     — fine-tunes DC level once shape is correct
    """
    BAD = {"mean_out": 0.0, "ripple": 1e9, "ripple_factor": 1e9, "shape_error": 1e9}
    try:
        mean_out, metrics = simulate(genome, load_resistance)
    except Exception:
        return 1e9, BAD

    # Clamp to avoid overflow on divergent simulations
    if not np.isfinite(mean_out) or abs(mean_out) > 1e6:
        return 1e9, BAD

    j_shape = min(metrics["shape_error"], 1e9)
    j_dc = (target_dc - mean_out) ** 2
    size_cost = size_penalty * len(genome.components)
    score = j_shape + dc_weight * j_dc + size_cost
    return score, metrics


def tournament(pop: Sequence[CircuitGenome], scores: List[float], k: int = 3) -> CircuitGenome:
    """Tournament selection based on scalar fitness score (lower is better)."""
    k = max(1, min(k, len(pop)))
    idxs = random.sample(range(len(pop)), k)
    best_idx = min(idxs, key=lambda i: float(scores[i]))
    return pop[best_idx]


def evolve(pop_size: int, gens: int, min_parts: int, max_parts: int, crossover_rate: float, add_rate: float,
           remove_rate: float, node_mutation_rate: float, value_jitter: float, tournament_size: int,
           elitism_size: int, dc_weight: float, target_dc: float, load_resistance: float,
           seed: int | None = None, use_selector: bool = False, selector_hidden: int = SELECTOR_HIDDEN,
            selector_activation: str = SELECTOR_ACTIVATION,
           selector_dropout: float = SELECTOR_DROPOUT, selector_layers: int = SELECTOR_LAYERS,
           selector_temperature: float = SELECTOR_TEMPERATURE, opt_values: bool = False,
           opt_maxiter: int = 50, parallel_workers: int = 0) -> Tuple[CircuitGenome, Dict[str, float]]:
    if seed is not None: random.seed(seed); np.random.seed(seed)
    population = [make_initial_genome(min_parts, max_parts) for _ in range(pop_size)]
    selector = None
    if use_selector:
        selector = SubtreeSelector(
            hidden_dim=selector_hidden,
            mlp_activation=selector_activation,
            mlp_dropout=selector_dropout,
            mlp_layers=selector_layers,
        )

    best_genome = None
    best_score = float("inf")
    best_metrics: Dict[str, float] = {}

    writer = tqdm.write if hasattr(tqdm, "write") else print

    pool = cf.ThreadPoolExecutor(max_workers=parallel_workers) if parallel_workers > 1 else None
    def _eval_fitness(ind):
        return fitness(ind, target_dc, dc_weight, load_resistance)

    for gen_idx in tqdm(range(gens), desc="GA generations", unit="gen"):
        if pool is not None:
            results = list(pool.map(_eval_fitness, population))
            scores = [r[0] for r in results]
            metrics_list = [r[1] for r in results]
        else:
            scores = []
            metrics_list = []
            for ind in population:
                score, m = _eval_fitness(ind)
                scores.append(score)
                metrics_list.append(m)

        # Update best by scalar score
        gen_best_idx = int(np.argmin(scores)) if scores else 0
        if scores and scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_genome = population[gen_best_idx]
            best_metrics = metrics_list[gen_best_idx]

        # Elitism: preserve top solutions by score
        next_pop: List[CircuitGenome] = []
        if scores and elitism_size > 0:
            elite_indices = sorted(range(len(population)), key=lambda i: scores[i])[:elitism_size]
            next_pop.extend([population[i] for i in elite_indices])

        # Report best by score
        if scores:
            gen_best_metrics = metrics_list[gen_best_idx]
            writer(
                f"Gen {gen_idx}: score={scores[gen_best_idx]:.6g}, "
                f"shape_err={gen_best_metrics.get('shape_error', 0):.1f}, "
                f"mean_out={gen_best_metrics['mean_out']:.3f}, "
                f"ripple_factor={gen_best_metrics['ripple_factor']:.4f}, "
                f"parts={len(population[gen_best_idx].components)}"
            )

        # Generate offspring
        while len(next_pop) < pop_size:
            parent_a = tournament(population, scores, k=tournament_size)
            parent_b = tournament(population, scores, k=tournament_size)
            child1, child2 = crossover(parent_a, parent_b, selector, selector_temperature, crossover_rate)
            child1 = mutate(child1, add_rate, remove_rate, node_mutation_rate, value_jitter, min_parts, max_parts)
            child2 = mutate(child2, add_rate, remove_rate, node_mutation_rate, value_jitter, min_parts, max_parts)
            next_pop.extend([child1, child2])

        population = next_pop[:pop_size]

    if pool is not None:
        pool.shutdown(wait=False)

    # Apply L-BFGS-B optimization to best solution if requested
    final_genome = best_genome or population[0]
    if opt_values:
        writer("Optimizing component values with L-BFGS-B...")
        final_genome = optimize_component_values(
            final_genome, target_dc, dc_weight, load_resistance, 
            maxiter=opt_maxiter
        )
        # Re-evaluate metrics after optimization
        try:
            _, best_metrics = fitness(final_genome, target_dc, dc_weight, load_resistance)
            writer(f"After L-BFGS-B: mean_out={best_metrics['mean_out']:.3f}V, "
                  f"ripple_factor={best_metrics['ripple_factor']:.4f}")
        except Exception:
            pass
    
    return final_genome, best_metrics


def save_genome(genome: CircuitGenome, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(genome.json_dict(), f, indent=2)


def cli():
    parser = argparse.ArgumentParser(description="Evolve a bridge rectifier topology with a GA")
    parser.add_argument("--pop", type=int, default=POP_SIZE, help="Population size")
    parser.add_argument("--gens", type=int, default=GENS, help="Number of generations")
    parser.add_argument("--out", type=str, default="evolved_circuit.json", help="Output JSON path")
    parser.add_argument("--min-parts", type=int, default=MIN_PARTS, help="Minimum parts per genome")
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS, help="Maximum parts per genome")
    parser.add_argument("--crossover-rate", type=float, default=CROSSOVER_RATE, help="Probability of crossover")
    parser.add_argument("--add-rate", type=float, default=ADD_RATE, help="Add-part mutation rate")
    parser.add_argument("--remove-rate", type=float, default=REMOVE_RATE, help="Remove-part mutation rate")
    parser.add_argument("--node-mutation", type=float, default=NODE_MUTATION_RATE, help="Node mutation rate per endpoint")
    parser.add_argument("--value-jitter", type=float, default=VALUE_JITTER, help="Log-normal jitter for values")
    parser.add_argument("--tournament-size", type=int, default=TOURNAMENT_SIZE, help="Tournament selection size")
    parser.add_argument("--elitism", type=int, default=ELITISM_SIZE, help="Number of elite solutions to preserve")
    parser.add_argument("--dc-weight", type=float, default=DC_WEIGHT, help="Weight λ for DC-error term in hierarchical fitness")
    parser.add_argument("--target-dc", type=float, default=TARGET_DC, help="Desired DC voltage after rectification")
    parser.add_argument("--load", type=float, default=LOAD_RESISTANCE, help="Load resistance in ohms")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--use-selector", action="store_true", help="Use SubtreeSelector to guide crossover")
    parser.add_argument("--selector-hidden", type=int, default=SELECTOR_HIDDEN, help="Hidden size for SubtreeSelector")
    parser.add_argument("--selector-activation", type=str, default=SELECTOR_ACTIVATION, help="Activation function (relu/gelu/tanh)")
    parser.add_argument("--selector-dropout", type=float, default=SELECTOR_DROPOUT, help="Dropout rate for selector")
    parser.add_argument("--selector-layers", type=int, default=SELECTOR_LAYERS, help="Number of layers in selector")
    parser.add_argument("--selector-temperature", type=float, default=SELECTOR_TEMPERATURE, help="Temperature for selector sampling")
    parser.add_argument("--opt-values", action="store_false", help="Optimize component values with L-BFGS-B after GA", default=True)
    parser.add_argument("--opt-maxiter", type=int, default=75, help="Max iterations for L-BFGS-B optimization")
    parser.add_argument("--parallel", action="store_true", help="Evaluate fitness in parallel using threads")
    parser.add_argument("--workers", type=int, default=0, help="Worker thread count (0=auto)")
    args = parser.parse_args()

    workers = args.workers
    if args.parallel and workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if not args.parallel:
        workers = 0

    best, metrics = evolve(
        pop_size=args.pop, gens=args.gens, min_parts=args.min_parts, max_parts=args.max_parts,
        crossover_rate=args.crossover_rate, add_rate=args.add_rate, remove_rate=args.remove_rate,
        node_mutation_rate=args.node_mutation, value_jitter=args.value_jitter, tournament_size=args.tournament_size,
        elitism_size=args.elitism, dc_weight=args.dc_weight, target_dc=args.target_dc,
        load_resistance=args.load, seed=args.seed, use_selector=args.use_selector,
        selector_hidden=args.selector_hidden, selector_activation=args.selector_activation, selector_dropout=args.selector_dropout,
        selector_layers=args.selector_layers, selector_temperature=args.selector_temperature,
        opt_values=args.opt_values, opt_maxiter=args.opt_maxiter,
        parallel_workers=workers
    )

    save_genome(best, args.out)
    print(f"Best circuit saved to {args.out}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    cli()
