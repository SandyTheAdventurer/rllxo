"""Pair-level semantic information for crossover: every child of two parents, scored cheaply.

A parent's tape holds each subtree's output on a fixed batch of training points. Replacing node i of A
by subtree j of B only changes the outputs on the path from i to the root, so walking up the tree once,
carrying a stack of the candidate outputs of every replaced descendant, gives the exact loss of every
possible child on the batch with one array op per tree node.
"""
import random
import numpy as np
from scipy.stats import rankdata
import main
from main import Unary, Binary


class Tape:
    __slots__ = ("nodes", "parent", "side", "left", "right", "S", "G", "out", "fp", "loss", "finite")


def build_tape(root, inputs, y):
    t = Tape(); t.nodes = []; parent = []; side = []; left = []; right = []; vals = []

    def visit(n, p, s):
        i = len(t.nodes)
        t.nodes.append(n); parent.append(p); side.append(s); left.append(-1); right.append(-1); vals.append(None)
        if isinstance(n, Unary):
            left[i] = len(t.nodes)
            v = main.UNARY_FUNCS[n.func_name](visit(n.child, i, 0))
        elif isinstance(n, Binary):
            left[i] = len(t.nodes)
            l = visit(n.left, i, 0)
            right[i] = len(t.nodes)
            v = main.BINARY_FUNCS[n.func_name](l, visit(n.right, i, 1))
        else:
            v = n.evaluate(inputs)
        vals[i] = v
        return v

    with np.errstate(all="ignore"):
        visit(root, -1, 0)
    ref = next(iter(inputs.values()))
    t.S = np.ascontiguousarray(np.stack([np.broadcast_to(np.asarray(v, float), ref.shape) for v in vals]))
    t.out = t.S[0]; t.G = None
    t.parent, t.side, t.left, t.right = parent, side, left, right
    R = np.round(np.nan_to_num(t.S[:, :24], nan=1e300, posinf=1e301, neginf=-1e301), 9) + 0.0
    t.fp = np.array([hash(r.tobytes()) for r in R])
    t.finite = np.isfinite(t.S).all(1)
    with np.errstate(all="ignore"):
        t.loss = float(np.mean((t.out - y) ** 2))
    return t


def _stack(t, i, SB):
    """(size of i's subtree, Lb, n): the output at node i with each descendant (pre-order) replaced by each row of SB."""
    n = t.nodes[i]
    if isinstance(n, Unary):
        return np.concatenate([SB[None], main.UNARY_FUNCS[n.func_name](_stack(t, t.left[i], SB))])
    if isinstance(n, Binary):
        f = main.BINARY_FUNCS[n.func_name]
        l, r = t.left[i], t.right[i]
        return np.concatenate([SB[None], f(_stack(t, l, SB), t.S[r]), f(t.S[l], _stack(t, r, SB))])
    return SB[None]


def child_losses(tA, tB, y):
    """(La, Lb) exact batch MSE of A with node i replaced by B's subtree j (inf if invalid)."""
    with np.errstate(all="ignore"):
        d = _stack(tA, 0, tB.S) - y
        out = np.einsum("ijt,ijt->ij", d, d) / len(y)
    out[~np.isfinite(out)] = np.inf
    return out


def jacobian(t):
    """G[i] = d(root output)/d(output of node i), per data point (pre-order, parents first)."""
    if t.G is not None:
        return t.G
    L, n = t.S.shape
    G = np.zeros((L, n)); G[0] = 1.0
    with np.errstate(all="ignore"):
        for i, nd in enumerate(t.nodes):
            g = G[i]
            if isinstance(nd, Unary):
                c = t.left[i]; v = t.S[c]; f = nd.func_name
                if f == "sin": G[c] = g * np.cos(v)
                elif f == "cos": G[c] = -g * np.sin(v)
                elif f == "exp": G[c] = g * t.S[i]
                elif f == "log": G[c] = g / v
                elif f == "sqrt": G[c] = 0.5 * g / t.S[i]
                else: G[c] = g
            elif isinstance(nd, Binary):
                l, r = t.left[i], t.right[i]; f = nd.func_name
                if f == "add": G[l] = g; G[r] = g
                elif f == "sub": G[l] = g; G[r] = -g
                elif f == "mul": G[l] = g * t.S[r]; G[r] = g * t.S[l]
                elif f == "div":
                    b = t.S[r] + 1e-6
                    G[l] = g / b; G[r] = -g * t.S[l] / (b * b)
    G[~np.isfinite(G)] = 0.0
    t.G = G
    return G


def first_order_losses(tA, tB, y):
    """(La, Lb) MSE of the same children predicted from the tree Jacobian: two matmuls, no child built."""
    G = jacobian(tA); r = np.nan_to_num(y - tA.out, nan=0.0, posinf=0.0, neginf=0.0)
    SA = np.nan_to_num(tA.S, nan=0.0, posinf=0.0, neginf=0.0); SB = np.nan_to_num(tB.S, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y); D = G * G
    with np.errstate(all="ignore"):
        Q = D @ (SB * SB).T - 2 * (D * SA) @ SB.T - 2 * (G * r) @ SB.T
        c = (D * SA * SA).sum(1) + 2 * (G * r * SA).sum(1) + (r * r).sum()
        out = np.maximum((Q + c[:, None]) / n, 0.0)
    out[~np.isfinite(out)] = np.inf
    out[:, ~tB.finite] = np.inf
    return out


DEFAULTS = dict(mode="exact", temp=1e9, points=None, noop=True, mix=1.0, directed=False, tabu=False, bias=False, cap=2500)
CFG = dict(DEFAULTS)
_cache, _data, _grids = {}, {}, {}
_bytes = {"tape": 0, "grid": 0}
TAPE_BYTES, GRID_BYTES = 150e6, 100e6  # caches are memory-bounded: bloated trees make single entries large
_seen = {"gen": None, "fp": set()}


def configure(**kw):
    CFG.clear(); CFG.update(DEFAULTS); CFG.update({k[4:]: v for k, v in kw.items()})
    _cache.clear(); _data.clear(); _grids.clear(); _seen.update(gen=None, fp=set()); _bytes.update(tape=0, grid=0)
    if main._libc is not None:
        main._libc.malloc_trim(0)


def batch():
    if not _data:
        inp, y = main.CROSSOVER_INPUTS, main.CROSSOVER_Y
        m = CFG["points"]
        if m and m < len(y):
            idx = np.random.default_rng(len(y) * 7 + 1).choice(len(y), m, replace=False)
            inp = {k: np.ascontiguousarray(v[idx]) for k, v in inp.items()}; y = np.ascontiguousarray(y[idx])
        _data.update(inp=inp, y=y)
    return _data["inp"], _data["y"]


def tape(root, key=None):
    k = key if key is not None else root.key()
    t = _cache.get(k)
    if t is None:
        if len(_cache) > 2500 or _bytes["tape"] > TAPE_BYTES:
            _cache.clear(); _bytes["tape"] = 0
        inp, y = batch()
        t = _cache[k] = build_tape(root, inp, y)
        _bytes["tape"] += 2 * t.S.nbytes  # S and, once computed, G
    return t


def directed(a, b, ka, kb):
    """Losses of a's children taking subtrees of b, memoized for repeat pairs (selection reuses parents)."""
    g = _grids.get((ka, kb))
    if g is None:
        if len(_grids) > 1500 or _bytes["grid"] > GRID_BYTES:
            _grids.clear(); _bytes["grid"] = 0
        tA, tB = tape(a, ka), tape(b, kb)
        _, y = batch()
        g = child_losses(tA, tB, y) if CFG["mode"] == "exact" else first_order_losses(tA, tB, y)
        _grids[(ka, kb)] = g
        _bytes["grid"] += g.nbytes
    return g


def noop_mask(tA, tB, c1, c2):
    same = tA.fp[:, None] == tB.fp[None, :]
    same[0, 0] = True  # swapping the two roots only exchanges the parents
    la, lb = tA.loss, tB.loss
    if c1 is not None and np.isfinite(la) and np.isfinite(lb):
        with np.errstate(all="ignore"):
            ta, tb = 1e-9 * max(la, 1e-30), 1e-9 * max(lb, 1e-30)
            same |= (np.abs(c1 - la) <= ta) & (np.abs(c2 - lb) <= tb)
            same |= (np.abs(c1 - lb) <= tb) & (np.abs(c2 - la) <= ta)
    return same


def _sample(v, idx):
    """A flat index from idx, softmax over the rank of v (0 = best) at CFG temperature."""
    if CFG["temp"] >= 1e8:
        return idx[random.randrange(len(idx))]
    r = (rankdata(v) - 1) / max(len(v) - 1, 1)
    p = np.exp(-(r - r.min()) / CFG["temp"])
    c = np.cumsum(p)
    return idx[min(int(np.searchsorted(c, random.random() * c[-1], side="right")), len(c) - 1)]


def _fresh(c, gen):
    """False where a child's batch loss matches one already produced this generation."""
    if _seen["gen"] != gen:
        _seen.update(gen=gen, fp=set())
    if not _seen["fp"]:
        return np.ones(c.shape, bool)
    with np.errstate(all="ignore"):
        q = np.round(np.log(np.maximum(c, 1e-300)), 9)
    return ~np.isin(q, np.fromiter(_seen["fp"], float))


def _mark(c):
    if np.isfinite(c):
        _seen["fp"].add(round(float(np.log(max(c, 1e-300))), 9))


def grids(a, b):
    ka, kb = a.key(), b.key()
    tA, tB = tape(a, ka), tape(b, kb)
    if CFG["mode"] == "none":
        return tA, tB, None, None
    return tA, tB, directed(a, b, ka, kb), directed(b, a, kb, ka).T


def pick(a, b, gen):
    """Crossover points (i1, j1, i2, j2): child 1 = a with i1 <- b's j1, child 2 = b with j2 <- a's i2."""
    la, lb = a.size(), b.size()
    # Exact grids cost La*Lb*depth, so bloated pairs fall back to uniform points
    if (CFG["mix"] < 1.0 and random.random() >= CFG["mix"]) or (CFG["mode"] == "exact" and la * lb > CFG["cap"]):
        i, j = random.randrange(la), random.randrange(lb)
        return i, j, i, j
    tA, tB, c1, c2 = grids(a, b)
    bad = noop_mask(tA, tB, c1, c2) if CFG["noop"] else np.zeros((la, lb), bool)
    if c1 is None:
        idx = np.flatnonzero(~bad)
        if len(idx) == 0:
            i, j = random.randrange(la), random.randrange(lb)
        else:
            i, j = divmod(int(idx[random.randrange(len(idx))]), lb)
        return i, j, i, j
    if CFG["directed"]:
        res = []
        for c in (c1, c2):
            ok = ~bad & np.isfinite(c)
            if CFG["tabu"]:
                ok &= _fresh(c, gen)
            idx = np.flatnonzero(ok)
            if len(idx) == 0:
                res.append((random.randrange(la), random.randrange(lb)))
                continue
            i, j = divmod(int(_sample(c.ravel()[idx], idx)), lb)
            if CFG["tabu"]:
                _mark(c[i, j])
            res.append((i, j))
        return res[0][0], res[0][1], res[1][0], res[1][1]
    best = np.minimum(c1, c2)
    ok = ~bad & np.isfinite(best)
    if CFG["tabu"]:
        ok &= _fresh(best, gen)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        i, j = random.randrange(la), random.randrange(lb)
        return i, j, i, j
    i, j = divmod(int(_sample(best.ravel()[idx], idx)), lb)
    if CFG["tabu"]:
        _mark(best[i, j])
    return i, j, i, j


def bias_grid(a, b):
    """Additive log-preference over pairs for the policy softmax: -inf on no-ops, -rank/temp otherwise."""
    la, lb = a.size(), b.size()
    if CFG["mode"] == "exact" and la * lb > CFG["cap"]:
        return np.zeros((la, lb))
    tA, tB, c1, c2 = grids(a, b)
    bad = noop_mask(tA, tB, c1, c2) if CFG["noop"] else np.zeros((la, lb), bool)
    out = np.full((la, lb), -np.inf)
    if c1 is None:
        out[~bad] = 0.0
    else:
        best = np.minimum(c1, c2)
        ok = ~bad & np.isfinite(best)
        if ok.any():
            v = best[ok]
            out[ok] = -((rankdata(v) - 1) / max(len(v) - 1, 1)) / CFG["temp"]
    if not np.isfinite(out).any():
        out[:] = 0.0
    return out
