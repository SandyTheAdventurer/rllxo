# Genetic algorithm for symbolic regression (simple, self-contained)

import random
import collections
import ctypes
import operator
import os
from typing import Dict, List, Mapping, Tuple
import warnings
import multiprocessing as mp
import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm  # Changed from trange to tqdm
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_SEED = 0

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_global_seed(DEFAULT_SEED)

# Set device for PyTorch operations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_selector_device(selector_device: str | None = "auto") -> torch.device:
    """'cpu' or 'cuda' (env RLLXO_SELECTOR_DEVICE overrides); anything else means CUDA if available."""
    requested = str(os.getenv("RLLXO_SELECTOR_DEVICE", selector_device or "auto")).strip().lower()
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


# Protected operators to avoid NaNs/Infs
def pdiv(a, b):
    return a / (b + 1e-6)


def ppow(a, b):
    # Avoid hard crashes on invalid domains; NaNs are handled by fitness.
    with np.errstate(all="ignore"):
        return np.power(a, b)

def plog(a):
    return np.log(np.abs(a) + 1e-10)

def pexp(a):
    # clip to avoid overflow
    return np.exp(np.clip(a, -50, 50))

DEFAULT_UNARY_FUNCS = {
    'sin': np.sin,
    'cos': np.cos,
    'exp': pexp,
    'log': plog,
}

BASE_BINARY_FUNCS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': pdiv,
}

# Sympy mappings for conversion/simplification
DEFAULT_SYMPY_UNARY = {
    'sin': sp.sin,
    'cos': sp.cos,
    'exp': sp.exp,
    'log': lambda a: sp.log(sp.Abs(a)),  # match plog, which takes log(|x|)
}

def _sympy_safe_div(a, b):
    if b == 0:
        return a / sp.Float(1e-12)
    return a / b

BASE_SYMPY_BINARY = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _sympy_safe_div,
}

BINARY_FUNCS = dict(BASE_BINARY_FUNCS)
SYMPY_BINARY = dict(BASE_SYMPY_BINARY)

UNARY_FUNCS = dict(DEFAULT_UNARY_FUNCS)
SYMPY_UNARY = {name: DEFAULT_SYMPY_UNARY[name] for name in UNARY_FUNCS}
ALL_UNARY = list(UNARY_FUNCS.keys())

ALL_BINARY = list(BINARY_FUNCS.keys())

def get_selector_feature_dim(base_dim: int = 10) -> int:
    return base_dim + len(ALL_UNARY) + len(ALL_BINARY)

VAR_NAMES = ['x']
VAR_SYMBOLS = {name: sp.Symbol(name) for name in VAR_NAMES}


def set_variables(names: List[str]):
    unique = list(dict.fromkeys(names))
    if not unique:
        raise ValueError("var_names must contain at least one name.")
    if len(unique) != len(names):
        raise ValueError("var_names must be unique.")
    global VAR_NAMES, VAR_SYMBOLS
    VAR_NAMES = unique
    VAR_SYMBOLS = {name: sp.Symbol(name) for name in VAR_NAMES}


class Node:
    def evaluate(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def clone(self) -> "Node":
        raise NotImplementedError

    def key(self):
        """Hashable structural identity (operators, variable names, exact constant values)."""
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def all_nodes(self) -> List["Node"]:
        raise NotImplementedError

    def to_sympy(self):
        raise NotImplementedError


class Const(Node):
    def __init__(self, value: float):
        self.value = float(value)

    def clone(self):
        return Const(self.value)

    def key(self):
        return self.value

    def evaluate(self, inputs):
        reference = next(iter(inputs.values()))
        return np.full_like(reference, self.value, dtype=float)

    def size(self):
        return 1

    def depth(self):
        return 0

    def to_string(self):
        return f"{self.value:.4g}"

    def all_nodes(self):
        return [self]

    def to_sympy(self):
        return sp.Float(self.value)


class Var(Node):
    def __init__(self, name: str = None):
        self.name = name or VAR_NAMES[0]

    def clone(self):
        return Var(self.name)

    def key(self):
        return self.name

    def evaluate(self, inputs):
        return inputs[self.name]

    def size(self):
        return 1

    def depth(self):
        return 0

    def to_string(self):
        return self.name

    def all_nodes(self):
        return [self]

    def to_sympy(self):
        return VAR_SYMBOLS[self.name]


class Unary(Node):
    def __init__(self, func_name: str, child: Node):
        self.func_name = func_name
        self.child = child

    def clone(self):
        return Unary(self.func_name, self.child.clone())

    def key(self):
        return (self.func_name, self.child.key())

    def evaluate(self, inputs):
        f = UNARY_FUNCS[self.func_name]
        return f(self.child.evaluate(inputs))

    def size(self):
        return 1 + self.child.size()

    def depth(self):
        return 1 + self.child.depth()

    def to_string(self):
        return f"{self.func_name}({self.child.to_string()})"

    def all_nodes(self):
        return [self] + self.child.all_nodes()

    def to_sympy(self):
        f = SYMPY_UNARY[self.func_name]
        return f(self.child.to_sympy())


class Binary(Node):
    def __init__(self, func_name: str, left: Node, right: Node):
        self.func_name = func_name
        self.left = left
        self.right = right

    def clone(self):
        return Binary(self.func_name, self.left.clone(), self.right.clone())

    def key(self):
        return (self.func_name, self.left.key(), self.right.key())

    def evaluate(self, inputs):
        f = BINARY_FUNCS[self.func_name]
        return f(self.left.evaluate(inputs), self.right.evaluate(inputs))

    def size(self):
        return 1 + self.left.size() + self.right.size()

    def depth(self):
        return 1 + max(self.left.depth(), self.right.depth())

    def to_string(self):
        op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}[self.func_name]
        return f"({self.left.to_string()} {op} {self.right.to_string()})"

    def all_nodes(self):
        return [self] + self.left.all_nodes() + self.right.all_nodes()

    def to_sympy(self):
        f = SYMPY_BINARY[self.func_name]
        return f(self.left.to_sympy(), self.right.to_sympy())


# Helpers to pick random node and replace subtree
def collect_nodes_with_parents(node: Node, parent=None, attr_name=None):
    res = [(node, parent, attr_name)]
    if isinstance(node, Unary):
        res += collect_nodes_with_parents(node.child, node, 'child')
    elif isinstance(node, Binary):
        res += collect_nodes_with_parents(node.left, node, 'left')
        res += collect_nodes_with_parents(node.right, node, 'right')
    return res


def gather_used_var_names(ind: Node) -> set[str]:
    return {node.name for node in ind.all_nodes() if isinstance(node, Var)}


def replace_child(parent: Node, attr_name: str, new_child: Node):
    if parent is None:
        raise ValueError("replace_child called with parent=None")
    if isinstance(parent, Unary) and attr_name == 'child':
        parent.child = new_child
    elif isinstance(parent, Binary) and attr_name in ('left', 'right'):
        if attr_name == 'left':
            parent.left = new_child
        else:
            parent.right = new_child
    else:
        setattr(parent, attr_name, new_child)


def point_mutation(ind: Node, prob_change=0.1) -> Node:
    ind_clone = ind.clone()
    for node, parent, attr in collect_nodes_with_parents(ind_clone):
        if isinstance(node, Const):
            if random.random() < prob_change:
                node.value += random.uniform(-1, 1)
        elif isinstance(node, Unary):
            if random.random() < prob_change:
                node.func_name = random.choice(ALL_UNARY)
        elif isinstance(node, Binary):
            if random.random() < prob_change:
                node.func_name = random.choice(ALL_BINARY)
        elif isinstance(node, Var) and len(VAR_NAMES) > 1:
            if random.random() < prob_change:
                node.name = random.choice(VAR_NAMES)
    return ind_clone


# Selection: tournament
def tournament_pick_index(pop_size: int, score_key, k=3) -> int:
    """Index of the best (lowest score_key) of k distinct random individuals."""
    if pop_size <= 0:
        raise ValueError("Population size must be > 0.")
    return min(random.sample(range(pop_size), min(k, pop_size)), key=score_key)


def pareto_objectives(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
                      inclusion_coef=1.0) -> Tuple[float, float]:
    """Return (error_with_inclusion, complexity) objectives for Pareto sorting."""
    return score_individual(ind, inputs, y, 0.0, inclusion_coef)[1]


def pareto_rank_and_crowding(objectives: List[Tuple[float, ...]]):
    """Compute Pareto rank and crowding distance (NSGA-II style)."""
    n = len(objectives)
    if n == 0:
        return [], []
    num_obj = len(objectives[0]) if objectives[0] else 0
    if num_obj == 0:
        return [0] * n, [0.0] * n
    obj = np.asarray(objectives, dtype=float)
    # dominates[i, j]: i is no worse than j everywhere and strictly better somewhere
    dominates = (obj[:, None] <= obj[None]).all(-1) & (obj[:, None] < obj[None]).any(-1)
    domination_sets = [set(np.flatnonzero(row).tolist()) for row in dominates]
    domination_counts = dominates.sum(axis=0).tolist()
    fronts = []

    current_front = [i for i in range(n) if domination_counts[i] == 0]
    rank = [0] * n
    front_rank = 0
    while current_front:
        fronts.append(current_front)
        for idx in current_front:
            rank[idx] = front_rank
            for dominated in domination_sets[idx]:
                domination_counts[dominated] -= 1
        next_front = []
        for idx in current_front:
            for dominated in domination_sets[idx]:
                if domination_counts[dominated] == 0:
                    next_front.append(dominated)
        # Deduplicate next front
        current_front = list(dict.fromkeys(next_front))
        front_rank += 1

    crowding = [0.0] * n
    for front in fronts:
        if len(front) <= 2:
            for idx in front:
                crowding[idx] = float("inf")
            continue
        for obj_idx in range(num_obj):
            sorted_front = sorted(front, key=lambda i: objectives[i][obj_idx])
            min_val = objectives[sorted_front[0]][obj_idx]
            max_val = objectives[sorted_front[-1]][obj_idx]
            crowding[sorted_front[0]] = float("inf")
            crowding[sorted_front[-1]] = float("inf")
            if max_val == min_val:
                continue
            for i in range(1, len(sorted_front) - 1):
                prev_val = objectives[sorted_front[i - 1]][obj_idx]
                next_val = objectives[sorted_front[i + 1]][obj_idx]
                crowding[sorted_front[i]] += (next_val - prev_val) / (max_val - min_val)
    return rank, crowding


# Neural networks that learn to select good subtree crossover points using A2C.
ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'gelu': nn.GELU,
}


def _activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}. Choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class SubtreeSelector(nn.Module):
    """Scores every node independently with an MLP; a pair's logit is the sum of its two node logits."""

    needs_distances = False

    def __init__(self, feature_dim=10, hidden_dim=64, mlp_layers=3, mlp_activation='relu',
                 mlp_dropout=0.0, use_critic=True):
        super().__init__()
        self.use_critic = use_critic
        layers = []
        input_size = feature_dim
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(input_size, hidden_dim), _activation(mlp_activation)]
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            input_size = hidden_dim
        self.shared_net = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_dim, 1)
        if self.use_critic:
            self.critic = nn.Linear(hidden_dim, 1)
        self.to(DEVICE)

    def forward(self, features, return_value=False):
        """Per-node action logits (N,), plus critic values (N,) or None when return_value is set."""
        shared = self.shared_net(features)
        logits = self.actor(shared).squeeze(-1)
        if not return_value:
            return logits
        return logits, (self.critic(shared).squeeze(-1) if self.use_critic else None)

    def score_pairs(self, a, b):
        """Joint logits (B, La, Lb) over node pairs, and critic values (B, La, Lb) or None.
        a and b are PairBatch sides; padded positions are masked out by the caller."""
        a_logits, a_values = self(a.feats, return_value=True)
        b_logits, b_values = self(b.feats, return_value=True)
        logits = a_logits.unsqueeze(2) + b_logits.unsqueeze(1)
        if a_values is None:
            return logits, None
        return logits, (a_values.unsqueeze(2) + b_values.unsqueeze(1)) / 2.0


class PairSelector(nn.Module):
    """Scores node pairs jointly: each parent's nodes attend to the other parent's nodes.

    Nodes are embedded by an MLP, refined by self-attention within their own tree (with a
    tree-distance bias so attention knows the structure) and cross-attention to the other
    parent, and every pair (i, j) is scored by a bilinear term plus an MLP on [h_i, g_j, h_i*g_j].
    """

    needs_distances = True

    def __init__(self, feature_dim=10, hidden_dim=64, num_heads=4, num_layers=1, mlp_activation='relu',
                 use_critic=True):
        super().__init__()
        self.use_critic = use_critic
        self.num_heads = num_heads
        self.embed = nn.Sequential(nn.Linear(feature_dim, hidden_dim), _activation(mlp_activation),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.self_attn = nn.ModuleList(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                                       for _ in range(num_layers))
        self.cross_attn = nn.ModuleList(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                                        for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2 * num_layers))
        # Learned per-head bias on tree distance (clipped), so self-attention sees the tree's shape
        self.max_dist = 8
        self.dist_bias = nn.Embedding(self.max_dist + 1, num_heads)
        self.bilinear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # First layer of the pair MLP on [h_i, g_j, h_i*g_j], split into its three blocks so the
        # (B, La, Lb, hidden) grid is only built once, for the elementwise-product term
        self.pair_h = nn.Linear(hidden_dim, hidden_dim)
        self.pair_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pair_hg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pair_act = _activation(mlp_activation)
        self.pair_out = nn.Linear(hidden_dim, 1)
        if self.use_critic:
            self.critic = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), _activation(mlp_activation),
                                        nn.Linear(hidden_dim, 1))
        self.to(DEVICE)

    def _mask(self, key_valid, dists=None, n_queries=None):
        """Float attention mask (B*heads, Lq, Lk): distance bias, and -1e9 on padded keys."""
        if dists is not None:
            mask = self.dist_bias(dists.clamp(max=self.max_dist)).permute(0, 3, 1, 2)
        else:
            mask = key_valid.new_zeros((key_valid.shape[0], self.num_heads, n_queries, key_valid.shape[1]),
                                       dtype=torch.float32)
        mask = mask.masked_fill(~key_valid[:, None, None, :], -1e9)
        return mask.reshape(-1, mask.shape[2], mask.shape[3])

    def score_pairs(self, a, b):
        h, g = self.embed(a.feats), self.embed(b.feats)
        a_self, b_self = self._mask(a.valid, a.dists), self._mask(b.valid, b.dists)
        a_cross, b_cross = self._mask(b.valid, n_queries=h.shape[1]), self._mask(a.valid, n_queries=g.shape[1])
        for k, (sa, ca) in enumerate(zip(self.self_attn, self.cross_attn)):
            h = self.norms[2 * k](h + sa(h, h, h, attn_mask=a_self, need_weights=False)[0])
            g = self.norms[2 * k](g + sa(g, g, g, attn_mask=b_self, need_weights=False)[0])
            h, g = (self.norms[2 * k + 1](h + ca(h, g, g, attn_mask=a_cross, need_weights=False)[0]),
                    self.norms[2 * k + 1](g + ca(g, h, h, attn_mask=b_cross, need_weights=False)[0]))
        d = h.shape[-1]
        hidden = (self.pair_h(h).unsqueeze(2) + self.pair_g(g).unsqueeze(1)
                  + self.pair_hg(h.unsqueeze(2) * g.unsqueeze(1)))
        logits = self.bilinear(h) @ g.transpose(1, 2) / d ** 0.5 + self.pair_out(self.pair_act(hidden)).squeeze(-1)
        if not self.use_critic:
            return logits, None
        # Value of the parent pair as a whole (mean over real nodes), shared by all its node pairs
        a_w, b_w = a.valid.unsqueeze(-1).float(), b.valid.unsqueeze(-1).float()
        pooled = torch.cat([(h * a_w).sum(1) / a_w.sum(1), (g * b_w).sum(1) / b_w.sum(1)], -1)
        return logits, self.critic(pooled).expand(-1, h.shape[1]).unsqueeze(-1).expand(-1, -1, g.shape[1])


# Fixed operator order, so node features from runs with different operator sets share one layout
CANONICAL_OPS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'add', 'sub', 'mul', 'div', 'pow']
CANONICAL_FEATURE_DIM = 10 + len(CANONICAL_OPS)


def canonical_feature_map(ops: List[str], base_dim: int = 10) -> torch.Tensor:
    """0/1 matrix (base_dim + len(ops), CANONICAL_FEATURE_DIM) taking node features whose operator
    one-hots follow ops (a run's ALL_UNARY + ALL_BINARY) to the canonical layout."""
    m = torch.zeros(base_dim + len(ops), CANONICAL_FEATURE_DIM)
    m[:base_dim, :base_dim] = torch.eye(base_dim)
    for k, name in enumerate(ops):
        m[base_dim + k, base_dim + CANONICAL_OPS.index(name)] = 1.0
    return m


SEMANTIC_CHANNELS = 2


def pair_scorer_net(hidden_dim: int = 64, semantic: bool = False) -> nn.Module:
    in_dim = 4 * CANONICAL_FEATURE_DIM + (SEMANTIC_CHANNELS if semantic else 0)
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


def semantic_channels(ssd: torch.Tensor) -> torch.Tensor:
    """Pair inputs from the semantic distance of two subtrees (mean absolute output difference):
    scaled log distance, and whether the two are semantically identical. (...) -> (..., 2)."""
    return torch.stack([torch.log10(ssd + 1e-6).clamp(-6, 6) / 6, (ssd < 1e-4).to(ssd.dtype)], -1)


def subtree_semantics(root: Node, inputs: Mapping[str, np.ndarray]) -> torch.Tensor:
    """Outputs of every subtree on the inputs, in pre-order (collect_nodes_with_parents order):
    (N, n_points) float32, non-finite values replaced by 1e6. One bottom-up pass."""
    out = []

    def visit(n):
        i = len(out)
        out.append(None)
        if isinstance(n, Unary):
            v = UNARY_FUNCS[n.func_name](visit(n.child))
        elif isinstance(n, Binary):
            left = visit(n.left)
            v = BINARY_FUNCS[n.func_name](left, visit(n.right))
        else:
            v = n.evaluate(inputs)
        out[i] = v
        return v

    with np.errstate(all='ignore'):
        visit(root)
        sem = np.stack([np.broadcast_to(np.asarray(v, dtype=float), out[0].shape) for v in out])
    sem = np.where(np.isfinite(sem), np.clip(sem, -1e6, 1e6), 1e6)
    return torch.as_tensor(sem, dtype=torch.float32)


def pair_inputs(fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    """[f_a, f_b, f_a*f_b, |f_a-f_b|] for every node pair: (B, La, C), (B, Lb, C) -> (B, La, Lb, 4C)."""
    fa, fb = fa.unsqueeze(2).expand(-1, -1, fb.shape[1], -1), fb.unsqueeze(1).expand(-1, fa.shape[1], -1, -1)
    return torch.cat([fa, fb, fa * fb, (fa - fb).abs()], -1)


class PretrainedPairScorer(nn.Module):
    """Frozen pair scorer trained offline (pretrain_selector.py) on crossovers from other
    equations: logit that a node pair gives a good crossover (the label it was trained on). A
    scorer trained with semantic channels also sees the two subtrees' semantic distance."""

    needs_distances = False

    def __init__(self, path: str, hidden_dim: int = 64):
        super().__init__()
        state = torch.load(path, map_location="cpu")
        self.needs_semantics = state['0.weight'].shape[1] > 4 * CANONICAL_FEATURE_DIM
        self.net = pair_scorer_net(hidden_dim, semantic=self.needs_semantics)
        self.net.load_state_dict(state)
        self.register_buffer('proj', canonical_feature_map(ALL_UNARY + ALL_BINARY))
        self.requires_grad_(False)
        self.to(DEVICE)

    def score_pairs(self, a, b):
        x = pair_inputs(a.feats @ self.proj, b.feats @ self.proj)
        if self.needs_semantics:
            ssd = torch.cdist(a.sem, b.sem, p=1) / a.sem.shape[-1]
            x = torch.cat([x, semantic_channels(ssd)], -1)
        return self.net(x).squeeze(-1), None


def heuristic_bias(a_feats: torch.Tensor, b_feats: torch.Tensor) -> torch.Tensor:
    """Hand-crafted compatibility H(n_i, m_j) for every node pair of two (batched) parents: (B, La, Lb)."""
    a_type = torch.argmax(a_feats[..., :4], dim=-1)
    b_type = torch.argmax(b_feats[..., :4], dim=-1)
    type_match = (a_type.unsqueeze(2) == b_type.unsqueeze(1)).to(a_feats.dtype)
    size_gap = torch.abs(a_feats[..., 4].unsqueeze(2) - b_feats[..., 4].unsqueeze(1))
    depth_gap = torch.abs(a_feats[..., 5].unsqueeze(2) - b_feats[..., 5].unsqueeze(1))
    op_match = a_feats[..., 10:] @ b_feats[..., 10:].transpose(1, 2)
    return 0.15 * type_match + 0.20 * op_match - 0.10 * size_gap - 0.08 * depth_gap


PairSide = collections.namedtuple("PairSide", ["feats", "valid", "dists", "sem"], defaults=(None,))


def make_side(feats: List[torch.Tensor], dists: List[torch.Tensor] | None = None,
              sem: List[torch.Tensor] | None = None) -> PairSide:
    """Pad one parent side of a batch: feats (B, L, F), valid (B, L), and optional dists (B, L, L)
    and subtree outputs sem (B, L, n_points)."""
    lengths = torch.tensor([f.shape[0] for f in feats], device=DEVICE)
    padded = nn.utils.rnn.pad_sequence([f.to(DEVICE) for f in feats], batch_first=True)
    valid = torch.arange(padded.shape[1], device=DEVICE).unsqueeze(0) < lengths.unsqueeze(1)
    padded_dists = None
    if dists is not None:
        padded_dists = torch.zeros(len(feats), padded.shape[1], padded.shape[1], dtype=torch.long, device=DEVICE)
        for i, dm in enumerate(dists):
            padded_dists[i, :dm.shape[0], :dm.shape[1]] = dm
    padded_sem = None if sem is None else nn.utils.rnn.pad_sequence([m.to(DEVICE) for m in sem], batch_first=True)
    return PairSide(padded, valid, padded_dists, padded_sem)


def pair_distribution(selector, a: PairSide, b: PairSide, temperature: float, heuristic_bias_scale: float):
    """Flattened joint logits (B, La*Lb) over node pairs, padded pairs at -inf, and matching
    critic values (B, La*Lb) or None. selector=None scores pairs by H alone."""
    if selector is None:
        logits, values = torch.zeros(a.feats.shape[0], a.feats.shape[1], b.feats.shape[1], device=DEVICE), None
    else:
        logits, values = selector.score_pairs(a, b)
    if heuristic_bias_scale:
        logits = logits + float(heuristic_bias_scale) * heuristic_bias(a.feats, b.feats)
    logits = torch.clamp(logits / max(float(temperature), 1e-6), -10.0, 10.0)
    logits = logits.masked_fill(~(a.valid.unsqueeze(2) & b.valid.unsqueeze(1)), float('-inf'))
    flat = logits.reshape(logits.shape[0], -1)
    return flat, (values.reshape(values.shape[0], -1) if values is not None else None)


def _node_features(node: Node, parent: Node | None, depth_from_root: int, size: int, depth: int,
                   tree_size: int, tree_depth: int) -> List[float]:
    """Feature row for the subtree rooted at node: type, structure, parent type, operator one-hots."""
    features = [
        float(isinstance(node, Const)),
        float(isinstance(node, Var)),
        float(isinstance(node, Unary)),
        float(isinstance(node, Binary)),
        size / max(tree_size, 1),
        depth / max(tree_depth, 1),
        depth_from_root / max(tree_depth, 1),
        float(parent is None),
        float(isinstance(parent, Unary)),
        float(isinstance(parent, Binary)),
    ]
    unary_one_hot = [0.0] * len(ALL_UNARY)
    if isinstance(node, Unary) and node.func_name in ALL_UNARY:
        unary_one_hot[ALL_UNARY.index(node.func_name)] = 1.0
    binary_one_hot = [0.0] * len(ALL_BINARY)
    if isinstance(node, Binary) and node.func_name in ALL_BINARY:
        binary_one_hot[ALL_BINARY.index(node.func_name)] = 1.0
    return features + unary_one_hot + binary_one_hot


def collect_nodes_with_features(root: Node, with_distances=False):
    """Pre-order [(node, parent, attr)], an (N, feature_dim) feature tensor and, if asked,
    the (N, N) matrix of tree (edge-count) distances between nodes."""
    nodes, info, parent_idx = [], [], []

    def visit(n, p, attr, d, p_idx):
        i = len(nodes)
        nodes.append((n, p, attr))
        info.append(None)
        parent_idx.append(p_idx)
        size, depth = 1, 0
        if isinstance(n, Unary):
            s, h = visit(n.child, n, 'child', d + 1, i)
            size, depth = 1 + s, 1 + h
        elif isinstance(n, Binary):
            ls, lh = visit(n.left, n, 'left', d + 1, i)
            rs, rh = visit(n.right, n, 'right', d + 1, i)
            size, depth = 1 + ls + rs, 1 + max(lh, rh)
        info[i] = (n, p, d, size, depth)
        return size, depth

    tree_size, tree_depth = visit(root, None, None, 0, -1)
    rows = [_node_features(*row, tree_size, tree_depth) for row in info]
    feats = torch.tensor(rows, dtype=torch.float32, device=DEVICE)
    if not with_distances:
        return nodes, feats, None
    # dist(i, j) = depth(i) + depth(j) - 2 * depth(lca); ancestors via parent pointers
    n = len(nodes)
    anc = np.zeros((n, n), dtype=bool)
    for i in range(n):
        anc[i, i] = True
        if parent_idx[i] >= 0:
            anc[i] |= anc[parent_idx[i]]
    depths = np.array([row[2] for row in info])
    common = anc.astype(np.int64) @ anc.T.astype(np.int64)  # number of shared ancestors = depth(lca) + 1
    dists = depths[:, None] + depths[None, :] - 2 * (common - 1)
    return nodes, feats, torch.as_tensor(dists, device=DEVICE)


# Selector state for the current run_ga call
subtree_selector = None
selector_optimizer = None
crossover_history = []
SELECTOR_TYPE = 'mlp'
SELECTOR_MIX = 1.0  # share of crossovers whose points the selector picks; the rest are uniform
CROSSOVER_INPUTS = None  # training inputs, for semantic crossover
SSC_BOUNDS = (1e-4, 0.4)
CROSSOVER_SCORE = None  # run_ga's fitness function, for greedy crossover
GREEDY_TRIALS = 8
INTERNAL_NODE_PROB = None  # Koza node selection for non-learned crossover and mutation points, or None for uniform


def pick_random_node(nodes):
    """A non-learned crossover or mutation point: uniform, or with INTERNAL_NODE_PROB an internal node
    with that probability and a leaf otherwise (Koza's node selector)."""
    if INTERNAL_NODE_PROB is None:
        return random.choice(nodes)
    internal = [n for n in nodes if isinstance(n[0], (Unary, Binary))]
    leaves = [n for n in nodes if not isinstance(n[0], (Unary, Binary))]
    if internal and (not leaves or random.random() < INTERNAL_NODE_PROB):
        return random.choice(internal)
    return random.choice(leaves)
PAIR_BATCH_BUDGET = 1 << 15  # max padded node pairs per batched selector forward pass
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:  # not glibc
    _libc = None


def r2_score_of(y: np.ndarray, y_pred: np.ndarray) -> float | None:
    """R² of a prediction, or None if it has NaN/inf or y is constant."""
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return None
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - np.sum((y - y_pred) ** 2) / ss_tot if ss_tot > 0 else None


# Genetic algorithm
def run_ga(
    x,
    y: np.ndarray,
    pop_size=300,
    gens=50,
    max_init_depth=4,
    max_depth=7,
    crossover_rate=0.7,
    mutation_rate=0.3,
    tournament_k=3,
    elitism=1,
    var_names=None,
    unary_funcs=None,
    use_learned_crossover=True,
    selector_type='mlp',
    selector_lr=0.001,
    mlp_layers=3,
    mlp_hidden_dim=64,
    mlp_activation='relu',
    mlp_dropout=0.0,
    attn_heads=4,
    attn_layers=1,
    selector_path=None,
    selector_mix=1.0,
    internal_node_prob=None,
    init_method='grow',
    mutation_depth=None,
    mutate_crossover_children=True,
    init_min_depth=2,
    ssc_bounds=(1e-4, 0.4),
    selector_reward='delta',
    greedy_trials=8,
    use_critic=True,
    stopping_metric='mse',
    stopping_threshold=None,
    complexity_coef=1e-3,
    inclusion_coef=1.0,
    prob_const=0.3,
    const_range=(-5, 5),
    point_mutation_prob=0.05,
    simplify_new_genomes=False,
    simplify_timeout=0.2,
    selector_update_freq=5,
    selector_warmup_gens=8,
    selector_recent_gen_window=2,
    selector_min_history=64,
    selector_reward_clip=5.0,
    temperature=0.5,
    critic_weight=0.5,
    entropy_weight=0.01,
    use_pareto=False,
    opt_constants=False,
    const_opt_freq=5,
    const_opt_method='L-BFGS-B',
    const_opt_maxiter=50,
    verbose=False,
    print_freq=1,
    seed=DEFAULT_SEED,
    metrics_callback=None,
    allow_pow=False,
    selector_device="auto",
    heuristic_bias_scale=1.0,
    linear_terms=False,
    max_terms=20,
    add_term_prob=0.0,
    quadratic_init_frac=0.0,
    var_dimensions=None,
    target_dimension=None,
):
    """Evolve an expression for y. selector_type picks the learned crossover policy:
    'mlp' (per-node SubtreeSelector), 'pair' (cross-attention PairSelector), 'pretrained'
    (frozen PretrainedPairScorer loaded from selector_path), 'heuristic' (H alone, no learning)
    or 'ssc' (semantic similarity-based crossover within ssc_bounds), or 'greedy' (an upper bound
    that scores the children of greedy_trials uniform point pairs and keeps the best). selector_mix is the share of
    crossovers whose points the selector picks. internal_node_prob=0.9 makes
    non-learned crossover points Koza's (90% internal nodes, 10% leaves) instead of uniform;
    init_method='ramped' is ramped half-and-half over init_min_depth..max_init_depth.
    mutation_depth caps mutation subtrees (None draws their depth from 0..max_depth), and with
    mutate_crossover_children=False crossover and mutation are exclusive operators. selector_reward is the selector's reward (see
    update_selector). Returns (best_tree, best_r2)."""
    global subtree_selector, selector_optimizer, crossover_history, SELECTOR_TYPE, DEVICE
    global SELECTOR_MIX, CROSSOVER_INPUTS, SSC_BOUNDS, CROSSOVER_SCORE, GREEDY_TRIALS, INTERNAL_NODE_PROB

    if seed is None:
        seed = DEFAULT_SEED
    set_global_seed(seed)

    inputs, resolved_names = prepare_inputs(x, var_names)
    set_variables(resolved_names)

    DEVICE = _resolve_selector_device(selector_device)
    if verbose and use_learned_crossover:
        print(f"Selector device: {DEVICE}")

    # Configure unary functions
    global UNARY_FUNCS, SYMPY_UNARY, ALL_UNARY
    if unary_funcs is not None:
        if not isinstance(unary_funcs, dict):
            raise ValueError("unary_funcs must be a dict mapping names to (numpy_func, sympy_func) tuples")
        UNARY_FUNCS, SYMPY_UNARY = {}, {}
        for name, funcs in unary_funcs.items():
            if not (isinstance(funcs, tuple) and len(funcs) == 2):
                raise ValueError(f"Each unary function must be a (numpy_func, sympy_func) tuple, got {funcs}")
            UNARY_FUNCS[name], SYMPY_UNARY[name] = funcs
    else:
        UNARY_FUNCS = dict(DEFAULT_UNARY_FUNCS)
        SYMPY_UNARY = {name: DEFAULT_SYMPY_UNARY[name] for name in UNARY_FUNCS}
    ALL_UNARY = list(UNARY_FUNCS.keys())

    # Configure binary functions (optional pow support)
    global BINARY_FUNCS, SYMPY_BINARY, ALL_BINARY
    BINARY_FUNCS = dict(BASE_BINARY_FUNCS)
    SYMPY_BINARY = dict(BASE_SYMPY_BINARY)
    if allow_pow:
        BINARY_FUNCS["pow"] = ppow
        SYMPY_BINARY["pow"] = lambda a, b: a ** b
    ALL_BINARY = list(BINARY_FUNCS.keys())

    global LINEAR_TERMS, MAX_TERMS
    LINEAR_TERMS = bool(linear_terms)
    MAX_TERMS = int(max_terms)
    set_dimensions(var_dimensions, target_dimension)

    crossover_history = []

    init_max_depth = max(1, max_init_depth)
    if init_method == 'ramped':
        # Ramped half-and-half: depths cycle over init_min_depth..max_init_depth, alternating full and grow
        depths = list(range(min(init_min_depth, init_max_depth), init_max_depth + 1))
        pop = [random_tree(depths[(i // 2) % len(depths)], prob_const, const_range, full=i % 2 == 0)
               for i in range(pop_size)]
    else:
        pop = [
            random_tree(max_depth=random.randint(1, init_max_depth), prob_const=prob_const, const_range=const_range)
            for _ in range(pop_size)
        ]
    for i in range(int(quadratic_init_frac * pop_size)):
        pop[i] = random_quadratic(random.randint(2, max(2, max_terms // 2)))
    if simplify_new_genomes:
        pop = [simplify_node_with_timeout(ind, simplify_timeout) for ind in pop]

    # Initialize the crossover selector
    if selector_type not in ('mlp', 'pair', 'pretrained', 'heuristic', 'ssc', 'greedy'):
        raise ValueError("selector_type must be 'mlp', 'pair', 'pretrained', 'heuristic', 'ssc' or 'greedy'")
    SELECTOR_TYPE = selector_type
    SELECTOR_MIX, CROSSOVER_INPUTS, SSC_BOUNDS = float(selector_mix), inputs, tuple(ssc_bounds)
    GREEDY_TRIALS = int(greedy_trials)
    INTERNAL_NODE_PROB = internal_node_prob
    subtree_selector = selector_optimizer = None
    if use_learned_crossover and selector_type == 'mlp':
        subtree_selector = SubtreeSelector(
            feature_dim=get_selector_feature_dim(),
            hidden_dim=mlp_hidden_dim,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            use_critic=use_critic,
        )
    elif use_learned_crossover and selector_type == 'pair':
        subtree_selector = PairSelector(
            feature_dim=get_selector_feature_dim(),
            hidden_dim=mlp_hidden_dim,
            num_heads=attn_heads,
            num_layers=attn_layers,
            mlp_activation=mlp_activation,
            use_critic=use_critic,
        )
    elif use_learned_crossover and selector_type == 'pretrained':
        subtree_selector = PretrainedPairScorer(selector_path, hidden_dim=mlp_hidden_dim)
    if subtree_selector is not None and selector_type != 'pretrained':
        selector_optimizer = optim.Adam(subtree_selector.parameters(), lr=selector_lr)
    if verbose:
        print(f"Crossover: {selector_type if use_learned_crossover else 'random'}"
              + (f" ({sum(p.numel() for p in subtree_selector.parameters())} params, "
                 f"{'A2C' if use_critic else 'policy gradient'})" if subtree_selector is not None else ""))

    # Validate stopping criterion
    if stopping_threshold is not None:
        if stopping_metric not in ['mse', 'r2', None]:
            raise ValueError("stopping_metric must be 'mse', 'r2', or None")
        if stopping_metric == 'r2' and (stopping_threshold < 0 or stopping_threshold > 1):
            print(f"Warning: R² stopping threshold {stopping_threshold} is outside typical range [0, 1]")
        if verbose:
            op = "<=" if stopping_metric == 'mse' else ">="
            print(f"Stopping criterion: {stopping_metric} {op} {stopping_threshold}")

    y = np.asarray(y, dtype=float).reshape(-1)
    n_samples = len(next(iter(inputs.values())))
    if y.shape[0] != n_samples:
        raise ValueError("y must have the same length as the input samples.")

    def evaluate_r2(individual):
        try:
            return r2_score_of(y, individual.evaluate(inputs))
        except Exception:
            return None

    # Many trees recur (elites, unchanged clones, offspring scored by the selector update)
    fitness_cache = {}

    def score(ind):
        key = ind.key()
        if key not in fitness_cache:
            if len(fitness_cache) >= 200_000:
                fitness_cache.clear()
            fitness_cache[key] = score_individual(ind, inputs, y, complexity_coef, inclusion_coef)
        return fitness_cache[key]

    CROSSOVER_SCORE = score

    best = None
    best_score = float('inf')
    best_r2 = None

    for gen in tqdm(range(gens), desc="GA generations", unit="gen"):
        if LINEAR_TERMS:
            pop = [bake_linear(ind, inputs, y) for ind in pop]
        scores, objectives = zip(*(score(ind) for ind in pop))
        scores = list(scores)
        if use_pareto:
            ranks, crowding = pareto_rank_and_crowding(list(objectives))
            tournament_key = lambda i: (ranks[i], -crowding[i], scores[i])
        else:
            tournament_key = lambda i: scores[i]
        sorted_idx = sorted(range(len(pop)), key=tournament_key)

        learning = use_learned_crossover and gen >= max(1, int(selector_warmup_gens))
        if learning and selector_optimizer is not None and gen % selector_update_freq == 0:
            update_selector(
                score,
                critic_weight=critic_weight,
                entropy_weight=entropy_weight,
                recent_gen_window=selector_recent_gen_window,
                min_history=selector_min_history,
                reward_clip=selector_reward_clip,
                current_gen=gen,
                temperature=temperature,
                heuristic_bias_scale=heuristic_bias_scale,
                reward_mode=selector_reward,
                pop_scores=scores,
            )

        # track best
        for ind, s in zip(pop, scores):
            if s < best_score:
                best_score = s
                best = ind.clone()
                r2_val = evaluate_r2(best)
                if r2_val is not None:
                    best_r2 = r2_val

        # Check stopping criterion
        if stopping_threshold is not None and best is not None:
            try:
                y_pred = best.evaluate(inputs)
                if not (np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred))):
                    if stopping_metric == 'mse':
                        value = float(np.mean((y - y_pred) ** 2))
                        reached = value <= stopping_threshold
                    else:
                        value = r2_score_of(y, y_pred) or 0.0
                        reached = stopping_metric == 'r2' and value >= stopping_threshold
                    if reached:
                        if verbose:
                            print(f"\nStopping criterion reached at generation {gen+1}: {stopping_metric} = {value:.6g}")
                        break
            except Exception:
                pass

        next_pop = []
        # elitism, optionally optimizing the elites' constants
        optimize_elites = opt_constants and gen > 0 and gen % const_opt_freq == 0
        if optimize_elites and verbose:
            print(f"\nOptimizing constants for elites at generation {gen+1}...")
        for i in range(elitism):
            elite = pop[sorted_idx[i]].clone()
            if optimize_elites:
                elite = optimize_constants(elite, inputs, y, method=const_opt_method, maxiter=const_opt_maxiter)
            if simplify_new_genomes:
                elite = simplify_node_with_timeout(elite, simplify_timeout)
            next_pop.append(elite)

        # produce rest
        while len(next_pop) < pop_size:
            if random.random() < crossover_rate:
                i1 = tournament_pick_index(len(pop), tournament_key, tournament_k)
                parent1 = pop[i1].clone()
                i2 = tournament_pick_index(len(pop), tournament_key, tournament_k)
                parent2 = pop[i2].clone()
                child1, child2, _ = learned_subtree_crossover(
                    parent1,
                    parent2,
                    use_learned=learning,
                    temperature=temperature,
                    record_gen=gen,
                    heuristic_bias_scale=heuristic_bias_scale,
                    parent_scores=(scores[i1], scores[i2]),
                )
                if mutate_crossover_children and random.random() < mutation_rate:
                    child1 = subtree_mutation(child1, max_depth, prob_const, const_range, mutation_depth)
                if mutate_crossover_children and random.random() < mutation_rate:
                    child2 = subtree_mutation(child2, max_depth, prob_const, const_range, mutation_depth)
                if simplify_new_genomes:
                    child1 = simplify_node_with_timeout(child1, simplify_timeout)
                    child2 = simplify_node_with_timeout(child2, simplify_timeout)
                if random.random() < add_term_prob:
                    child1 = add_term_mutation(child1)
                if random.random() < add_term_prob:
                    child2 = add_term_mutation(child2)
                next_pop.append(parent1.clone() if too_big(child1, max_depth) else child1)
                if len(next_pop) < pop_size:
                    next_pop.append(parent2.clone() if too_big(child2, max_depth) else child2)
            else:
                parent = pop[tournament_pick_index(len(pop), tournament_key, tournament_k)].clone()
                child = parent
                if random.random() < mutation_rate:
                    child = subtree_mutation(child, max_depth, prob_const, const_range, mutation_depth)
                # small point mutations
                child = point_mutation(child, prob_change=point_mutation_prob)
                if simplify_new_genomes:
                    child = simplify_node_with_timeout(child, simplify_timeout)
                if random.random() < add_term_prob:
                    child = add_term_mutation(child)
                next_pop.append(parent.clone() if too_big(child, max_depth) else child)
        pop = next_pop

        if verbose and ((gen + 1) % print_freq == 0 or gen == 0):
            r2_best = evaluate_r2(best)
            r2_text = f" R²: {r2_best:.6f}" if r2_best is not None else ""
            print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g}{r2_text} expr: {best.to_string()}")

        if metrics_callback is not None and best is not None:
            r2 = evaluate_r2(best)
            metrics_callback(gen, float(r2) if r2 is not None else float("nan"))

    # best_r2 is only refreshed when the new best evaluates cleanly; report the final best's own
    if best is not None:
        best_r2 = evaluate_r2(best)

    return best, best_r2


# Random tree generation with configurable constants
def random_tree(max_depth: int, prob_const=0.3, const_range=(-5, 5), full=False) -> Node:
    """Random tree of depth <= max_depth (Koza's grow), or with full every branch reaches exactly
    max_depth (Koza's full)."""
    if max_depth == 0:
        if random.random() < prob_const:
            return Const(random.uniform(*const_range))
        return Var(random.choice(VAR_NAMES))

    # Build choices dynamically so empty operator sets don't crash.
    choices = [] if full and (ALL_UNARY or ALL_BINARY) else ["const", "var"]
    weights = [0.2] * len(choices)
    if ALL_UNARY:
        choices.append("unary")
        weights.append(0.3)
    if ALL_BINARY:
        choices.append("binary")
        weights.append(0.3)

    total = sum(weights)
    pick = random.random() * total
    cumulative = 0.0
    for choice, weight in zip(choices, weights):
        cumulative += weight
        if pick <= cumulative:
            if choice == "const":
                return Const(random.uniform(*const_range))
            if choice == "var":
                return Var(random.choice(VAR_NAMES))
            if choice == "unary":
                func = random.choice(ALL_UNARY)
                return Unary(func, random_tree(max_depth - 1, prob_const, const_range, full))
            func = random.choice(ALL_BINARY)
            return Binary(
                func,
                random_tree(max_depth - 1, prob_const, const_range, full),
                random_tree(max_depth - 1, prob_const, const_range, full),
            )

    # Fallback (should not hit)
    return Var()


# Linear-terms mode: the top-level sum of a tree is treated as a linear model whose
# weights and intercept are fitted by least squares instead of being evolved.
LINEAR_TERMS = False
MAX_TERMS = 20

# --- Dimensional consistency (opt-in via var_dimensions/target_dimension in run_ga) ---
#
# Each variable is tagged with a "unit" -- a dict of {category: exponent}, e.g. {"Pr": 1}
# for a plain precipitation-typed variable, or {} for a dimensionless one (a fraction,
# a sin/cos of month, or a plain number). Two rules are enforced when a target unit is set:
#   - add/sub:  both sides must carry the same unit, unless one side is a pure number
#               (a constant, or an expression built only from constants), which is always
#               allowed since it acts like an implicit offset in whatever unit the other
#               side has.
#   - sin/cos/exp/log: the argument must reduce to dimensionless. A constant that enters
#               through *multiplication* (never addition) is allowed to cancel out
#               whatever unit is left over, e.g. cos(Pr * C) is fine because C can be
#               seen as carrying the inverse unit of Pr -- but a bare cos(Te) is not,
#               and neither is cos(Pr + 5), since an added constant can't rescale a unit.
# Multiplication/division themselves are never rejected -- Pr*Te is a legal compound
# unit -- so this stays "loose": it only stops additive apples-and-oranges mistakes and
# transcendental functions being fed a raw dimensioned quantity with no scale factor.
DimInfo = collections.namedtuple("DimInfo", ["units", "has_var", "has_const"])
DIMENSIONLESS = DimInfo({}, False, False)

VAR_DIMENSIONS: Dict[str, Dict[str, int]] = {}
TARGET_DIMENSION: Dict[str, int] | None = None
VALID_TERM_VARS: List[str] = []          # single vars whose unit already matches the target
VALID_TERM_PAIRS: List[Tuple[str, str]] = []  # var pairs whose combined unit matches the target


def _merge_units(a: Dict[str, int], b: Dict[str, int], sign: int) -> Dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + sign * v
    return {k: v for k, v in out.items() if v != 0}


def compute_dimension(node: Node) -> DimInfo | None:
    """Return this subtree's DimInfo, or None if it violates a dimensional rule."""
    if not VAR_DIMENSIONS and TARGET_DIMENSION is None:
        return DIMENSIONLESS  # dimension checking is off
    if isinstance(node, Const):
        return DimInfo({}, False, True)
    if isinstance(node, Var):
        return DimInfo(VAR_DIMENSIONS.get(node.name, {}), True, False)
    if isinstance(node, Unary):
        child = compute_dimension(node.child)
        if child is None:
            return None
        if node.func_name in ("sin", "cos", "exp", "log"):
            if child.units and not child.has_const:
                return None  # dimensioned argument with no scale factor to cancel it
            return DimInfo({}, child.has_var, False)
        return DimInfo(child.units, child.has_var, child.has_const)
    if isinstance(node, Binary):
        left = compute_dimension(node.left)
        right = compute_dimension(node.right)
        if left is None or right is None:
            return None
        if node.func_name in ("add", "sub"):
            if left.has_var and right.has_var and left.units != right.units:
                return None  # apples and oranges
            units = left.units if left.has_var else right.units
            has_var = left.has_var or right.has_var
            has_const = (not left.has_var) and (not right.has_var)  # only "free" if the whole thing is a number
            return DimInfo(units, has_var, has_const)
        if node.func_name in ("mul", "div"):
            units = _merge_units(left.units, right.units, 1 if node.func_name == "mul" else -1)
            return DimInfo(units, left.has_var or right.has_var, left.has_const or right.has_const)
        if node.func_name == "pow":
            if left.has_var or right.has_var:
                return None  # only constant-only subexpressions may use pow under dimension checking
            return DimInfo({}, False, True)
    return DIMENSIONLESS


def term_is_valid(term: Node) -> bool:
    """Whether a bare (unweighted) term's dimension matches the target's, given the current
    dimension registry. Always True when dimension checking is off."""
    if not VAR_DIMENSIONS and TARGET_DIMENSION is None:
        return True
    d = compute_dimension(term)
    if d is None:
        return False
    return (not d.has_var) or d.units == TARGET_DIMENSION


def set_dimensions(var_dimensions: Dict[str, Dict[str, int]] | None, target_dimension: Dict[str, int] | None):
    """Configure dimensional checking for the current run_ga call. Pass None for both to
    turn checking off (the default)."""
    global VAR_DIMENSIONS, TARGET_DIMENSION, VALID_TERM_VARS, VALID_TERM_PAIRS
    VAR_DIMENSIONS = dict(var_dimensions) if var_dimensions else {}
    TARGET_DIMENSION = target_dimension
    VALID_TERM_VARS = []
    VALID_TERM_PAIRS = []
    if not VAR_DIMENSIONS and target_dimension is None:
        return
    for name in VAR_NAMES:
        if term_is_valid(Var(name)):
            VALID_TERM_VARS.append(name)
    for a in VAR_NAMES:
        for b in VAR_NAMES:
            if term_is_valid(Binary("mul", Var(a), Var(b))):
                VALID_TERM_PAIRS.append((a, b))
    if not VALID_TERM_VARS and not VALID_TERM_PAIRS:
        raise ValueError(
            "No combination of the given variables (as a single variable or a product of two) "
            "produces the required target_dimension. Check var_dimensions/target_dimension."
        )


def split_terms(node: Node) -> List[Node]:
    """Split a tree on top-level add/sub into terms, dropping bare constants and constant weights."""
    if isinstance(node, Binary) and node.func_name in ("add", "sub"):
        return split_terms(node.left) + split_terms(node.right)
    if isinstance(node, Binary) and node.func_name == "mul":
        if isinstance(node.left, Const):
            return split_terms(node.right) if not isinstance(node.right, Const) else []
        if isinstance(node.right, Const):
            return split_terms(node.left)
    if isinstance(node, Const):
        return []
    return [node]


def linear_fit(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray):
    """Least-squares fit of y ~ b0 + sum(w_i * term_i). Returns (b0, weights, terms, y_pred) or None."""
    terms = [t for t in split_terms(ind) if term_is_valid(t)]
    cols, kept = [], []
    for t in terms:
        try:
            v = np.asarray(t.evaluate(inputs), dtype=float)
        except Exception:
            return None
        if not np.all(np.isfinite(v)) or np.abs(v).max() > 1e8:
            return None
        if v.std() > 1e-9:
            cols.append(v)
            kept.append(t)
    b0 = float(np.mean(y))
    if not cols:
        return b0, np.zeros(0), [], np.full_like(y, b0, dtype=float)
    A = np.column_stack(cols)
    A_mu = A.mean(axis=0)
    w, *_ = np.linalg.lstsq(A - A_mu, y - b0, rcond=1e-10)
    b0 = b0 - float(A_mu @ w)
    return b0, w, kept, b0 + A @ w


def bake_linear(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray) -> Node:
    """Rebuild the tree as b0 + w1*t1 + ... with least-squares weights."""
    fit = linear_fit(ind, inputs, y)
    if fit is None:
        return ind
    b0, w, terms, _ = fit
    tree: Node = Const(b0)
    for wi, t in zip(w, terms):
        tree = Binary("add", tree, Binary("mul", Const(float(wi)), t.clone()))
    return tree


def predict(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    if LINEAR_TERMS:
        fit = linear_fit(ind, inputs, y)
        if fit is None:
            raise ValueError("invalid terms")
        return fit[3]
    return ind.evaluate(inputs)


def too_big(ind: Node, max_depth: int) -> bool:
    if LINEAR_TERMS:
        terms = split_terms(ind)
        return len(terms) > MAX_TERMS or any(t.depth() > max_depth for t in terms)
    return ind.depth() > max_depth


def random_term() -> Node:
    """A linear or quadratic term: x_i or x_i * x_j. When dimension checking is on, only
    draws from the combinations whose unit actually matches the target's."""
    if VALID_TERM_VARS or VALID_TERM_PAIRS:
        if VALID_TERM_VARS and (not VALID_TERM_PAIRS or random.random() < 0.3):
            return Var(random.choice(VALID_TERM_VARS))
        a, b = random.choice(VALID_TERM_PAIRS)
        return Binary("mul", Var(a), Var(b))
    if random.random() < 0.3:
        return Var(random.choice(VAR_NAMES))
    return Binary("mul", Var(random.choice(VAR_NAMES)), Var(random.choice(VAR_NAMES)))


def add_term_mutation(ind: Node) -> Node:
    return Binary("add", ind.clone(), random_term())


def random_quadratic(n_terms: int) -> Node:
    tree = random_term()
    for _ in range(n_terms - 1):
        tree = Binary("add", tree, random_term())
    return tree


# Fitness of a tree whose output is NaN or inf: above any finite MSE, so invalid trees never beat
# valid ones (a fixed 1e6 did when the target's variance is large), ordered by size among themselves
INVALID_SCORE = 1e18


def score_individual(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
                     complexity_coef=1e-3, inclusion_coef=1.0) -> Tuple[float, Tuple[float, float]]:
    """Evaluate once; return (fitness, pareto objectives (error_with_inclusion, size))."""
    size = ind.size()
    try:
        y_pred = predict(ind, inputs, y)
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            return INVALID_SCORE + size, (INVALID_SCORE + size, float(size))
        mse = float(np.mean((y - y_pred)**2))
    except Exception:
        return INVALID_SCORE + size, (INVALID_SCORE + size, float(size))
    inclusion_penalty = 0.0
    if inclusion_coef:
        used_vars = gather_used_var_names(ind)
        missing_vars = set(inputs.keys()) - used_vars
        # Compare raw tree outputs: in linear-terms mode y_pred is the refitted prediction
        base_pred = ind.evaluate(inputs) if LINEAR_TERMS else y_pred
        missing_vars |= detect_inert_variables(ind, inputs, base_pred, candidate_vars=used_vars)
        inclusion_penalty = inclusion_coef * len(missing_vars)
    return mse + complexity_coef * size + inclusion_penalty, (mse + inclusion_penalty, float(size))


# Fitness: mean squared error + complexity penalty
def fitness(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
            complexity_coef=1e-3, inclusion_coef=1.0) -> float:
    return score_individual(ind, inputs, y, complexity_coef, inclusion_coef)[0]


def subtree_mutation(ind: Node, max_depth: int, prob_const=0.3, const_range=(-5, 5), new_depth=None) -> Node:
    """Replace a random subtree by a grown one of depth <= new_depth, or of a depth drawn from
    0..max_depth when new_depth is None."""
    ind_clone = ind.clone()
    nodes = collect_nodes_with_parents(ind_clone)
    node, parent, attr = pick_random_node(nodes)
    depth = random.randint(0, max_depth) if new_depth is None else new_depth
    new_sub = random_tree(max_depth=depth, prob_const=prob_const, const_range=const_range)
    if parent is None:
        return new_sub
    else:
        replace_child(parent, attr, new_sub)
        return ind_clone


def semantic_similarity_pick(a_nodes, b_nodes, trials=12):
    """Semantic similarity-based crossover (Uy et al., 2011): retry random point pairs
    (pick_random_node) until the
    two subtrees' mean absolute output difference on the training inputs lies inside SSC_BOUNDS,
    keeping the last pair after `trials` misses."""
    lower, upper = SSC_BOUNDS
    for _ in range(trials):
        pa, pb = pick_random_node(a_nodes), pick_random_node(b_nodes)
        with np.errstate(all='ignore'):
            dist = np.mean(np.abs(pa[0].evaluate(CROSSOVER_INPUTS) - pb[0].evaluate(CROSSOVER_INPUTS)))
        if lower < dist < upper:
            break
    return pa, pb


_NODE_SCORES, _NODE_SCORES_GEN = {}, None


def node_scores(tree: Node, gen):
    """(feature tensor, features as numpy, MLP node logits) for a tree, cached for one generation
    (the selector's weights only change between generations)."""
    global _NODE_SCORES, _NODE_SCORES_GEN
    if gen != _NODE_SCORES_GEN or len(_NODE_SCORES) > 5000:
        _NODE_SCORES, _NODE_SCORES_GEN = {}, gen
    key = tree.key()
    if key not in _NODE_SCORES:
        _, feats, _ = collect_nodes_with_features(tree)
        with torch.no_grad():
            logits = subtree_selector(feats).double().cpu().numpy()
        _NODE_SCORES[key] = (feats.cpu(), feats.cpu().numpy(), logits)
    return _NODE_SCORES[key]


def heuristic_bias_np(a_feats: np.ndarray, b_feats: np.ndarray) -> np.ndarray:
    """heuristic_bias for one unbatched pair of parents, in numpy: (La, Lb)."""
    type_match = np.argmax(a_feats[:, :4], 1)[:, None] == np.argmax(b_feats[:, :4], 1)[None, :]
    size_gap = np.abs(a_feats[:, 4, None] - b_feats[None, :, 4])
    depth_gap = np.abs(a_feats[:, 5, None] - b_feats[None, :, 5])
    op_match = a_feats[:, 10:] @ b_feats[:, 10:].T
    return 0.15 * type_match + 0.20 * op_match - 0.10 * size_gap - 0.08 * depth_gap


def learned_subtree_crossover(
    a: Node,
    b: Node,
    use_learned=True,
    temperature=0.5,
    record_gen: int | None = None,
    heuristic_bias_scale: float = 1.0,
    parent_scores: Tuple[float, float] | None = None,
) -> Tuple[Node, Node, int | None]:
    """Swap one subtree between clones of a and b. With use_learned, the crossover points are
    sampled from the selector's joint pair distribution (or H alone when there is no network);
    otherwise they are uniform. Learned choices are recorded for the next selector update."""
    if use_learned and SELECTOR_TYPE == 'greedy':
        # Upper bound on point selection: keep the uniform pair whose better child scores best
        trials = [learned_subtree_crossover(a, b, use_learned=False) for _ in range(GREEDY_TRIALS)]
        return min(trials, key=lambda t: min(CROSSOVER_SCORE(t[0])[0], CROSSOVER_SCORE(t[1])[0]))

    a_clone = a.clone()
    b_clone = b.clone()
    use_selector = (use_learned and (subtree_selector is not None or SELECTOR_TYPE == 'heuristic')
                    and (SELECTOR_MIX >= 1.0 or random.random() < SELECTOR_MIX))

    if use_learned and SELECTOR_TYPE == 'ssc':
        (node_a, parent_a, attr_a), (node_b, parent_b, attr_b) = semantic_similarity_pick(
            collect_nodes_with_parents(a_clone), collect_nodes_with_parents(b_clone))
    elif use_selector and isinstance(subtree_selector, SubtreeSelector):
        # The MLP's pair logit is a sum of node logits, so each parent's features and node logits
        # are computed once per generation (tournaments pick the same parents many times) and the
        # pair distribution is built in numpy; the distribution is the same as pair_distribution's
        a_nodes, b_nodes = collect_nodes_with_parents(a_clone), collect_nodes_with_parents(b_clone)
        (a_feats, a_np, a_logits), (b_feats, b_np, b_logits) = (node_scores(t, record_gen) for t in (a, b))
        a_dists = b_dists = None
        logits = a_logits[:, None] + b_logits[None, :]
        if heuristic_bias_scale:
            logits = logits + float(heuristic_bias_scale) * heuristic_bias_np(a_np, b_np)
        logits = np.clip(logits / max(float(temperature), 1e-6), -10.0, 10.0).ravel()
        probs = np.exp(logits - logits.max())
        cdf = np.cumsum(probs)
        flat_idx = min(int(np.searchsorted(cdf, random.random() * cdf[-1], side='right')), len(cdf) - 1)
        a_idx, b_idx = divmod(flat_idx, len(b_nodes))
        node_a, parent_a, attr_a = a_nodes[a_idx]
        node_b, parent_b, attr_b = b_nodes[b_idx]
    elif use_selector:
        with_dists = getattr(subtree_selector, 'needs_distances', False)
        with_sem = getattr(subtree_selector, 'needs_semantics', False)
        a_nodes, a_feats, a_dists = collect_nodes_with_features(a_clone, with_dists)
        b_nodes, b_feats, b_dists = collect_nodes_with_features(b_clone, with_dists)
        with torch.no_grad():
            flat_logits, _ = pair_distribution(
                subtree_selector,
                make_side([a_feats], [a_dists] if with_dists else None,
                          [subtree_semantics(a_clone, CROSSOVER_INPUTS)] if with_sem else None),
                make_side([b_feats], [b_dists] if with_dists else None,
                          [subtree_semantics(b_clone, CROSSOVER_INPUTS)] if with_sem else None),
                temperature,
                heuristic_bias_scale,
            )
            probs = torch.softmax(flat_logits[0], dim=0)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()
            flat_idx = int(torch.multinomial(probs, 1).item()) if probs.numel() > 1 else 0
        a_idx, b_idx = divmod(flat_idx, len(b_nodes))
        node_a, parent_a, attr_a = a_nodes[a_idx]
        node_b, parent_b, attr_b = b_nodes[b_idx]
    else:
        node_a, parent_a, attr_a = pick_random_node(collect_nodes_with_parents(a_clone))
        node_b, parent_b, attr_b = pick_random_node(collect_nodes_with_parents(b_clone))

    # Swap subtrees
    if parent_a is None:
        new_a = node_b.clone()
    else:
        new_a = a_clone
        replace_child(parent_a, attr_a, node_b.clone())

    if parent_b is None:
        new_b = node_a.clone()
    else:
        new_b = b_clone
        replace_child(parent_b, attr_b, node_a.clone())

    # Store crossover info for later learning
    history_idx = None
    if use_selector and selector_optimizer is not None:
        crossover_history.append({
            'a': (a_feats.cpu(), None if a_dists is None else a_dists.cpu()),
            'b': (b_feats.cpu(), None if b_dists is None else b_dists.cpu()),
            'a_idx': a_idx,
            'b_idx': b_idx,
            'parents': (a, b),
            'parent_scores': parent_scores,
            'offspring': (new_a.clone(), new_b.clone()),
            'gen': record_gen,
        })
        history_idx = len(crossover_history) - 1

    return new_a, new_b, history_idx


def update_selector(
    score_fn,
    critic_weight=0.5,
    entropy_weight=0.01,
    recent_gen_window=2,
    min_history=64,
    reward_clip=5.0,
    current_gen: int | None = None,
    temperature: float = 0.5,
    heuristic_bias_scale: float = 1.0,
    reward_mode: str = 'delta',
    pop_scores=None,
) -> float | None:
    """One A2C step on the recent crossovers. Reward (reward_mode):
    'delta'       best parent fitness - best child fitness
    'rank'        share of the current population (pop_scores) the best child beats
    'rank_novel'  'rank', but 0 when the best child computes the same as a parent (mean absolute
                  output difference below SSC_BOUNDS[0]), so semantically neutral swaps are not rewarded
    Returns the mean raw reward, or None."""
    global crossover_history

    if not crossover_history or subtree_selector is None:
        return None

    # Train on the previous generation's crossovers (and up to recent_gen_window before it)
    gens = {record['gen'] for record in crossover_history}
    latest_gen = current_gen - 1 if current_gen is not None and current_gen - 1 in gens else max(gens)
    min_gen = latest_gen - max(int(recent_gen_window), 0)
    history = [r for r in crossover_history if min_gen <= r['gen'] <= latest_gen]
    if not history:
        crossover_history.clear()
        return None
    if len(history) < max(1, int(min_history)):
        crossover_history = history[-max(1, int(min_history)) * 4:]
        return None

    rewards = []
    for record in history:
        if record['parent_scores'] is not None:
            parent_best = min(record['parent_scores'])
        else:
            parent_best = min(score_fn(p)[0] for p in record['parents'])
        child_scores = [score_fn(c)[0] for c in record['offspring']]
        child_best = min(child_scores)
        if reward_mode == 'delta' or pop_scores is None:
            rewards.append(parent_best - child_best)
            continue
        reward = float(np.mean(np.asarray(pop_scores) > child_best))
        if reward_mode == 'rank_novel':
            child = record['offspring'][int(np.argmin(child_scores))]
            with np.errstate(all='ignore'):
                out = np.nan_to_num(child.evaluate(CROSSOVER_INPUTS), nan=1e6, posinf=1e6, neginf=-1e6)
                if any(np.mean(np.abs(out - np.nan_to_num(p.evaluate(CROSSOVER_INPUTS), nan=1e6, posinf=1e6,
                                                          neginf=-1e6))) < SSC_BOUNDS[0]
                       for p in record['parents']):
                    reward = 0.0
        rewards.append(reward)
    mean_reward = float(np.mean(rewards))
    crossover_history.clear()

    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    rewards_std = rewards.std(unbiased=False)
    if rewards_std < 1e-8 or torch.isnan(rewards).any():
        return mean_reward
    rewards = (rewards - rewards.mean()) / (rewards_std + 1e-6)
    if reward_clip is not None and reward_clip > 0:
        rewards = torch.clamp(rewards, -float(reward_clip), float(reward_clip))

    # Batched forward passes over chunks of similarly sized parent pairs
    with_dists = getattr(subtree_selector, 'needs_distances', False)
    order = sorted(range(len(history)), key=lambda i: history[i]['a'][0].shape[0] * history[i]['b'][0].shape[0])
    chunks, chunk, max_a, max_b = [], [], 0, 0
    for i in order:
        la, lb = history[i]['a'][0].shape[0], history[i]['b'][0].shape[0]
        if chunk and (len(chunk) + 1) * max(max_a, la) * max(max_b, lb) > PAIR_BATCH_BUDGET:
            chunks.append(chunk)
            chunk, max_a, max_b = [], 0, 0
        chunk.append(i)
        max_a, max_b = max(max_a, la), max(max_b, lb)
    chunks.append(chunk)

    # The loss is a mean over records, so each chunk's share is backpropagated as soon as it is
    # built and its graph freed; holding every chunk's graph for one backward grows with tree bloat
    selector_optimizer.zero_grad()
    total_loss, n = 0.0, len(history)
    for chunk in chunks:
        recs = [history[i] for i in chunk]
        a = make_side([r['a'][0] for r in recs], [r['a'][1] for r in recs] if with_dists else None)
        b = make_side([r['b'][0] for r in recs], [r['b'][1] for r in recs] if with_dists else None)
        flat_logits, flat_values = pair_distribution(subtree_selector, a, b, temperature, heuristic_bias_scale)
        idx = torch.tensor([[r['a_idx'] * b.feats.shape[1] + r['b_idx']] for r in recs], device=DEVICE)
        pair_log_probs = torch.log_softmax(flat_logits, dim=1)
        pair_probs = torch.softmax(flat_logits, dim=1)
        log_probs = pair_log_probs.gather(1, idx).squeeze(1)
        entropies = -torch.sum(pair_probs * torch.log(pair_probs + 1e-10), dim=1)
        chunk_rewards = rewards[chunk]
        if flat_values is not None:
            values = flat_values.gather(1, idx).squeeze(1)
            loss = (-torch.sum(log_probs * (chunk_rewards - values.detach()))
                    + critic_weight * torch.sum((chunk_rewards - values) ** 2))
        else:
            loss = -torch.sum(log_probs * chunk_rewards)
        loss = (loss - entropy_weight * torch.sum(entropies)) / n
        loss.backward()
        total_loss += loss.item()

    if np.isfinite(total_loss):
        torch.nn.utils.clip_grad_norm_(subtree_selector.parameters(), max_norm=1.0)
        selector_optimizer.step()
    # glibc keeps freed tensor memory as a high-water mark; hand it back to the OS
    if _libc is not None:
        _libc.malloc_trim(0)
    return mean_reward


def optimize_constants(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray, 
                      method='L-BFGS-B', maxiter=50) -> Node:
    """Optimize constants in the tree using numerical optimization while keeping structure fixed."""
    # Collect all constant nodes
    const_nodes = [node for node in ind.all_nodes() if isinstance(node, Const)]
    
    if not const_nodes:
        return ind  # No constants to optimize
    
    # Extract initial values
    initial_values = np.array([node.value for node in const_nodes], dtype=float)
    
    # Define objective function (MSE)
    def objective(const_values):
        # Temporarily update constant values
        for node, val in zip(const_nodes, const_values):
            node.value = float(val)
        
        try:
            y_pred = ind.evaluate(inputs)
            if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
                return 1e10  # Large penalty for invalid predictions
            mse = float(np.mean((y - y_pred)**2))
            return mse
        except Exception:
            return 1e10
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_values,
            method=method,
            options={'maxiter': maxiter}
        )
        
        # Update constants with optimized values
        if result.success:
            for node, val in zip(const_nodes, result.x):
                node.value = float(val)
        else:
            for node, val in zip(const_nodes, initial_values):
                node.value = float(val)
    except Exception:
        for node, val in zip(const_nodes, initial_values):
            node.value = float(val)
    
    return ind


def _distribute(e):
    """Expand while letting identical factors cancel: push outer multipliers into each summand first."""
    if isinstance(e, sp.Add):
        return sp.Add(*[_distribute(a) for a in e.args])
    if isinstance(e, sp.Mul):
        sums = [a for a in e.args if isinstance(a, sp.Add)]
        if sums:
            s = sums[0]
            rest = [a for a in e.args if a is not s]
            return sp.Add(*[_distribute(sp.Mul(*rest, a)) for a in s.args])
    return sp.expand(e)


def simplify_equation(expr, data: Mapping[str, np.ndarray], y: np.ndarray, tol=0.005, min_r2=None, verbose=True):
    """Numerically simplify a sympy equation against data.

    The equation is expanded into a sum of terms, the term coefficients are refit by least squares,
    and terms are greedily dropped while R^2 stays within tol of the original (or above min_r2 if
    given). All numbers are then rounded to the fewest significant figures that keep R^2 within
    0.001 of the pruned fit. Returns (simplified_expr, r2).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)

    def r2(p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    def fit(cols):
        A = np.column_stack([np.ones(n)] + cols)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return w, A @ w

    e = sp.powsimp(_distribute(sp.sympify(expr)))
    syms = sorted(e.free_symbols, key=lambda s: s.name)
    cols = [np.asarray(data[s.name], dtype=float) for s in syms]

    def evaluate(ex):
        return np.broadcast_to(np.asarray(sp.lambdify(syms, ex, "numpy")(*cols), float), (n,))

    r2_orig = r2(evaluate(e))

    # Basis functions: each term with its numeric coefficient stripped
    mons = list(dict.fromkeys(m for m in (t.as_coeff_Mul()[1] for t in sp.Add.make_args(e)) if m.free_symbols))
    vals = [evaluate(m) for m in mons]

    # Backward elimination: drop the term whose removal hurts R^2 least, while above the floor
    floor = r2_orig - tol if min_r2 is None else min_r2
    keep = list(range(len(mons)))
    while len(keep) > 1:
        score, drop = max((r2(fit([vals[k] for k in keep if k != i])[1]), i) for i in keep)
        if score < floor:
            break
        keep.remove(drop)

    w, p = fit([vals[k] for k in keep])
    r2_pruned = r2(p)
    exact = sp.Float(w[0]) + sum(sp.Float(wi) * mons[k] for wi, k in zip(w[1:], keep))

    # Round every number (including those inside functions) to the fewest significant figures
    # that keep R^2 within 0.001 of the pruned fit
    for sig in range(3, 10):
        simp = exact.xreplace({f: sp.Float(float(f"{float(f):.{sig}g}")) for f in exact.atoms(sp.Float)})
        r2_new = r2(evaluate(simp))
        if r2_new >= r2_pruned - 0.001:
            break
    simp = simp.xreplace({f: sp.Float(f, sig) for f in simp.atoms(sp.Float)})
    if verbose:
        print(f"Simplified: {len(mons)} -> {len(keep)} terms, {sig} sig figs, R2 {r2_orig:.4f} -> {r2_new:.4f}")
    return simp, float(r2_new)


def example():
    # target: y = x^2 + 0.5 * y_var
    def target(x_var, y_var):
        return x_var**2 + 0.5 * y_var

    num_points = 150
    inputs = {
        'x': np.linspace(-1, 1, num_points),
        'y': np.linspace(0, 2, num_points),
    }
    Y = target(inputs['x'], inputs['y'])
    best, best_r2 = run_ga(
        inputs,
        Y,
        pop_size=200,
        gens=80,
        max_init_depth=3,
        max_depth=6,
        crossover_rate=0.8,
        mutation_rate=0.3,
        tournament_k=3,
        elitism=2,
        var_names=['x', 'y'],
        stopping_metric='mse',
        stopping_threshold=0.01,
        complexity_coef=5e-4,
        inclusion_coef=0.5,
        prob_const=0.35,
        const_range=(-3, 3),
        point_mutation_prob=0.1,
        selector_update_freq=3,
        temperature=0.7,
        critic_weight=0.6,
        entropy_weight=0.02,
        opt_constants=True,         # Changed parameter name
        const_opt_freq=10,
        const_opt_maxiter=30,
        verbose=True,
        seed=42,
    )

    # convert to sympy and simplify
    if best is not None:
        sym_expr = sp.simplify(best.to_sympy())
    else:
        sym_expr = None

    print("\nBest expression (tree):", best.to_string())
    print("Returned R²:", best_r2)
    print("Sympy simplified:", sym_expr)
    # use sympy-lambdified function for final MSE
    if sym_expr is not None:
        symbols = [VAR_SYMBOLS[name] for name in VAR_NAMES]
        f_sym = sp.lambdify(symbols, sym_expr, 'numpy')
        preds = np.array(f_sym(*[inputs[name] for name in VAR_NAMES]), dtype=float)
    else:
        preds = best.evaluate(inputs) if best is not None else np.zeros_like(Y, dtype=float)
    
    mse = float(np.mean((Y - preds)**2))
    # Calculate R²
    ss_res = np.sum((Y - preds)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    final_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    print("MSE:", mse)
    print("R²:", final_r2)
    
    # show a few example points
    step = max(1, len(Y) // 5)
    for idx in range(0, len(Y), step):
        sample_vals = ", ".join(f"{name}={inputs[name][idx]:.2f}" for name in VAR_NAMES)
        print(f"{sample_vals} y_true={Y[idx]:.4g} y_pred={preds[idx]:.4g}")

def prepare_inputs(x, var_names=None) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if isinstance(x, Mapping):
        data = {key: np.asarray(val, dtype=float).reshape(-1) for key, val in x.items()}
        inferred = list(data.keys())
        if var_names is None:
            var_names = inferred
        else:
            missing = set(var_names) - data.keys()
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Missing inputs for variables: {missing_list}")
            data = {name: data[name] for name in var_names}
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            if var_names is None:
                var_names = ['x']
            elif len(var_names) != 1:
                raise ValueError("var_names length must match x shape.")
            data = {var_names[0]: arr}
        elif arr.ndim == 2:
            n_vars = arr.shape[1]
            if var_names is None:
                var_names = [f'x{i}' for i in range(n_vars)]
            elif len(var_names) != n_vars:
                raise ValueError("var_names length must match number of columns in x.")
            data = {name: arr[:, idx] for idx, name in enumerate(var_names)}
        else:
            raise ValueError("x must be 1D, 2D, or a mapping of arrays.")
    lengths = {len(col) for col in data.values()}
    if len(lengths) != 1:
        raise ValueError("All variable arrays must share the same length.")
    return data, list(var_names)

def detect_inert_variables(
    ind: Node,
    inputs: Mapping[str, np.ndarray],
    base_pred: np.ndarray,
    atol=1e-5,
    rtol=1e-4,
    candidate_vars: set[str] | None = None,
) -> set[str]:
    if candidate_vars is None:
        var_names = [name for name in inputs.keys()]
    else:
        var_names = [name for name in candidate_vars if name in inputs]

    inert = set()
    if not var_names:
        return inert

    perturbed = dict(inputs)
    rolled_inputs = {name: np.roll(inputs[name], 1) for name in var_names}

    for name in var_names:
        original = inputs[name]
        perturbed[name] = rolled_inputs[name]
        try:
            alt_pred = ind.evaluate(perturbed)
        except Exception:
            inert.add(name)
            perturbed[name] = original
            continue
        if np.allclose(base_pred, alt_pred, atol=atol, rtol=rtol):
            inert.add(name)
        perturbed[name] = original
    return inert


def _timed_simplify_worker(expr, queue):
    try:
        queue.put(sp.simplify(expr))
    except Exception as exc:
        queue.put(exc)


def timed_sympy_simplify(expr: sp.Expr, timeout: float | None):
    if timeout is None or timeout <= 0:
        return None
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_timed_simplify_worker, args=(expr, queue))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    if queue.empty():
        return None
    result = queue.get()
    if isinstance(result, Exception):
        return None
    return result


def sympy_to_node(expr: sp.Expr) -> Node:
    if expr.is_Number:
        return Const(float(expr))
    if expr.is_Symbol:
        name = str(expr)
        if name not in VAR_SYMBOLS:
            raise ValueError(f"Unknown symbol '{name}' in simplified expression.")
        return Var(name)
    sympy_func_to_name = {func: name for name, func in SYMPY_UNARY.items()}
    if expr.func in sympy_func_to_name:
        if len(expr.args) != 1:
            raise ValueError("Unsupported arity for unary function.")
        return Unary(sympy_func_to_name[expr.func], sympy_to_node(expr.args[0]))
    if expr.func == sp.Add:
        args = list(expr.args)
        node = sympy_to_node(args[0])
        for arg in args[1:]:
            node = Binary('add', node, sympy_to_node(arg))
        return node
    if expr.func == sp.Mul:
        numerators, denominators = [], []
        for arg in expr.args:
            if isinstance(arg, sp.Pow) and arg.exp == -1:
                denominators.append(arg.base)
            else:
                numerators.append(arg)
        node = sympy_to_node(numerators[0]) if numerators else Const(1.0)
        for arg in numerators[1:]:
            node = Binary('mul', node, sympy_to_node(arg))
        for arg in denominators:
            node = Binary('div', node, sympy_to_node(arg))
        return node
    if isinstance(expr, sp.Pow):
        if 'pow' in ALL_BINARY:
            return Binary('pow', sympy_to_node(expr.base), sympy_to_node(expr.exp))
        if expr.exp == -1:
            return Binary('div', Const(1.0), sympy_to_node(expr.base))
        if expr.exp == 2:
            base_node = sympy_to_node(expr.base)
            return Binary('mul', base_node.clone(), base_node.clone())
        raise ValueError("Unsupported exponent encountered during simplification.")
    raise ValueError("Unsupported sympy construct during simplification.")


def simplify_node_with_timeout(node: Node, timeout: float | None) -> Node:
    if timeout is None or timeout <= 0:
        return node
    try:
        simplified_expr = timed_sympy_simplify(node.to_sympy(), timeout)
    except Exception:
        return node
    if simplified_expr is None:
        return node
    try:
        return sympy_to_node(simplified_expr)
    except ValueError:
        return node
