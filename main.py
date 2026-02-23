# Genetic algorithm for symbolic regression (simple, self-contained)

import random
import copy
import operator
import time
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_global_seed(DEFAULT_SEED)

# Set device for PyTorch operations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _print_device_info():
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

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
    'log': sp.log,
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

    def clone(self):
        return copy.deepcopy(self)

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


def subtree_crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    a_clone = a.clone()
    b_clone = b.clone()
    a_nodes = collect_nodes_with_parents(a_clone)
    b_nodes = collect_nodes_with_parents(b_clone)
    # avoid replacing the entire tree in both directions maybe
    node_a, parent_a, attr_a = random.choice(a_nodes)
    node_b, parent_b, attr_b = random.choice(b_nodes)
    # Swap node_a and node_b in clones
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
    return new_a, new_b


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
def tournament_select(pop: List[Node], scores: List[float], k=3, timeout=10.0) -> Node:
    if not pop:
        raise ValueError("Population must contain at least one individual.")
    deadline = time.perf_counter() + timeout
    sample_size = min(k, len(pop))
    while True:
        try:
            selected_idx = random.sample(range(len(pop)), sample_size)
            best = min(selected_idx, key=lambda i: scores[i])
            return pop[best].clone()
        except Exception:
            if time.perf_counter() >= deadline:
                return random.choice(pop).clone()


def pareto_objectives(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray,
                      inclusion_coef=1.0) -> Tuple[float, float]:
    """Return (error_with_inclusion, complexity) objectives for Pareto sorting."""
    try:
        y_pred = ind.evaluate(inputs)
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            return 1e6 + ind.size(), float(ind.size())
        mse = float(np.mean((y - y_pred) ** 2))
    except Exception:
        return 1e6 + ind.size(), float(ind.size())
    inert_vars = detect_inert_variables(ind, inputs, y_pred)
    missing_vars = (set(inputs.keys()) - gather_used_var_names(ind)) | inert_vars
    inclusion_penalty = inclusion_coef * len(missing_vars)
    return mse + inclusion_penalty, float(ind.size())


def _pareto_dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def pareto_rank_and_crowding(objectives: List[Tuple[float, float]]):
    """Compute Pareto rank and crowding distance (NSGA-II style)."""
    n = len(objectives)
    if n == 0:
        return [], []
    domination_sets = [set() for _ in range(n)]
    domination_counts = [0] * n
    fronts = []

    for i in range(n):
        for j in range(i + 1, n):
            if _pareto_dominates(objectives[i], objectives[j]):
                domination_sets[i].add(j)
                domination_counts[j] += 1
            elif _pareto_dominates(objectives[j], objectives[i]):
                domination_sets[j].add(i)
                domination_counts[i] += 1

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
        for obj_idx in range(2):
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


def tournament_select_pareto(pop: List[Node], ranks: List[int], crowding: List[float], scores: List[float],
                             k=3, timeout=10.0) -> Node:
    if not pop:
        raise ValueError("Population must contain at least one individual.")
    deadline = time.perf_counter() + timeout
    sample_size = min(k, len(pop))
    while True:
        try:
            selected_idx = random.sample(range(len(pop)), sample_size)
            best = min(
                selected_idx,
                key=lambda i: (ranks[i], -crowding[i], scores[i])
            )
            return pop[best].clone()
        except Exception:
            if time.perf_counter() >= deadline:
                return random.choice(pop).clone()


# Neural network that learns to select good subtree crossover points using A2C.
class SubtreeSelector(nn.Module):
    def __init__(self, feature_dim=10, hidden_dim=64, use_transformer=False, num_heads=4, num_layers=2,
                 mlp_layers=3, mlp_activation='relu', mlp_dropout=0.0, use_critic=True):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_critic = use_critic
        
        if use_transformer:
            # Transformer-based feature extractor
            self.feature_projection = nn.Linear(feature_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Actor head (policy)
            self.actor = nn.Linear(hidden_dim, 1)
            # Critic head (value function) - only if enabled
            if self.use_critic:
                self.critic = nn.Linear(hidden_dim, 1)
        else:
            # Customizable MLP-based feature extractor
            activation_map = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'leaky_relu': nn.LeakyReLU,
                'elu': nn.ELU,
                'gelu': nn.GELU,
            }
            
            if mlp_activation not in activation_map:
                raise ValueError(f"Unsupported activation: {mlp_activation}. Choose from {list(activation_map.keys())}")
            
            activation_fn = activation_map[mlp_activation]
            
            # Build shared MLP layers
            shared_layers = []
            input_size = feature_dim
            
            for i in range(mlp_layers - 1):
                shared_layers.append(nn.Linear(input_size, hidden_dim))
                shared_layers.append(activation_fn())
                if mlp_dropout > 0:
                    shared_layers.append(nn.Dropout(mlp_dropout))
                input_size = hidden_dim
            
            self.shared_net = nn.Sequential(*shared_layers)
            
            # Actor head (policy) - outputs action logits
            self.actor = nn.Linear(hidden_dim, 1)
            
            # Critic head (value function) - only if enabled
            if self.use_critic:
                self.critic = nn.Linear(hidden_dim, 1)
        
        # Move model to device
        self.to(DEVICE)
    
    def forward(self, features, return_value=False):
        """Returns action logits and optionally value estimates.
        
        Args:
            features: (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
            return_value: If True, also return value estimates from critic
        
        Returns:
            If return_value=False: action_logits
            If return_value=True: (action_logits, values) or (action_logits, None) if no critic
        """
        if self.use_transformer:
            # If 2D, add sequence dimension
            if features.dim() == 2:
                features = features.unsqueeze(0)  # (1, seq_len, feature_dim)
            
            # Project features to transformer dimension
            x = self.feature_projection(features)  # (batch, seq_len, hidden_dim)
            
            # Apply transformer encoder
            x = self.transformer(x)  # (batch, seq_len, hidden_dim)
            
            # Actor and critic outputs
            action_logits = self.actor(x).squeeze(-1)  # (batch, seq_len)
            
            # Remove batch dimension if it was added
            if action_logits.dim() == 2 and action_logits.size(0) == 1:
                action_logits = action_logits.squeeze(0)
            
            if return_value:
                if self.use_critic:
                    values = self.critic(x).squeeze(-1)  # (batch, seq_len)
                    if values.dim() == 2 and values.size(0) == 1:
                        values = values.squeeze(0)
                else:
                    values = None
                return action_logits, values
            else:
                return action_logits
        else:
            # MLP path
            shared_features = self.shared_net(features)
            action_logits = self.actor(shared_features).squeeze(-1)
            
            if return_value:
                if self.use_critic:
                    values = self.critic(shared_features).squeeze(-1)
                else:
                    values = None
                return action_logits, values
            else:
                return action_logits


def extract_subtree_features(node: Node, parent: Node, depth_from_root: int, 
                            tree_size: int, tree_depth: int) -> torch.Tensor:
    """Extract features for a subtree rooted at 'node'."""
    features = []
    
    # Node type features (one-hot-ish)
    features.append(1.0 if isinstance(node, Const) else 0.0)
    features.append(1.0 if isinstance(node, Var) else 0.0)
    features.append(1.0 if isinstance(node, Unary) else 0.0)
    features.append(1.0 if isinstance(node, Binary) else 0.0)
    
    # Structural features
    features.append(node.size() / max(tree_size, 1))  # Relative size
    features.append(node.depth() / max(tree_depth, 1))  # Relative depth
    features.append(depth_from_root / max(tree_depth, 1))  # Position in tree
    
    # Parent features
    features.append(1.0 if parent is None else 0.0)  # Is root
    features.append(1.0 if isinstance(parent, Unary) else 0.0)
    features.append(1.0 if isinstance(parent, Binary) else 0.0)

    # Specific operator features
    unary_one_hot = [0.0] * len(ALL_UNARY)
    if isinstance(node, Unary):
        try:
            unary_one_hot[ALL_UNARY.index(node.func_name)] = 1.0
        except ValueError:
            pass
    features.extend(unary_one_hot)

    binary_one_hot = [0.0] * len(ALL_BINARY)
    if isinstance(node, Binary):
        try:
            binary_one_hot[ALL_BINARY.index(node.func_name)] = 1.0
        except ValueError:
            pass
    features.extend(binary_one_hot)
    
    return torch.tensor(features, dtype=torch.float32, device=DEVICE)

def collect_nodes_with_features(node: Node, parent=None, attr_name=None, depth=0):
    """Collect nodes with parent info and features."""
    tree_size = node.size()
    tree_depth = node.depth()
    
    def recurse(n, p, attr, d):
        features = extract_subtree_features(n, p, d, tree_size, tree_depth)
        res = [(n, p, attr, features)]
        if isinstance(n, Unary):
            res += recurse(n.child, n, 'child', d + 1)
        elif isinstance(n, Binary):
            res += recurse(n.left, n, 'left', d + 1)
            res += recurse(n.right, n, 'right', d + 1)
        return res
    
    return recurse(node, parent, attr_name, depth)

# Global selector (will be initialized in run_ga)
subtree_selector = None
selector_optimizer = None
crossover_history = []
selector_feature_dim = None

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
    selector_lr=0.001,
    use_transformer=False,
    transformer_heads=4,
    transformer_layers=2,
    mlp_layers=3,
    mlp_hidden_dim=64,
    mlp_activation='relu',
    mlp_dropout=0.0,
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
    temperature=0.5,
    critic_weight=0.5,
    entropy_weight=0.01,
    use_pareto=False,
    opt_constants=False,  # Renamed to avoid shadowing the function
    const_opt_freq=5,
    const_opt_method='L-BFGS-B',
    const_opt_maxiter=50,
    verbose=False,
    print_freq=1,
    seed=DEFAULT_SEED,
    metrics_callback=None,
    allow_pow=False,
):
    global subtree_selector, selector_optimizer, crossover_history, selector_feature_dim
    
    if seed is None:
        seed = DEFAULT_SEED
    set_global_seed(seed)
    
    inputs, resolved_names = prepare_inputs(x, var_names)
    set_variables(resolved_names)
    
    # Configure unary functions
    global UNARY_FUNCS, SYMPY_UNARY, ALL_UNARY
    if unary_funcs is not None:
        if not isinstance(unary_funcs, dict):
            raise ValueError("unary_funcs must be a dict mapping names to (numpy_func, sympy_func) tuples")
        UNARY_FUNCS = {}
        SYMPY_UNARY = {}
        for name, funcs in unary_funcs.items():
            if isinstance(funcs, tuple) and len(funcs) == 2:
                UNARY_FUNCS[name] = funcs[0]
                SYMPY_UNARY[name] = funcs[1]
            else:
                raise ValueError(f"Each unary function must be a (numpy_func, sympy_func) tuple, got {funcs}")
        ALL_UNARY = list(UNARY_FUNCS.keys())
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

    # Reset crossover history whenever we start a new GA run so feature sizes stay consistent
    crossover_history = []
    selector_feature_dim = get_selector_feature_dim()
    
    init_max_depth = max(1, max_init_depth)
    pop = [
        random_tree(
            max_depth=random.randint(1, init_max_depth),
            prob_const=prob_const,
            const_range=const_range,
        )
        for _ in range(pop_size)
    ]
    if simplify_new_genomes:
        pop = [simplify_node_with_timeout(ind, simplify_timeout) for ind in pop]
    
    # Initialize subtree selector
    if use_learned_crossover:
        subtree_selector = SubtreeSelector(
            feature_dim=get_selector_feature_dim(),
            hidden_dim=mlp_hidden_dim,
            use_transformer=use_transformer,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            mlp_layers=mlp_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            use_critic=use_critic,  # Pass the critic flag
        )
        selector_optimizer = optim.Adam(subtree_selector.parameters(), lr=selector_lr)
        if verbose:
            critic_mode = "A2C (with critic)" if use_critic else "Policy Gradient (no critic)"
            if use_transformer:
                print(f"Using learned subtree crossover with Transformer encoder ({transformer_layers} layers, {transformer_heads} heads, {critic_mode})")
            else:
                print(f"Using learned subtree crossover with MLP ({mlp_layers} layers, {mlp_hidden_dim} hidden, {mlp_activation} activation, {mlp_dropout} dropout, {critic_mode})")
    else:
        subtree_selector = None
        if verbose:
            print("Using random subtree crossover")
    
    # Validate stopping criterion
    if stopping_threshold is not None:
        if stopping_metric not in ['mse', 'r2', None]:
            raise ValueError("stopping_metric must be 'mse', 'r2', or None")
        if stopping_metric == 'r2' and (stopping_threshold < 0 or stopping_threshold > 1):
            print(f"Warning: R² stopping threshold {stopping_threshold} is outside typical range [0, 1]")
        if verbose:
            print(f"Stopping criterion: {stopping_metric} <= {stopping_threshold}" if stopping_metric == 'mse' else f"Stopping criterion: {stopping_metric} >= {stopping_threshold}")
    
    y = np.asarray(y, dtype=float).reshape(-1)
    n_samples = len(next(iter(inputs.values())))
    if y.shape[0] != n_samples:
        raise ValueError("y must have the same length as the input samples.")

    def evaluate_r2(individual):
        try:
            preds = individual.evaluate(inputs)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return None
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else None
        except Exception:
            return None

    best = None
    best_score = float('inf')
    best_r2 = None

    gen_iterator = tqdm(range(gens), desc="GA generations", unit="gen")  # Changed from trange
    for gen in gen_iterator:
        scores = [fitness(ind, inputs, y, complexity_coef, inclusion_coef) for ind in pop]
        if use_pareto:
            pareto_objectives_list = [
                pareto_objectives(ind, inputs, y, inclusion_coef)
                for ind in pop
            ]
            pareto_ranks, pareto_crowding = pareto_rank_and_crowding(pareto_objectives_list)
        else:
            pareto_ranks, pareto_crowding = None, None
        
        # Update selector every few generations
        if use_learned_crossover and gen > 0 and gen % selector_update_freq == 0:
            update_selector(
                pop,
                scores,
                inputs,
                y,
                complexity_coef,
                inclusion_coef,
                critic_weight=critic_weight,
                entropy_weight=entropy_weight,
                current_gen=gen,
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
                        current_mse = float(np.mean((y - y_pred)**2))
                        if current_mse <= stopping_threshold:
                            if verbose:
                                print(f"\nStopping criterion reached at generation {gen+1}")
                                print(f"MSE: {current_mse:.6g} <= {stopping_threshold}")
                            break
                    elif stopping_metric == 'r2':
                        ss_res = np.sum((y - y_pred)**2)
                        ss_tot = np.sum((y - np.mean(y))**2)
                        current_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                        if current_r2 >= stopping_threshold:
                            if verbose:
                                print(f"\nStopping criterion reached at generation {gen+1}")
                                print(f"R²: {current_r2:.6g} >= {stopping_threshold}")
                            break
            except Exception:
                pass  # Continue if evaluation fails
        
        next_pop = []
        # elitism
        if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
            sorted_idx = sorted(
                range(len(pop)),
                key=lambda i: (pareto_ranks[i], -pareto_crowding[i], scores[i])
            )
        else:
            sorted_idx = sorted(range(len(pop)), key=lambda i: scores[i])
        # Optimize constants only for elites (optional, periodic)
        optimize_elites = opt_constants and gen > 0 and gen % const_opt_freq == 0
        if optimize_elites and verbose:
            print(f"\nOptimizing constants for elites at generation {gen+1}...")
        for i in range(elitism):
            elite = pop[sorted_idx[i]].clone()
            if optimize_elites:
                elite = optimize_constants(
                    elite,
                    inputs,
                    y,
                    method=const_opt_method,
                    maxiter=const_opt_maxiter,
                )
            if simplify_new_genomes:
                elite = simplify_node_with_timeout(elite, simplify_timeout)
            next_pop.append(elite)
        # produce rest
        while len(next_pop) < pop_size:
            if random.random() < crossover_rate:
                try:
                    if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
                        parent1 = tournament_select_pareto(pop, pareto_ranks, pareto_crowding, scores, tournament_k)
                    else:
                        parent1 = tournament_select(pop, scores, tournament_k)
                except RecursionError:
                    parent1 = random.choice(pop).clone()
                try:
                    if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
                        parent2 = tournament_select_pareto(pop, pareto_ranks, pareto_crowding, scores, tournament_k)
                    else:
                        parent2 = tournament_select(pop, scores, tournament_k)
                except RecursionError:
                    parent2 = random.choice(pop).clone()
                child1, child2, history_idx = learned_subtree_crossover(
                    parent1,
                    parent2,
                    use_learned=use_learned_crossover,
                    temperature=temperature,
                    record_gen=gen,
                )
                # optionally mutate children
                if random.random() < mutation_rate:
                    child1 = subtree_mutation(child1, max_depth, prob_const, const_range)
                if random.random() < mutation_rate:
                    child2 = subtree_mutation(child2, max_depth, prob_const, const_range)
                if simplify_new_genomes:
                    child1 = simplify_node_with_timeout(child1, simplify_timeout)
                    child2 = simplify_node_with_timeout(child2, simplify_timeout)
                if history_idx is not None and 0 <= history_idx < len(crossover_history):
                    crossover_history[history_idx]['offspring'] = (
                        child1.clone(),
                        child2.clone(),
                    )
                next_pop.append(child1)
                if len(next_pop) < pop_size:
                    next_pop.append(child2)
            else:
                try:
                    if use_pareto and pareto_ranks is not None and pareto_crowding is not None:
                        parent = tournament_select_pareto(pop, pareto_ranks, pareto_crowding, scores, tournament_k)
                    else:
                        parent = tournament_select(pop, scores, tournament_k)
                except RecursionError:
                    parent = random.choice(pop).clone()
                child = parent
                if random.random() < mutation_rate:
                    child = subtree_mutation(child, max_depth, prob_const, const_range)
                # small point mutations
                child = point_mutation(child, prob_change=point_mutation_prob)
                if simplify_new_genomes:
                    child = simplify_node_with_timeout(child, simplify_timeout)
                next_pop.append(child)
        pop = next_pop

        if verbose and ((gen + 1) % print_freq == 0 or gen == 0):
            # Calculate R² for the best individual
            try:
                y_pred_best = best.evaluate(inputs)
                if not (np.any(np.isinf(y_pred_best)) or np.any(np.isnan(y_pred_best))):
                    ss_res = np.sum((y - y_pred_best)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2_best = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g} R²: {r2_best:.6f} expr: {best.to_string()}")
                else:
                    print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g} expr: {best.to_string()}")
            except Exception:
                print(f"Gen {gen+1:3d} best MSE+penalty: {best_score:.6g} expr: {best.to_string()}")

        if metrics_callback is not None and best is not None:
            try:
                y_pred = np.asarray(best.evaluate(inputs), dtype=float)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    raise ValueError("Invalid predictions for metrics callback.")
                ss_res = float(np.sum((y - y_pred) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
            except Exception:
                r2 = float("nan")
            metrics_callback(gen, float(r2))

    if best is not None and best_r2 is None:
        best_r2 = evaluate_r2(best)

    return best, best_r2


# Random tree generation with configurable constants
def random_tree(max_depth: int, prob_const=0.3, const_range=(-5, 5)) -> Node:
    if max_depth == 0:
        if random.random() < prob_const:
            return Const(random.uniform(*const_range))
        return Var()

    # Build choices dynamically so empty operator sets don't crash.
    choices = ["const", "var"]
    weights = [0.2, 0.2]
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
                return Var()
            if choice == "unary":
                func = random.choice(ALL_UNARY)
                return Unary(func, random_tree(max_depth - 1, prob_const, const_range))
            func = random.choice(ALL_BINARY)
            return Binary(
                func,
                random_tree(max_depth - 1, prob_const, const_range),
                random_tree(max_depth - 1, prob_const, const_range),
            )

    # Fallback (should not hit)
    return Var()


# Fitness: mean squared error + complexity penalty
def fitness(ind: Node, inputs: Mapping[str, np.ndarray], y: np.ndarray, 
            complexity_coef=1e-3, inclusion_coef=1.0) -> float:
    try:
        y_pred = ind.evaluate(inputs)
        if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)):
            return 1e6 + ind.size()
        mse = float(np.mean((y - y_pred)**2))
    except Exception:
        return 1e6 + ind.size()
    inert_vars = detect_inert_variables(ind, inputs, y_pred)
    missing_vars = (set(inputs.keys()) - gather_used_var_names(ind)) | inert_vars
    inclusion_penalty = inclusion_coef * len(missing_vars)
    return mse + complexity_coef * ind.size() + inclusion_penalty


def subtree_mutation(ind: Node, max_depth: int, prob_const=0.3, const_range=(-5, 5)) -> Node:
    ind_clone = ind.clone()
    nodes = collect_nodes_with_parents(ind_clone)
    node, parent, attr = random.choice(nodes)
    new_sub = random_tree(max_depth=random.randint(0, max_depth), prob_const=prob_const, const_range=const_range)
    if parent is None:
        return new_sub
    else:
        replace_child(parent, attr, new_sub)
        return ind_clone


def learned_subtree_crossover(
    a: Node,
    b: Node,
    use_learned=True,
    temperature=0.5,
    record_gen: int | None = None,
) -> Tuple[Node, Node, int | None]:
    """Crossover using learned subtree selection with custom temperature."""
    global subtree_selector, selector_feature_dim, crossover_history
    
    a_clone = a.clone()
    b_clone = b.clone()
    a_nodes = collect_nodes_with_features(a_clone)
    b_nodes = collect_nodes_with_features(b_clone)
    
    a_features = None
    b_features = None
    
    expected_dim = get_selector_feature_dim()
    if selector_feature_dim is not None and expected_dim != selector_feature_dim:
        use_learned = False
        crossover_history.clear()

    if use_learned and subtree_selector is not None:
        # Use neural network to select crossover points
        with torch.no_grad():
            a_features = torch.stack([feat for _, _, _, feat in a_nodes])
            b_features = torch.stack([feat for _, _, _, feat in b_nodes])
            
            # Get action logits and values from A2C
            a_scores, a_values = subtree_selector(a_features, return_value=True)
            b_scores, b_values = subtree_selector(b_features, return_value=True)
            
            # Clip scores to prevent overflow in softmax
            a_scores = torch.clamp(a_scores, -10, 10)
            b_scores = torch.clamp(b_scores, -10, 10)
            
            # Sample from softmax distribution with custom temperature
            safe_temp = max(float(temperature), 1e-6)
            a_probs = torch.softmax(a_scores / safe_temp, dim=0)
            b_probs = torch.softmax(b_scores / safe_temp, dim=0)
            
            # Additional safety check
            if torch.isnan(a_probs).any() or torch.isinf(a_probs).any() or (a_probs < 0).any():
                a_idx = random.randint(0, len(a_nodes) - 1)
            else:
                a_idx = torch.multinomial(a_probs, 1).item()
            
            if torch.isnan(b_probs).any() or torch.isinf(b_probs).any() or (b_probs < 0).any():
                b_idx = random.randint(0, len(b_nodes) - 1)
            else:
                b_idx = torch.multinomial(b_probs, 1).item()
    else:
        # Random selection (baseline)
        a_idx = random.randint(0, len(a_nodes) - 1)
        b_idx = random.randint(0, len(b_nodes) - 1)
    
    node_a, parent_a, attr_a, feat_a = a_nodes[a_idx]
    node_b, parent_b, attr_b, feat_b = b_nodes[b_idx]
    
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
    if use_learned and a_features is not None and b_features is not None:
        crossover_history.append({
            'a_features_all': a_features.cpu(),
            'b_features_all': b_features.cpu(),
            'a_idx': a_idx,
            'b_idx': b_idx,
            'offspring': (new_a.clone(), new_b.clone()),
            'gen': record_gen,
        })
        history_idx = len(crossover_history) - 1
    
    return new_a, new_b, history_idx


def update_selector(
    pop: List[Node],
    scores: List[float],
    inputs,
    y,
    complexity_coef=1e-3,
    inclusion_coef=1.0,
    critic_weight=0.5,
    entropy_weight=0.01,
    current_gen: int | None = None,
):
    global subtree_selector, selector_optimizer, crossover_history
    
    if not crossover_history or subtree_selector is None:
        return
    
    # Filter history to only include records matching the current feature dimension
    expected_dim = get_selector_feature_dim()
    latest_gen = None
    available_gens = []
    for record in crossover_history:
        if 'gen' in record and record['gen'] is not None:
            available_gens.append(record['gen'])

    if available_gens:
        latest_gen = max(available_gens)
        if current_gen is not None:
            desired_gen = current_gen - 1
            if desired_gen in available_gens:
                latest_gen = desired_gen

    filtered_history = []
    for record in crossover_history:
        if record['a_features_all'].dim() != 2 or record['b_features_all'].dim() != 2:
            continue
        if record['a_features_all'].size(-1) != expected_dim:
            continue
        if record['b_features_all'].size(-1) != expected_dim:
            continue
        if latest_gen is not None and record.get('gen') != latest_gen:
            continue
        filtered_history.append(record)
    if not filtered_history:
        crossover_history.clear()
        return

    # Evaluate offspring fitness (rewards)
    rewards = []
    avg_fitness = sum(scores) / len(scores) if scores else 0.0

    for record in filtered_history:
        child1, child2 = record['offspring']
        fitness1 = fitness(child1, inputs, y, complexity_coef, inclusion_coef)
        fitness2 = fitness(child2, inputs, y, complexity_coef, inclusion_coef)
        reward = avg_fitness - min(fitness1, fitness2)
        rewards.append(reward)

    if not rewards:
        crossover_history.clear()
        return

    # Convert to tensors
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

    # Skip update if rewards are all the same or invalid
    if rewards_tensor.std() < 1e-8 or torch.isnan(rewards_tensor).any():
        crossover_history.clear()
        return

    selected_log_probs = []
    selected_values = []
    entropies = []

    for record in filtered_history:
        a_features = record['a_features_all'].to(DEVICE)
        b_features = record['b_features_all'].to(DEVICE)
        a_idx = record['a_idx']
        b_idx = record['b_idx']

        action_logits_a, values_a = subtree_selector(a_features, return_value=True)
        action_logits_b, values_b = subtree_selector(b_features, return_value=True)

        log_probs_a = torch.log_softmax(action_logits_a, dim=0)
        log_probs_b = torch.log_softmax(action_logits_b, dim=0)
        selected_log_probs.append(log_probs_a[a_idx] + log_probs_b[b_idx])

        probs_a = torch.softmax(action_logits_a, dim=0)
        probs_b = torch.softmax(action_logits_b, dim=0)
        entropy_a = -torch.sum(probs_a * torch.log(probs_a + 1e-10))
        entropy_b = -torch.sum(probs_b * torch.log(probs_b + 1e-10))
        entropies.append((entropy_a + entropy_b) / 2.0)

        if subtree_selector.use_critic and values_a is not None and values_b is not None:
            selected_values.append((values_a[a_idx] + values_b[b_idx]) / 2.0)

    selected_log_probs = torch.stack(selected_log_probs)

    if subtree_selector.use_critic and selected_values:
        selected_values = torch.stack(selected_values)
        advantages = rewards_tensor - selected_values.detach()
        actor_loss = -torch.mean(selected_log_probs * advantages)
        critic_loss = torch.mean((rewards_tensor - selected_values) ** 2)
        critic_weight_local = critic_weight
    else:
        actor_loss = -torch.mean(selected_log_probs * rewards_tensor)
        critic_loss = torch.tensor(0.0, device=DEVICE)
        critic_weight_local = 0.0

    entropy_weight_local = entropy_weight
    entropy_loss = -torch.mean(torch.stack(entropies)) if entropies else torch.tensor(0.0, device=DEVICE)

    total_loss = actor_loss + critic_weight_local * critic_loss + entropy_weight_local * entropy_loss
    
    # Backpropagation
    if not torch.isnan(total_loss) and not torch.isinf(total_loss):
        selector_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(subtree_selector.parameters(), max_norm=1.0)
        selector_optimizer.step()
    
    # Clear history
    crossover_history.clear()


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
            options={'maxiter': maxiter, 'disp': False}
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
        use_transformer=True,
        transformer_heads=4,
        transformer_layers=2,
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

def detect_inert_variables(ind: Node, inputs: Mapping[str, np.ndarray], base_pred: np.ndarray,
                           atol=1e-5, rtol=1e-4) -> set[str]:
    inert = set()
    for name in inputs:
        perturbed = {k: v.copy() for k, v in inputs.items()}
        perturbed[name] = np.roll(perturbed[name], 1)  # cheap permutation
        try:
            alt_pred = ind.evaluate(perturbed)
        except Exception:
            inert.add(name)
            continue
        if np.allclose(base_pred, alt_pred, atol=atol, rtol=rtol):
            inert.add(name)
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


if __name__ == "__main__":
    _print_device_info()