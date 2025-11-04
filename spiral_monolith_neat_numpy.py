"""
Spiral NEAT Implementation with Shared Memory Support
NeuroEvolution of Augmenting Topologies with backpropagation fitness
"""
import os
import numpy as np
from typing import Callable, List, Optional, Tuple
import matplotlib
import warnings
import pickle as _pickle
import json as _json

try:  # Python 3.8+
    from multiprocessing import shared_memory as _shm
except Exception:
    _shm = None

# === Safety & Runtime Preamble ===============================================
matplotlib.use('Agg')  # non-interactive backend
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch_dtype.*deprecated.*Use `dtype`.*")

def _is_picklable(obj) -> bool:
    """Test if an object can be pickled."""
    try:
        _pickle.dumps(obj)
        return True
    except Exception:
        return False

# === Shared-memory datasets (for process-parallel, zero-copy) =================
_SHM_LOCAL = {}   # parent-owned SharedMemory objects (for cleanup)
_SHM_META  = {}   # {label -> {'name','shape','dtype','readonly'}}
_SHM_CACHE = {}   # worker-attached numpy views

def shm_register_dataset(label: str, arr: "np.ndarray", readonly: bool = True) -> dict:
    """Create shared memory for arr (parent), return metadata dict."""
    if _shm is None:
        raise RuntimeError("shared_memory is unavailable on this Python.")
    arr = np.asarray(arr)
    size = int(arr.nbytes)
    # unique name
    name = f"sm_{label}_{np.random.randint(1, 1<<30):08x}"
    shm = _shm.SharedMemory(create=True, size=size, name=name)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    _SHM_LOCAL[label] = (shm, arr.shape, str(arr.dtype), bool(readonly))
    meta = {"name": name, "shape": tuple(arr.shape), "dtype": str(arr.dtype), "readonly": bool(readonly)}
    _SHM_META[label] = meta
    return meta

def shm_set_worker_meta(meta: dict | None):
    """Install metadata in worker; views are attached lazily on demand."""
    global _SHM_META, _SHM_CACHE
    _SHM_META = dict(meta or {})
    _SHM_CACHE = {}

def get_shared_dataset(label: str) -> "np.ndarray":
    """Worker-side: return cached numpy view to shared dataset by label."""
    if label in _SHM_CACHE:
        return _SHM_CACHE[label][0]  # Return array, not tuple
    meta = _SHM_META.get(label)
    if not meta:
        raise KeyError(f"Shared dataset '{label}' not found.")
    if _shm is None:
        raise RuntimeError("shared_memory is unavailable in worker.")
    shm = _shm.SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    if bool(meta.get("readonly", True)):
        try:
            arr.setflags(write=False)
        except Exception:
            pass
    # Cache both the SharedMemory object and the array to prevent GC
    _SHM_CACHE[label] = (arr, shm)
    return arr

def shm_release_all():
    """Parent-side: close & unlink all owned segments."""
    if not _SHM_LOCAL:
        return
    for _label, (shm, _shape, _dtype, _ro) in list(_SHM_LOCAL.items()):
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
    _SHM_LOCAL.clear()

def shm_cleanup_worker_cache():
    """Worker-side: close cached SharedMemory objects."""
    global _SHM_CACHE
    for _label, (arr, shm) in list(_SHM_CACHE.items()):
        try:
            shm.close()
        except Exception:
            pass
    _SHM_CACHE.clear()

def _proc_init_worker(meta: dict | None = None):
    """ProcessPool initializer: cap BLAS threads and install SHM metadata."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if meta:
        try:
            shm_set_worker_meta(meta)
        except Exception:
            pass


# === Genome and supporting structures ========================================
class NodeGene:
    """Represents a node in the neural network."""
    def __init__(self, node_id: int, node_type: str):
        self.id = node_id
        self.type = node_type  # 'input', 'hidden', 'output'
        self.bias = 0.0
        
class ConnectionGene:
    """Represents a connection between nodes."""
    def __init__(self, in_node: int, out_node: int, weight: float, enabled: bool = True, innovation: int = 0):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class Genome:
    """NEAT genome representation."""
    def __init__(self, genome_id: int = 0):
        self.id = genome_id
        self.nodes = {}  # node_id -> NodeGene
        self.connections = {}  # innovation -> ConnectionGene
        self.fitness = 0.0
        self.sex = 'hermaphrodite'
        self.regen = False
        self.regen_mode = 'split'
        self.hybrid_scale = 1.0
        
    def copy(self):
        """Create a deep copy of the genome."""
        new_genome = Genome(self.id)
        new_genome.nodes = {k: v for k, v in self.nodes.items()}
        new_genome.connections = {k: v for k, v in self.connections.items()}
        new_genome.fitness = self.fitness
        new_genome.sex = self.sex
        new_genome.regen = self.regen
        new_genome.regen_mode = self.regen_mode
        new_genome.hybrid_scale = self.hybrid_scale
        return new_genome


class EvalMode:
    """Evaluation mode configuration."""
    def __init__(self, vanilla: bool = True, enable_regen_reproduction: bool = False, 
                 species_low: int = 2, species_high: int = 10):
        self.vanilla = vanilla
        self.enable_regen_reproduction = enable_regen_reproduction
        self.species_low = species_low
        self.species_high = species_high


# === Mock functions that would be implemented elsewhere ======================
def fitness_backprop_classifier(genome, Xtr, ytr, Xva, yva, steps=40, lr=5e-3, l2=1e-4, 
                                alpha_nodes=1e-3, alpha_edges=5e-4):
    """Mock fitness function - would implement actual backprop training."""
    # Simplified placeholder - real implementation would train the network
    return np.random.random() * 0.8 + 0.1

def diff_scars(prev_best, curr_best, scars, birth_gen, regen_mode_for_new):
    """Mock function for tracking lineage scars."""
    return scars or {}


# === Main NEAT Class ==========================================================
class ReproPlanaNEATPlus:
    def __init__(self, num_inputs, num_outputs, population_size=150, rng=None, output_activation='sigmoid'):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.pop_size = population_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mode = EvalMode(vanilla=True, enable_regen_reproduction=False)
        self.max_hidden_nodes = 128
        self.max_edges = 1024
        self.complexity_threshold: Optional[float] = 1.0
        
        # ---- Hardened knobs
        self.grad_clip = 5.0
        self.weight_clip = 12.0
        self.snapshot_stride = 1 if self.pop_size <= 256 else 2
        self.snapshot_max = 320
        self.min_conn_after_regen = 0.65
        self.diversity_push = 0.15
        self.max_attempts_guard = 16
        
        # Parallel eval
        try:
            cpus = int((os.cpu_count() or 2))
            self.eval_workers = int(os.environ.get("NEAT_EVAL_WORKERS", max(1, cpus - 1)))
            self.parallel_backend = os.environ.get("NEAT_EVAL_BACKEND", "thread")
        except Exception:
            self.eval_workers = 1
            self.parallel_backend = "thread"
        
        # Auto curriculum toggle
        self.auto_curriculum = True
        
        # ---- Speciation target learning (dynamic)
        self.species_target = None  # set lazily around (species_low+species_high)/2
        self.species_target_min = 2.0
        self.species_target_max = max(float(self.mode.species_high), float(self.pop_size) / 3.0)
        self.species_target_step = 0.5            # hill-climb step
        self.species_target_update_every = 5      # generations
        self._spec_learn = {"dir": 1.0, "last_best": None, "last_tgt": None, "last_reward": None}
        
        # PID controller & bandit switching
        self.species_target_mode = "auto"         # "pid" | "hill" | "auto"
        self.pid_kp = 0.35
        self.pid_ki = 0.02
        self.pid_kd = 0.10
        self.pid_i_clip = 50.0
        self._spec_learn.update({
            "pid_i": 0.0, 
            "pid_prev_err": None, 
            "score_pid": 0.0, 
            "score_hill": 0.0, 
            "eps": 0.10, 
            "last_method": "pid"
        })
        
        # Process pool "rolling restart"
        self.pool_keepalive = int(os.environ.get("NEAT_POOL_KEEPALIVE", "0"))
        self.pool_restart_every = int(os.environ.get("NEAT_POOL_RESTART_EVERY", "25"))
        self._proc_pool = None
        self._proc_pool_age = 0
        self._shm_meta = None
        
        # Additional attributes for NEAT operation
        self.population = []
        self.generation = 0
        self.compatibility_threshold = 3.0
        self.env = {}
        self.env_history = []
        self.best_ids = []
        self.snapshots_genomes = []
        self.snapshots_scars = []
        self.hidden_counts_history = []
        self.edge_counts_history = []
        self.event_log = []
        self.sex_fitness_scale = {'hermaphrodite': 1.0}
        self.regen_bonus = 0.0
        self.pollen_flow_rate = 0.1
        self.mix_asexual_base = 0.5
        
        # Initialize population
        self._init_population()
    
    def _init_population(self):
        """Initialize the population with minimal genomes."""
        for i in range(self.pop_size):
            genome = Genome(i)
            # Add input nodes
            for j in range(self.num_inputs):
                genome.nodes[j] = NodeGene(j, 'input')
            # Add output nodes
            for j in range(self.num_outputs):
                node_id = self.num_inputs + j
                genome.nodes[node_id] = NodeGene(node_id, 'output')
            # Add some initial connections
            innov = 0
            for inp in range(self.num_inputs):
                for out in range(self.num_outputs):
                    out_id = self.num_inputs + out
                    weight = self.rng.normal(0, 1)
                    genome.connections[innov] = ConnectionGene(inp, out_id, weight, True, innov)
                    innov += 1
            self.population.append(genome)
    
    def _evaluate_population(self, fitness_fn: Callable[[Genome], float]) -> List[float]:
        """Parallel evaluation (thread/process). Process mode initializes SHM metadata and restarts rolling pool if needed."""
        workers = int(getattr(self, "eval_workers", 1))
        if workers <= 1:
            return [fitness_fn(g) for g in self.population]
        backend = getattr(self, "parallel_backend", "thread")
        if backend == "process" and not _is_picklable(fitness_fn):
            print("[WARN] fitness_fn is not picklable; falling back to threads")
            backend = "thread"
        try:
            import concurrent.futures as _cf
            if backend == "process":
                import multiprocessing as _mp
                start = os.environ.get("NEAT_PROCESS_START_METHOD", "spawn")
                try:
                    ctx = _mp.get_context(start)
                except ValueError:
                    ctx = _mp.get_context("spawn")
                initargs = (getattr(self, "_shm_meta", None),)
                # persistent pool?
                if int(getattr(self, "pool_keepalive", 0)) > 0:
                    need_new = (self._proc_pool is None) or (int(getattr(self, "_proc_pool_age", 0)) >= int(getattr(self, "pool_restart_every", 25)))
                    if need_new:
                        self._close_pool()
                        self._proc_pool = _cf.ProcessPoolExecutor(
                            max_workers=workers, mp_context=ctx, initializer=_proc_init_worker, initargs=initargs
                        )
                        self._proc_pool_age = 0
                    ex = self._proc_pool
                    out = list(ex.map(fitness_fn, self.population, chunksize=max(1, len(self.population)//workers)))
                    self._proc_pool_age += 1
                    return out
                else:
                    with _cf.ProcessPoolExecutor(
                        max_workers=workers, mp_context=ctx, initializer=_proc_init_worker, initargs=initargs
                    ) as ex:
                        return list(ex.map(fitness_fn, self.population, chunksize=max(1, len(self.population)//workers)))
            else:
                with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
                    return list(ex.map(fitness_fn, self.population))
        except Exception as e:
            print("[WARN] parallel evaluation disabled:", e)
            return [fitness_fn(g) for g in self.population]
    
    def _close_pool(self):
        ex = getattr(self, "_proc_pool", None)
        if ex is not None:
            try:
                ex.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._proc_pool = None
            self._proc_pool_age = 0
    
    def _complexity_penalty(self, genome: Genome) -> float:
        """Calculate complexity penalty for a genome."""
        if self.complexity_threshold is None or self.complexity_threshold <= 0:
            return 0.0
        num_hidden = sum(1 for n in genome.nodes.values() if n.type == 'hidden')
        num_edges = sum(1 for c in genome.connections.values() if c.enabled)
        return self.complexity_threshold * (num_hidden * 0.01 + num_edges * 0.001)
    
    def _adapt_compat_threshold(self, num_species: int):
        # Aggressive, goal-driven adaptation
        low = int(self.mode.species_low)
        high = int(self.mode.species_high)
        # Lazy init target (updated by learning)
        if getattr(self, "species_target", None) is None:
            self.species_target = float((low + high) * 0.5)
        target = float(self.species_target)
        if target <= 0:
            target = (low + high) * 0.5
        err = (float(num_species) - target) / max(1.0, target)
        self.compatibility_threshold *= float(np.exp(0.18 * err))
        self.compatibility_threshold = float(np.clip(self.compatibility_threshold, 0.3, 50.0))
    
    def _learn_species_target(self, num_species: int, best_fit: float, gen: int) -> None:
        """Learn species_target: PID and Hill-Climb with bandit switching (auto mode)."""
        low, high = int(self.mode.species_low), int(self.mode.species_high)
        if self.species_target is None:
            self.species_target = float((low + high) * 0.5)
            self._spec_learn["last_best"] = float(best_fit)
            self._spec_learn["last_tgt"]  = float(self.species_target)
            self._spec_learn["last_reward"] = 0.0
            return
        # Update interval
        if gen % int(self.species_target_update_every) != 0:
            return
        st = self._spec_learn
        last_best = st.get("last_best", None)
        if last_best is None:
            st["last_best"] = float(best_fit)
            return
        reward = float(best_fit) - float(last_best)
        mode = getattr(self, "species_target_mode", "auto")
        # choose method
        method = "pid"
        if mode == "hill":
            method = "hill"
        elif mode == "auto":
            eps = float(st.get("eps", 0.10))
            # epsilon-greedy over 2 arms
            if self.rng.random() < eps:
                method = "pid" if (self.rng.random() < 0.5) else "hill"
            else:
                method = "pid" if (st.get("score_pid", 0.0) >= st.get("score_hill", 0.0)) else "hill"
        # run update
        if method == "pid":
            # error = actual - target (if too many species, harder to increase/easier to decrease target)
            err = float(num_species) - float(self.species_target)
            prev = st.get("pid_prev_err", 0.0) or 0.0
            itg  = float(st.get("pid_i", 0.0)) + err
            itg  = float(np.clip(itg, -float(self.pid_i_clip), float(self.pid_i_clip)))
            de   = err - prev
            delta = float(self.pid_kp)*err + float(self.pid_ki)*itg + float(self.pid_kd)*de
            step_max = max(0.5, 0.75)  # Don't move too much in one step
            new_t = float(self.species_target) + float(np.clip(delta, -step_max, step_max))
            new_t = float(np.clip(new_t, float(self.species_target_min), float(self.species_target_max)))
            self.species_target = new_t
            st["pid_prev_err"] = err
            st["pid_i"] = itg
            # EWMA reward
            st["score_pid"] = 0.85*float(st.get("score_pid", 0.0)) + 0.15*reward
            st["score_hill"] = 0.98*float(st.get("score_hill", 0.0))
        else:
            dir_ = float(st.get("dir", 1.0))
            last_reward = float(st.get("last_reward") or 0.0)
            if reward < (last_reward - 1e-6):
                dir_ = -dir_
            err_s = float(num_species) - float(self.species_target)
            if err_s != 0.0:
                dir_ = 0.7*dir_ + 0.3*np.sign(err_s)
            step = float(self.species_target_step)
            new_t = float(self.species_target) + step * dir_
            new_t = float(np.clip(new_t, float(self.species_target_min), float(self.species_target_max)))
            self.species_target = new_t
            st["dir"] = dir_
            st["score_hill"] = 0.85*float(st.get("score_hill", 0.0)) + 0.15*reward
            st["score_pid"] = 0.98*float(st.get("score_pid", 0.0))
        st["last_best"] = float(best_fit)
        st["last_tgt"]  = float(self.species_target)
        st["last_reward"] = float(reward)
    
    def _adaptive_refine_fitness(self, fitnesses: List[float], fitness_fn: Callable[[Genome], float]) -> List[float]:
        """Re-evaluate top individuals with extra backprop steps (lightweight two-stage evaluation)."""
        if not hasattr(fitness_fn, "refine_raw"):
            return fitnesses
        n = len(fitnesses)
        if n == 0:
            return fitnesses
        k = max(1, int(0.10 * n))
        idxs = np.argsort(fitnesses)[-k:]
        improved = list(fitnesses)
        best_now = float(np.max(fitnesses))
        for i in map(int, idxs):
            try:
                gap = best_now - float(fitnesses[i])
                factor = 2.0 if gap > 0.02 else 1.5
                raw2 = float(fitness_fn.refine_raw(self.population[i], factor=factor))
                f2 = raw2
                if not self.mode.vanilla:
                    g = self.population[i]
                    f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * (getattr(g, 'hybrid_scale', 1.0))
                    if g.regen:
                        f2 += self.regen_bonus
                f2 -= self._complexity_penalty(self.population[i])
                if np.isfinite(f2):
                    improved[i] = f2
            except Exception:
                pass
        return improved
    
    def speciate(self, fitnesses: List[float]) -> List[List[Genome]]:
        """Simple speciation - in real NEAT this would use compatibility distance."""
        # Simplified: just divide population into groups
        num_species = max(2, min(10, len(self.population) // 10))
        species = [[] for _ in range(num_species)]
        for i, genome in enumerate(self.population):
            species[i % num_species].append(genome)
        return [s for s in species if len(s) > 0]
    
    def reproduce(self, species: List[List[Genome]], fitnesses: List[float]):
        """Generate next generation through reproduction."""
        # Simplified reproduction
        new_pop = []
        for spec in species:
            if len(spec) > 0:
                # Keep best from each species
                best = max(spec, key=lambda g: fitnesses[self.population.index(g)])
                new_pop.append(best.copy())
        
        # Fill rest with mutations/crossovers
        while len(new_pop) < self.pop_size:
            if len(self.population) > 0:
                parent = self.rng.choice(self.population)
                child = parent.copy()
                child.id = len(new_pop)
                # Simple mutation
                for conn in child.connections.values():
                    if self.rng.random() < 0.1:
                        conn.weight += self.rng.normal(0, 0.1)
                new_pop.append(child)
        
        self.population = new_pop[:self.pop_size]
        self.event_log.append({
            'sexual_within': 0,
            'sexual_cross': 0,
            'asexual_regen': 0
        })
    
    def _auto_env_schedule(self, gen: int, history: List[Tuple[float, float]]) -> dict:
        """Automatic curriculum scheduling."""
        # Simple progressive difficulty
        difficulty = min(1.0, gen / 50.0)
        noise_std = difficulty * 0.1
        return {'difficulty': difficulty, 'noise_std': noise_std}
    
    def evolve(self, fitness_fn: Callable[[Genome], float], n_generations=100, target_fitness=None, verbose=True, env_schedule=None):
        history = []
        best_ever = None
        best_ever_fit = -1e9
        from math import isnan
        scars = None
        prev_best = None
        
        for gen in range(n_generations):
            self.generation = gen
            prev = history[-1] if history else (None, None)
            
            # === Curriculum ===
            if env_schedule is not None:
                env = env_schedule(gen, {'gen': gen, 'prev_best': prev[0] if prev else None, 'prev_avg': prev[1] if prev else None})
            elif getattr(self, "auto_curriculum", True):
                env = self._auto_env_schedule(gen, history)
            else:
                env = None
            
            if env is not None:
                self.env.update({k: v for k, v in env.items() if k not in {'enable_regen'}})
                if 'enable_regen' in env:
                    flag = bool(env['enable_regen'])
                    self.mode.enable_regen_reproduction = flag
                    if flag:
                        self.mix_asexual_base = max(self.mix_asexual_base, 0.30)
            
            self.env_history.append({'gen': gen, **self.env, 'regen_enabled': self.mode.enable_regen_reproduction})
            
            # Pollen flow rate based on difficulty
            diff = float(self.env.get('difficulty', 0.0))
            self.pollen_flow_rate = float(min(0.5, max(0.1, 0.1 + 0.35 * diff)))
            
            # Propagate noise to fitness instance (pickled each time for processes)
            if hasattr(fitness_fn, "set_noise_std"):
                try:
                    fitness_fn.set_noise_std(float(self.env.get("noise_std", 0.0)))
                except Exception:
                    pass
            
            # === Evaluate (parallel-aware) ===
            raw = self._evaluate_population(fitness_fn)
            fitnesses = []
            for g, f in zip(self.population, raw):
                f2 = float(f)
                if not self.mode.vanilla:
                    f2 *= self.sex_fitness_scale.get(g.sex, 1.0) * (getattr(g, 'hybrid_scale', 1.0))
                    if g.regen:
                        f2 += self.regen_bonus
                f2 -= self._complexity_penalty(g)
                if not np.isfinite(f2):
                    f2 = float(np.nan_to_num(f2, nan=-1e6, posinf=-1e6, neginf=-1e6))
                fitnesses.append(f2)
            
            # Adaptive refine for elites
            try:
                fitnesses = self._adaptive_refine_fitness(fitnesses, fitness_fn)
            except Exception:
                pass
            
            best_idx = int(np.argmax(fitnesses))
            best_fit = float(fitnesses[best_idx])
            avg_fit = float(np.mean(fitnesses))
            
            # === Snapshots (decimated & bounded) ===
            try:
                curr_best = self.population[best_idx].copy()
                scars = diff_scars(prev_best, curr_best, scars, birth_gen=gen, regen_mode_for_new=getattr(curr_best, 'regen_mode', 'split'))
                stride = int(getattr(self, "snapshot_stride", 1))
                if (gen % max(1, stride) == 0) or (gen == n_generations - 1):
                    if len(self.snapshots_genomes) >= int(getattr(self, "snapshot_max", 320)):
                        self.snapshots_genomes.pop(0)
                        self.snapshots_scars.pop(0)
                    self.snapshots_genomes.append(curr_best)
                    self.snapshots_scars.append(scars)
                prev_best = curr_best
            except Exception:
                pass
            
            history.append((best_fit, avg_fit))
            self.best_ids.append(self.population[best_idx].id)
            
            # complexity traces
            try:
                self.hidden_counts_history.append([sum(1 for n in g.nodes.values() if n.type == 'hidden') for g in self.population])
                self.edge_counts_history.append([sum(1 for c in g.connections.values() if c.enabled) for g in self.population])
            except Exception:
                self.hidden_counts_history.append([])
                self.edge_counts_history.append([])
            
            if verbose:
                noise = float(self.env.get('noise_std', 0.0))
                ev = self.event_log[-1] if self.event_log else {'sexual_within': 0, 'sexual_cross': 0, 'asexual_regen': 0}
                print(
                    f"Gen {gen:3d} | best {best_fit:.4f} | avg {avg_fit:.4f} | difficulty {diff:.2f} | noise {noise:.2f} | "
                    f"sexual {ev.get('sexual_within', 0)+ev.get('sexual_cross', 0)} | regen {ev.get('asexual_regen', 0)}"
                )
            
            if best_fit > best_ever_fit:
                best_ever_fit = best_fit
                best_ever = self.population[best_idx].copy()
            
            if target_fitness is not None and best_fit >= target_fitness:
                break
            
            species = self.speciate(fitnesses)
            
            # Learn target -> compatibility threshold
            try:
                self._learn_species_target(len(species), best_fit, gen)
            except Exception as _spe:
                print("[WARN] species target learning skipped:", _spe)
            
            self._adapt_compat_threshold(len(species))
            self.reproduce(species, fitnesses)
        
        # Champion across all generations
        if best_ever is None and self.population:
            best_ever = self.population[0].copy()
        
        # Close rolling pool if exists
        try:
            self._close_pool()
        except Exception:
            pass
        
        return best_ever, history


# === Shared-memory aware fitness =============================================
class FitnessBackpropShared:
    """
    Picklable callable that reads datasets from shared memory by label.
    Also provides refine_raw(genome, factor) for adaptive extra-steps.
    """
    def __init__(self, keys=("Xtr", "ytr", "Xva", "yva"), steps=40, lr=5e-3, l2=1e-4, alpha_nodes=1e-3, alpha_edges=5e-4):
        self.keys = tuple(keys)
        self.steps = int(steps)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.alpha_nodes = float(alpha_nodes)
        self.alpha_edges = float(alpha_edges)
        self.noise_std = 0.0
    
    def set_noise_std(self, s: float):
        self.noise_std = float(max(0.0, s))
    
    def _load(self):
        Xtr = get_shared_dataset(self.keys[0])
        ytr = get_shared_dataset(self.keys[1])
        Xva = get_shared_dataset(self.keys[2])
        yva = get_shared_dataset(self.keys[3])
        return Xtr, ytr, Xva, yva
    
    def _aug(self, X):
        s = float(self.noise_std)
        if s <= 0.0:
            return X
        rng = np.random.default_rng()
        return X + rng.normal(0.0, s, size=X.shape)
    
    def __call__(self, g: "Genome") -> float:
        Xtr, ytr, Xva, yva = self._load()
        return fitness_backprop_classifier(g, self._aug(Xtr), ytr, self._aug(Xva), yva,
                                           steps=self.steps, lr=self.lr, l2=self.l2,
                                           alpha_nodes=self.alpha_nodes, alpha_edges=self.alpha_edges)
    
    def refine_raw(self, g: "Genome", factor: float = 2.0) -> float:
        Xtr, ytr, Xva, yva = self._load()
        steps = int(max(1, round(self.steps * float(factor))))
        return fitness_backprop_classifier(g, self._aug(Xtr), ytr, self._aug(Xva), yva,
                                           steps=steps, lr=self.lr, l2=self.l2,
                                           alpha_nodes=self.alpha_nodes, alpha_edges=self.alpha_edges)


# === Experiment runner ========================================================
def _default_difficulty_schedule(gen, info):
    """Default curriculum schedule."""
    progress = gen / 60.0
    difficulty = min(1.0, progress)
    noise_std = difficulty * 0.05
    return {'difficulty': difficulty, 'noise_std': noise_std}


def run_backprop_neat_experiment(task: str, gens=60, pop=64, steps=80, out_prefix="out/exp", make_gifs: bool = True, make_lineage: bool = True, rng_seed: int = 0):
    """Run a NEAT experiment with backprop fitness evaluation."""
    # Mock data - in real use would load actual datasets
    rng = np.random.default_rng(rng_seed)
    Xtr = rng.random((100, 10))
    ytr = rng.integers(0, 2, 100)
    Xva = rng.random((30, 10))
    yva = rng.integers(0, 2, 30)
    
    num_inputs = Xtr.shape[1]
    num_outputs = 2
    
    neat = ReproPlanaNEATPlus(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        population_size=pop,
        rng=rng,
        output_activation='sigmoid'
    )
    
    # --- Shared memory registration for process-parallel zero-copy
    shm_meta = {}
    try:
        shm_meta["Xtr"] = shm_register_dataset("Xtr", Xtr, readonly=True)
        shm_meta["ytr"] = shm_register_dataset("ytr", ytr, readonly=True)
        shm_meta["Xva"] = shm_register_dataset("Xva", Xva, readonly=True)
        shm_meta["yva"] = shm_register_dataset("yva", yva, readonly=True)
        neat._shm_meta = shm_meta
    except Exception:
        neat._shm_meta = None
    
    fit = FitnessBackpropShared(steps=steps, lr=5e-3, l2=1e-4, alpha_nodes=1e-3, alpha_edges=5e-4)
    
    best, hist = neat.evolve(
        fit,
        n_generations=gens,
        verbose=True,
        env_schedule=_default_difficulty_schedule,
    )
    
    # SHM cleanup
    try:
        shm_release_all()
    except Exception:
        pass
    
    return best, hist


# === Main entry point =========================================================
if __name__ == "__main__":
    print("Running NEAT experiment with shared memory support...")
    best, history = run_backprop_neat_experiment(
        task="demo",
        gens=20,
        pop=32,
        steps=40,
        rng_seed=42
    )
    print(f"\nBest fitness: {history[-1][0]:.4f}")
    print(f"Final average: {history[-1][1]:.4f}")
