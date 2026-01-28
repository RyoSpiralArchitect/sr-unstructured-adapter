# Spiral Monolith NEAT with Shared Memory Support

This module implements NeuroEvolution of Augmenting Topologies (NEAT) with shared memory support for efficient parallel processing.

## Features

- **NEAT Algorithm**: Full implementation of the NEAT genetic algorithm for evolving neural networks
- **Shared Memory Support**: Zero-copy data sharing across processes using Python's `multiprocessing.shared_memory`
- **PID Controller**: Adaptive species target control using PID and Hill-Climb algorithms with bandit switching
- **Parallel Evaluation**: Support for both thread-based and process-based parallel fitness evaluation
- **Backpropagation Fitness**: Integrated backpropagation-based fitness evaluation for classification tasks
- **Process Pool Management**: Rolling restart capability for long-running evolutionary experiments

## Installation

The module requires numpy and matplotlib. Install them as optional dependencies:

```bash
pip install -e .[neat]
```

## Quick Start

```python
import spiral_monolith_neat_numpy as sm
import numpy as np

# Create sample data
rng = np.random.default_rng(42)
Xtr = rng.random((100, 10))
ytr = rng.integers(0, 2, 100)

# Run experiment
best, history = sm.run_backprop_neat_experiment(
    task="demo",
    gens=50,
    pop=64,
    steps=40,
    rng_seed=42
)

print(f"Best fitness: {history[-1][0]:.4f}")
```

## Shared Memory Usage

For process-based parallel evaluation with zero-copy data sharing:

```python
# Register datasets in shared memory
shm_meta = {}
shm_meta["Xtr"] = sm.shm_register_dataset("Xtr", Xtr, readonly=True)
shm_meta["ytr"] = sm.shm_register_dataset("ytr", ytr, readonly=True)

# Create NEAT instance
neat = sm.ReproPlanaNEATPlus(
    num_inputs=10,
    num_outputs=2,
    population_size=64
)
neat._shm_meta = shm_meta

# Use picklable fitness function that reads from shared memory
fit = sm.FitnessBackpropShared(
    keys=("Xtr", "ytr", "Xva", "yva"),
    steps=40
)

# Evolve with process-based parallelism
best, history = neat.evolve(fit, n_generations=100)

# Cleanup
sm.shm_release_all()
```

## Configuration

### Environment Variables

- `NEAT_EVAL_WORKERS`: Number of parallel workers (default: CPU count - 1)
- `NEAT_EVAL_BACKEND`: Parallel backend, either "thread" or "process" (default: "thread")
- `NEAT_PROCESS_START_METHOD`: Multiprocessing start method - "spawn", "fork", or "forkserver" (default: "spawn")
- `NEAT_POOL_KEEPALIVE`: If > 0, keeps process pool alive across generations (default: 0)
- `NEAT_POOL_RESTART_EVERY`: Restart pool after N generations when keepalive is enabled (default: 25)

### Key Parameters

**ReproPlanaNEATPlus**:
- `num_inputs`: Number of input neurons
- `num_outputs`: Number of output neurons
- `population_size`: Size of the population (default: 150)
- `species_target_mode`: Species target control mode - "pid", "hill", or "auto" (default: "auto")
- `pid_kp`, `pid_ki`, `pid_kd`: PID controller gains (defaults: 0.35, 0.02, 0.10)

**FitnessBackpropShared**:
- `keys`: Tuple of dataset labels in shared memory (default: ("Xtr","ytr","Xva","yva"))
- `steps`: Number of backpropagation steps (default: 40)
- `lr`: Learning rate (default: 5e-3)
- `l2`: L2 regularization (default: 1e-4)
- `alpha_nodes`: Node complexity penalty (default: 1e-3)
- `alpha_edges`: Edge complexity penalty (default: 5e-4)

## Architecture

### Core Classes

- **Genome**: Represents a neural network with nodes and connections
- **NodeGene**: Individual node in the network (input, hidden, or output)
- **ConnectionGene**: Connection between nodes with weight and enabled status
- **ReproPlanaNEATPlus**: Main NEAT algorithm implementation
- **FitnessBackpropShared**: Picklable fitness function using shared memory
- **EvalMode**: Configuration for evaluation mode

### Key Methods

**ReproPlanaNEATPlus**:
- `evolve()`: Run the evolutionary algorithm
- `_evaluate_population()`: Parallel fitness evaluation with thread or process backend
- `_learn_species_target()`: Adaptive species target control with PID/Hill-Climb
- `_adaptive_refine_fitness()`: Two-stage evaluation with extra backprop steps for elites

### Shared Memory Functions

- `shm_register_dataset()`: Register numpy array in shared memory
- `shm_set_worker_meta()`: Install shared memory metadata in worker process
- `get_shared_dataset()`: Retrieve dataset from shared memory in worker
- `shm_release_all()`: Cleanup all shared memory segments
- `shm_cleanup_worker_cache()`: Cleanup worker-side shared memory handles

## Adaptive Features

### PID Controller
The module includes a PID controller for adaptive species target management, which:
- Dynamically adjusts the target number of species based on fitness progress
- Uses epsilon-greedy bandit switching between PID and Hill-Climb methods
- Maintains exponentially weighted moving averages of method performance

### Process Pool Management
For long-running experiments, the module supports:
- Persistent process pools to avoid repeated spawn overhead
- Rolling restart every N generations to prevent memory leaks
- Shared memory metadata passing to workers for zero-copy data access

## Testing

Run the test suite:

```bash
pytest tests/test_spiral_neat.py -v
```

## Performance Tips

1. **Use process parallelism** with shared memory for large datasets:
   ```bash
   export NEAT_EVAL_BACKEND=process
   export NEAT_EVAL_WORKERS=8
   ```

2. **Enable persistent pools** for multi-generation runs:
   ```bash
   export NEAT_POOL_KEEPALIVE=1
   export NEAT_POOL_RESTART_EVERY=25
   ```

3. **Adjust PID parameters** if species count oscillates:
   ```python
   neat.pid_kp = 0.5  # Increase for faster response
   neat.pid_ki = 0.01  # Decrease to reduce overshoot
   neat.pid_kd = 0.15  # Increase for damping
   ```

## Troubleshooting

**Segmentation faults with shared memory**:
- Ensure you call `shm_cleanup_worker_cache()` before `shm_release_all()`
- Use "spawn" start method on Linux/macOS: `export NEAT_PROCESS_START_METHOD=spawn`

**Fitness function not picklable**:
- The module will automatically fall back to thread-based parallelism
- Use `FitnessBackpropShared` for process-based parallelism

**Poor species diversity**:
- Adjust species target range: `neat.species_target_min` and `neat.species_target_max`
- Increase diversity push: `neat.diversity_push = 0.20`

## License

SPDX-License-Identifier: AGPL-3.0-or-later
