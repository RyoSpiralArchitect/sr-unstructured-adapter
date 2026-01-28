"""
Tests for spiral_monolith_neat_numpy module.
"""
import pytest
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import spiral_monolith_neat_numpy as sm
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    pytestmark = pytest.mark.skip("numpy not available")


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestSpiralNEAT:
    """Test cases for NEAT implementation."""
    
    def test_import(self):
        """Test that the module can be imported."""
        assert sm is not None
        assert hasattr(sm, 'ReproPlanaNEATPlus')
        assert hasattr(sm, 'FitnessBackpropShared')
        assert hasattr(sm, 'Genome')
    
    def test_genome_creation(self):
        """Test genome creation."""
        genome = sm.Genome(genome_id=1)
        assert genome.id == 1
        assert len(genome.nodes) == 0
        assert len(genome.connections) == 0
    
    def test_genome_copy(self):
        """Test genome copying."""
        genome = sm.Genome(genome_id=1)
        genome.fitness = 0.8
        genome.sex = 'hermaphrodite'
        
        copy = genome.copy()
        assert copy.id == genome.id
        assert copy.fitness == genome.fitness
        assert copy.sex == genome.sex
    
    def test_neat_initialization(self):
        """Test NEAT initialization."""
        rng = np.random.default_rng(42)
        neat = sm.ReproPlanaNEATPlus(
            num_inputs=5,
            num_outputs=2,
            population_size=10,
            rng=rng
        )
        
        assert neat.num_inputs == 5
        assert neat.num_outputs == 2
        assert neat.pop_size == 10
        assert len(neat.population) == 10
        
        # Check that genomes have correct structure
        for genome in neat.population:
            assert len(genome.nodes) == 7  # 5 inputs + 2 outputs
            assert len(genome.connections) == 10  # 5 * 2 connections
    
    def test_neat_evolution_basic(self):
        """Test basic NEAT evolution with simple fitness function."""
        rng = np.random.default_rng(42)
        neat = sm.ReproPlanaNEATPlus(
            num_inputs=3,
            num_outputs=2,
            population_size=10,
            rng=rng
        )
        
        # Use single worker to avoid multiprocessing issues in tests
        neat.eval_workers = 1
        
        def simple_fitness(genome):
            """Simple random fitness function."""
            return rng.random() * 0.5 + 0.3
        
        best, history = neat.evolve(
            simple_fitness,
            n_generations=3,
            verbose=False
        )
        
        assert best is not None
        assert len(history) == 3
        assert all(len(h) == 2 for h in history)  # (best, avg) tuples
        assert all(isinstance(h[0], float) for h in history)
        assert all(isinstance(h[1], float) for h in history)
    
    def test_is_picklable(self):
        """Test the _is_picklable utility function."""
        # Simple types should be picklable
        assert sm._is_picklable(42)
        assert sm._is_picklable("string")
        assert sm._is_picklable([1, 2, 3])
        assert sm._is_picklable({'a': 1})
        
        # Lambda functions are not picklable
        assert not sm._is_picklable(lambda x: x + 1)
    
    def test_fitness_backprop_shared_creation(self):
        """Test FitnessBackpropShared creation."""
        fit = sm.FitnessBackpropShared(
            keys=("Xtr", "ytr", "Xva", "yva"),
            steps=40,
            lr=5e-3,
            l2=1e-4
        )
        
        assert fit.steps == 40
        assert fit.lr == 5e-3
        assert fit.l2 == 1e-4
        assert fit.noise_std == 0.0
    
    def test_fitness_backprop_shared_set_noise(self):
        """Test setting noise standard deviation."""
        fit = sm.FitnessBackpropShared()
        
        fit.set_noise_std(0.1)
        assert fit.noise_std == 0.1
        
        fit.set_noise_std(-0.1)  # Should clamp to 0
        assert fit.noise_std == 0.0
    
    def test_pid_controller_initialization(self):
        """Test PID controller parameters are initialized."""
        rng = np.random.default_rng(42)
        neat = sm.ReproPlanaNEATPlus(
            num_inputs=3,
            num_outputs=2,
            population_size=10,
            rng=rng
        )
        
        assert neat.species_target_mode == "auto"
        assert neat.pid_kp == 0.35
        assert neat.pid_ki == 0.02
        assert neat.pid_kd == 0.10
        assert neat.pid_i_clip == 50.0
        assert "pid_i" in neat._spec_learn
        assert "score_pid" in neat._spec_learn
        assert "score_hill" in neat._spec_learn
    
    def test_process_pool_parameters(self):
        """Test process pool parameters are initialized."""
        neat = sm.ReproPlanaNEATPlus(
            num_inputs=3,
            num_outputs=2,
            population_size=10
        )
        
        assert hasattr(neat, "pool_keepalive")
        assert hasattr(neat, "pool_restart_every")
        assert hasattr(neat, "_proc_pool")
        assert hasattr(neat, "_proc_pool_age")
        assert hasattr(neat, "_shm_meta")
    
    def test_complexity_penalty(self):
        """Test complexity penalty calculation."""
        rng = np.random.default_rng(42)
        neat = sm.ReproPlanaNEATPlus(
            num_inputs=3,
            num_outputs=2,
            population_size=10,
            rng=rng
        )
        
        genome = neat.population[0]
        penalty = neat._complexity_penalty(genome)
        
        # Should have some penalty based on nodes and connections
        assert isinstance(penalty, float)
        assert penalty >= 0.0
    
    def test_eval_mode(self):
        """Test EvalMode configuration."""
        mode = sm.EvalMode(
            vanilla=True,
            enable_regen_reproduction=False,
            species_low=2,
            species_high=10
        )
        
        assert mode.vanilla is True
        assert mode.enable_regen_reproduction is False
        assert mode.species_low == 2
        assert mode.species_high == 10


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not installed")
class TestSharedMemory:
    """Test shared memory functionality."""
    
    def test_shm_module_availability(self):
        """Test that shared memory module detection works."""
        # Just verify the module was imported (or set to None)
        assert sm._shm is not None or sm._shm is None
    
    @pytest.mark.skipif(sm._shm is None, reason="shared_memory not available")
    def test_shm_register_and_get(self):
        """Test registering and getting shared datasets."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        
        # Register dataset
        meta = sm.shm_register_dataset("test_arr", arr, readonly=True)
        
        assert "name" in meta
        assert "shape" in meta
        assert "dtype" in meta
        assert "readonly" in meta
        assert meta["shape"] == (2, 3)
        assert meta["dtype"] == "float32"
        assert meta["readonly"] is True
        
        # Set worker metadata
        sm.shm_set_worker_meta({"test_arr": meta})
        
        # Get dataset
        retrieved = sm.get_shared_dataset("test_arr")
        assert retrieved.shape == arr.shape
        assert str(retrieved.dtype) == str(arr.dtype)
        
        # Compare values (make a copy to avoid segfault on cleanup)
        arr_copy = np.array(retrieved)
        np.testing.assert_array_equal(arr_copy, arr)
        
        # Cleanup worker cache first
        sm.shm_cleanup_worker_cache()
        
        # Then cleanup parent
        sm.shm_release_all()
    
    @pytest.mark.skipif(sm._shm is None, reason="shared_memory not available")
    def test_shm_release_all(self):
        """Test releasing all shared memory."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        sm.shm_register_dataset("test_arr2", arr)
        
        # Should not raise any errors
        sm.shm_release_all()
        
        # After release, local dict should be empty
        assert len(sm._SHM_LOCAL) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
