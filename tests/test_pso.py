"""
Unit tests for PSO optimizer.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer.pso import ParticleSwarmOptimizer


class TestPSO:
    def test_simple_quadratic(self):
        """PSO should find minimum of (x - 0.5)² near x=0.5."""
        def fitness(x):
            return -(x - 0.5) ** 2  # Negative because we maximize

        pso = ParticleSwarmOptimizer(
            fitness_fn=fitness,
            num_particles=20,
            max_iter=50,
            bounds=(0.0, 1.0),
            maximize=True,
        )
        best_x, best_fit = pso.optimize()
        assert abs(best_x - 0.5) < 0.05

    def test_bounds_respected(self):
        """All particle positions must stay within bounds."""
        def fitness(x):
            return -x  # Drives toward lower bound

        pso = ParticleSwarmOptimizer(
            fitness_fn=fitness,
            num_particles=10,
            max_iter=30,
            bounds=(0.2, 0.8),
            maximize=True,
        )
        best_x, _ = pso.optimize()
        assert 0.2 <= best_x <= 0.8

    def test_gbest_improves(self):
        """Global best fitness should improve over iterations."""
        def fitness(x):
            return -(x - 0.3) ** 2

        pso = ParticleSwarmOptimizer(
            fitness_fn=fitness,
            num_particles=15,
            max_iter=40,
            bounds=(0.0, 1.0),
            maximize=True,
        )
        pso.optimize()
        history = pso.get_convergence_history()
        gbest_vals = history['gbest_fitness']
        # Should be non-decreasing (maximizing)
        for i in range(1, len(gbest_vals)):
            assert gbest_vals[i] >= gbest_vals[i - 1] - 1e-10

    def test_convergence_history(self):
        """History should be recorded."""
        def fitness(x):
            return -x ** 2

        pso = ParticleSwarmOptimizer(
            fitness_fn=fitness,
            num_particles=5,
            max_iter=10,
            bounds=(0.0, 1.0),
            maximize=True,
        )
        pso.optimize()
        history = pso.get_convergence_history()
        assert len(history['gbest_fitness']) > 0
        assert len(history['gbest_position']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
