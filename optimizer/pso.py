"""
Particle Swarm Optimization (PSO) Engine.

Implements the PSO algorithm from Eq. 12-13 in the paper:

    v_i(t+1) = w · v_i(t) + c1 · r1 · (xbest_i - x_i) + c2 · r2 · (gbest - x_i)
    x_i(t+1) = x_i(t) + v_i(t+1)

where:
    w  — inertia weight (controls exploration vs exploitation)
    c1 — cognitive coefficient (attraction to personal best)
    c2 — social coefficient (attraction to global best)
    r1, r2 — uniform random in [0,1]

Used to optimize the middle step value x(i,n) in OPT-PCIRM
with STOI as the fitness function.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ParticleSwarmOptimizer:
    """PSO optimizer for single-parameter optimization.
    
    Optimizes a scalar value x ∈ [lb, ub] by maximizing or minimizing
    a fitness function f(x).
    
    The algorithm maintains N particles, each with a position and velocity.
    Particles are attracted to their personal best (xbest) and the swarm's
    global best (gbest).
    
    Usage:
        pso = ParticleSwarmOptimizer(fitness_fn, ...)
        best_value, best_fitness = pso.optimize()
    """
    
    def __init__(self, fitness_fn, num_particles=None, max_iter=None,
                 w=None, c1=None, c2=None, bounds=None, maximize=True,
                 verbose=False):
        """
        Args:
            fitness_fn: Callable that takes a scalar x and returns fitness.
            num_particles: N — number of particles in the swarm.
            max_iter: Maximum number of PSO iterations.
            w: Inertia weight.
            c1: Cognitive acceleration coefficient.
            c2: Social acceleration coefficient.
            bounds: Tuple (lower_bound, upper_bound) for x.
            maximize: If True, maximize fitness; else minimize.
            verbose: Print progress during optimization.
        """
        self.fitness_fn = fitness_fn
        self.N = num_particles or config.PSO_NUM_PARTICLES
        self.max_iter = max_iter or config.PSO_MAX_ITER
        self.w = w if w is not None else config.PSO_W
        self.c1 = c1 if c1 is not None else config.PSO_C1
        self.c2 = c2 if c2 is not None else config.PSO_C2
        self.lb, self.ub = bounds or config.PSO_BOUNDS
        self.maximize = maximize
        self.verbose = verbose
        
        # History for analysis
        self.history = {
            'gbest_fitness': [],
            'gbest_position': [],
            'mean_fitness': [],
        }
    
    def optimize(self):
        """Run PSO optimization.
        
        Returns:
            Tuple (best_position, best_fitness):
                best_position: The optimal x value found.
                best_fitness: The fitness at the optimal position.
        """
        N = self.N
        lb, ub = self.lb, self.ub
        
        # ── Initialize particles ──
        # Random positions in [lb, ub]
        positions = np.random.uniform(lb, ub, N)
        
        # Random velocities in [-range, range]
        v_range = (ub - lb) * 0.1
        velocities = np.random.uniform(-v_range, v_range, N)
        
        # Evaluate initial fitness
        fitness = np.array([self.fitness_fn(x) for x in positions])
        
        # Personal best
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        
        # Global best
        if self.maximize:
            gbest_idx = np.argmax(fitness)
        else:
            gbest_idx = np.argmin(fitness)
        
        gbest_position = positions[gbest_idx]
        gbest_fitness = fitness[gbest_idx]
        
        self.history['gbest_fitness'].append(gbest_fitness)
        self.history['gbest_position'].append(gbest_position)
        self.history['mean_fitness'].append(np.mean(fitness))
        
        # ── Main PSO loop ──
        for iteration in range(self.max_iter):
            for i in range(N):
                # Random coefficients
                r1 = np.random.random()
                r2 = np.random.random()
                
                # Eq. 12: Velocity update
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Clamp velocity
                max_v = (ub - lb) * 0.5
                velocities[i] = np.clip(velocities[i], -max_v, max_v)
                
                # Eq. 13: Position update
                positions[i] = positions[i] + velocities[i]
                
                # Enforce bounds (reflecting boundary)
                if positions[i] < lb:
                    positions[i] = lb
                    velocities[i] = abs(velocities[i]) * 0.5
                elif positions[i] > ub:
                    positions[i] = ub
                    velocities[i] = -abs(velocities[i]) * 0.5
                
                # Evaluate fitness
                fit = self.fitness_fn(positions[i])
                fitness[i] = fit
                
                # Update personal best
                if self.maximize:
                    if fit > pbest_fitness[i]:
                        pbest_fitness[i] = fit
                        pbest_positions[i] = positions[i]
                else:
                    if fit < pbest_fitness[i]:
                        pbest_fitness[i] = fit
                        pbest_positions[i] = positions[i]
            
            # Update global best
            if self.maximize:
                best_idx = np.argmax(pbest_fitness)
                if pbest_fitness[best_idx] > gbest_fitness:
                    gbest_fitness = pbest_fitness[best_idx]
                    gbest_position = pbest_positions[best_idx]
            else:
                best_idx = np.argmin(pbest_fitness)
                if pbest_fitness[best_idx] < gbest_fitness:
                    gbest_fitness = pbest_fitness[best_idx]
                    gbest_position = pbest_positions[best_idx]
            
            # Record history
            self.history['gbest_fitness'].append(gbest_fitness)
            self.history['gbest_position'].append(gbest_position)
            self.history['mean_fitness'].append(np.mean(fitness))
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"  PSO iter {iteration+1}/{self.max_iter}: "
                      f"gbest={gbest_position:.4f}, "
                      f"fitness={gbest_fitness:.4f}")
            
            # Early stopping: if all particles converged
            if np.std(positions) < 1e-6:
                if self.verbose:
                    print(f"  PSO converged at iteration {iteration+1}")
                break
        
        return gbest_position, gbest_fitness
    
    def get_convergence_history(self):
        """Return the optimization history for analysis.
        
        Returns:
            Dict with 'gbest_fitness', 'gbest_position', 'mean_fitness' lists.
        """
        return self.history
