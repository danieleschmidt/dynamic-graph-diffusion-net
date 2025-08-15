"""
Self-tuning optimization and hyperparameter evolution systems.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque
import json
import threading

@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameters."""
    name: str
    current_value: Union[float, int, str]
    value_type: str  # 'float', 'int', 'categorical'
    bounds: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None
    mutation_strength: float = 0.1
    performance_history: List[float] = field(default_factory=list)

@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    config: Dict[str, Any]
    performance: float
    improvement: float
    evaluation_time: float
    timestamp: float

class SelfTuningOptimizer:
    """
    Self-tuning optimizer that automatically adjusts hyperparameters
    based on performance feedback.
    """
    
    def __init__(self, 
                 optimization_frequency: int = 10,
                 population_size: int = 20,
                 elite_fraction: float = 0.2,
                 mutation_rate: float = 0.1):
        
        self.optimization_frequency = optimization_frequency
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        
        # Hyperparameter configurations
        self.hyperparameters = {}
        self.optimization_history = deque(maxlen=1000)
        self.current_generation = 0
        self.evaluation_count = 0
        
        # Performance tracking
        self.baseline_performance = None
        self.best_performance = float('-inf')
        self.best_config = None
        
        self.logger = logging.getLogger(f'{__name__}.SelfTuningOptimizer')
        
        # Population-based optimization
        self.population = []
        self.fitness_scores = []
        
        # Bayesian optimization components
        self.gaussian_process = None
        self.acquisition_function = "expected_improvement"
        
    def register_hyperparameter(self, 
                               name: str,
                               current_value: Union[float, int, str],
                               value_type: str,
                               bounds: Optional[Tuple[float, float]] = None,
                               categories: Optional[List[str]] = None,
                               mutation_strength: float = 0.1) -> None:
        """Register a hyperparameter for optimization."""
        
        config = HyperparameterConfig(
            name=name,
            current_value=current_value,
            value_type=value_type,
            bounds=bounds,
            categories=categories,
            mutation_strength=mutation_strength
        )
        
        self.hyperparameters[name] = config
        self.logger.info(f"Registered hyperparameter: {name} = {current_value}")
    
    def optimize_step(self, 
                     performance_metric: float,
                     model: Optional[nn.Module] = None,
                     validation_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Perform one step of hyperparameter optimization.
        
        Args:
            performance_metric: Current performance (higher is better)
            model: Model for evaluation (optional)
            validation_data: Validation data for evaluation (optional)
            
        Returns:
            New hyperparameter configuration
        """
        self.evaluation_count += 1
        
        # Initialize baseline if this is the first evaluation
        if self.baseline_performance is None:
            self.baseline_performance = performance_metric
            self.best_performance = performance_metric
            self.best_config = self.get_current_config()
        
        # Update best performance
        if performance_metric > self.best_performance:
            self.best_performance = performance_metric
            self.best_config = self.get_current_config()
            self.logger.info(f"New best performance: {performance_metric:.4f}")
        
        # Record current configuration performance
        current_config = self.get_current_config()
        result = OptimizationResult(
            config=current_config,
            performance=performance_metric,
            improvement=performance_metric - self.baseline_performance,
            evaluation_time=time.time(),
            timestamp=time.time()
        )
        self.optimization_history.append(result)
        
        # Update hyperparameter performance history
        for name, value in current_config.items():
            if name in self.hyperparameters:
                self.hyperparameters[name].performance_history.append(performance_metric)
        
        # Decide whether to optimize
        should_optimize = (
            self.evaluation_count % self.optimization_frequency == 0 and
            len(self.optimization_history) >= 3
        )
        
        if should_optimize:
            return self._perform_optimization()
        else:
            return current_config
    
    def _perform_optimization(self) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        self.logger.info(f"Starting optimization at generation {self.current_generation}")
        
        # Choose optimization strategy based on available data
        if len(self.optimization_history) < 10:
            # Use random search for exploration
            new_config = self._random_search()
        elif len(self.optimization_history) < 50:
            # Use evolutionary optimization
            new_config = self._evolutionary_optimization()
        else:
            # Use Bayesian optimization for exploitation
            new_config = self._bayesian_optimization()
        
        # Apply new configuration
        self._apply_configuration(new_config)
        
        self.current_generation += 1
        return new_config
    
    def _random_search(self) -> Dict[str, Any]:
        """Random search optimization."""
        new_config = {}
        
        for name, hp_config in self.hyperparameters.items():
            if hp_config.value_type == 'float':
                if hp_config.bounds:
                    low, high = hp_config.bounds
                    new_value = random.uniform(low, high)
                else:
                    # Random perturbation around current value
                    current = float(hp_config.current_value)
                    perturbation = random.gauss(0, hp_config.mutation_strength * abs(current))
                    new_value = max(0, current + perturbation)
                new_config[name] = new_value
            
            elif hp_config.value_type == 'int':
                if hp_config.bounds:
                    low, high = hp_config.bounds
                    new_value = random.randint(int(low), int(high))
                else:
                    current = int(hp_config.current_value)
                    perturbation = random.randint(-2, 2)
                    new_value = max(1, current + perturbation)
                new_config[name] = new_value
            
            elif hp_config.value_type == 'categorical':
                if hp_config.categories:
                    new_value = random.choice(hp_config.categories)
                else:
                    new_value = hp_config.current_value
                new_config[name] = new_value
        
        return new_config
    
    def _evolutionary_optimization(self) -> Dict[str, Any]:
        """Evolutionary optimization using population-based methods."""
        
        # Initialize population if needed
        if not self.population:
            self._initialize_population()
        
        # Select parents based on fitness
        elite_count = max(1, int(self.population_size * self.elite_fraction))
        fitness_scores = self._evaluate_population()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_indices = sorted_indices[:elite_count]
        
        # Create next generation
        new_population = []
        
        # Keep elite individuals
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1_idx = random.choice(elite_indices)
            parent2_idx = random.choice(elite_indices)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            offspring = self._crossover(parent1, parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
        
        self.population = new_population
        
        # Return best individual
        best_idx = elite_indices[0]
        return self.population[best_idx]
    
    def _bayesian_optimization(self) -> Dict[str, Any]:
        """Bayesian optimization using Gaussian process surrogate model."""
        # This is a simplified Bayesian optimization
        # In practice, you'd use libraries like scikit-optimize or GPyOpt
        
        # Extract features and targets from history
        features = []
        targets = []
        
        for result in list(self.optimization_history)[-50:]:  # Use recent history
            feature_vector = []
            for name in sorted(self.hyperparameters.keys()):
                value = result.config.get(name, 0)
                if isinstance(value, str):
                    # Simple categorical encoding
                    value = hash(value) % 100
                feature_vector.append(float(value))
            
            features.append(feature_vector)
            targets.append(result.performance)
        
        if len(features) < 3:
            return self._random_search()
        
        # Simple GP approximation using nearest neighbors
        features = np.array(features)
        targets = np.array(targets)
        
        # Generate candidate points
        best_candidate = None
        best_acquisition = float('-inf')
        
        for _ in range(20):  # Evaluate 20 random candidates
            candidate_config = self._random_search()
            candidate_vector = []
            
            for name in sorted(self.hyperparameters.keys()):
                value = candidate_config.get(name, 0)
                if isinstance(value, str):
                    value = hash(value) % 100
                candidate_vector.append(float(value))
            
            candidate_vector = np.array(candidate_vector)
            
            # Simple acquisition function (Upper Confidence Bound)
            distances = np.linalg.norm(features - candidate_vector, axis=1)
            weights = np.exp(-distances / np.std(distances))
            weights /= weights.sum()
            
            predicted_mean = np.sum(weights * targets)
            predicted_std = np.sqrt(np.sum(weights * (targets - predicted_mean) ** 2))
            
            # UCB acquisition
            acquisition_value = predicted_mean + 2.0 * predicted_std
            
            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_candidate = candidate_config
        
        return best_candidate or self._random_search()
    
    def _initialize_population(self) -> None:
        """Initialize population for evolutionary optimization."""
        self.population = []
        
        for _ in range(self.population_size):
            individual = self._random_search()
            self.population.append(individual)
    
    def _evaluate_population(self) -> List[float]:
        """Evaluate fitness of current population."""
        # For simplicity, use historical performance data
        fitness_scores = []
        
        for individual in self.population:
            # Find most similar configuration in history
            best_performance = self.baseline_performance
            
            for result in self.optimization_history:
                similarity = self._config_similarity(individual, result.config)
                if similarity > 0.8:  # Similar configuration
                    best_performance = max(best_performance, result.performance)
            
            fitness_scores.append(best_performance)
        
        return fitness_scores
    
    def _config_similarity(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calculate similarity between two configurations."""
        if not config1 or not config2:
            return 0.0
        
        similarities = []
        
        for name in self.hyperparameters:
            if name in config1 and name in config2:
                val1, val2 = config1[name], config2[name]
                
                if self.hyperparameters[name].value_type in ['float', 'int']:
                    # Numerical similarity
                    if val1 == val2:
                        sim = 1.0
                    else:
                        max_val = max(abs(val1), abs(val2), 1e-8)
                        sim = 1.0 - abs(val1 - val2) / max_val
                else:
                    # Categorical similarity
                    sim = 1.0 if val1 == val2 else 0.0
                
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover operation for evolutionary optimization."""
        offspring = {}
        
        for name in self.hyperparameters:
            if random.random() < 0.5:
                offspring[name] = parent1.get(name, self.hyperparameters[name].current_value)
            else:
                offspring[name] = parent2.get(name, self.hyperparameters[name].current_value)
        
        return offspring
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for evolutionary optimization."""
        mutated = individual.copy()
        
        for name, hp_config in self.hyperparameters.items():
            if random.random() < 0.3:  # 30% chance to mutate each parameter
                if hp_config.value_type == 'float':
                    current = float(mutated[name])
                    mutation = random.gauss(0, hp_config.mutation_strength * abs(current))
                    mutated[name] = max(0, current + mutation)
                    
                    if hp_config.bounds:
                        low, high = hp_config.bounds
                        mutated[name] = max(low, min(high, mutated[name]))
                
                elif hp_config.value_type == 'int':
                    current = int(mutated[name])
                    mutation = random.randint(-1, 1)
                    mutated[name] = max(1, current + mutation)
                    
                    if hp_config.bounds:
                        low, high = hp_config.bounds
                        mutated[name] = max(int(low), min(int(high), mutated[name]))
                
                elif hp_config.value_type == 'categorical' and hp_config.categories:
                    mutated[name] = random.choice(hp_config.categories)
        
        return mutated
    
    def _apply_configuration(self, config: Dict[str, Any]) -> None:
        """Apply new configuration to hyperparameters."""
        for name, value in config.items():
            if name in self.hyperparameters:
                self.hyperparameters[name].current_value = value
        
        self.logger.info(f"Applied new configuration: {config}")
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current hyperparameter configuration."""
        return {name: hp.current_value for name, hp in self.hyperparameters.items()}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress."""
        if not self.optimization_history:
            return {"status": "no_data"}
        
        history = list(self.optimization_history)
        recent_performances = [r.performance for r in history[-10:]]
        
        return {
            "total_evaluations": len(history),
            "current_generation": self.current_generation,
            "best_performance": self.best_performance,
            "best_config": self.best_config,
            "baseline_performance": self.baseline_performance,
            "improvement": self.best_performance - self.baseline_performance if self.baseline_performance else 0,
            "recent_avg_performance": np.mean(recent_performances),
            "optimization_trend": self._calculate_trend(),
            "hyperparameters_count": len(self.hyperparameters)
        }
    
    def _calculate_trend(self) -> str:
        """Calculate optimization trend."""
        if len(self.optimization_history) < 5:
            return "insufficient_data"
        
        recent_performances = [r.performance for r in list(self.optimization_history)[-5:]]
        trend = np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]
        
        if trend > 0.001:
            return "improving"
        elif trend < -0.001:
            return "degrading"
        else:
            return "stable"

class HyperparameterEvolution:
    """
    Advanced hyperparameter evolution system using genetic algorithms
    and neural architecture search principles.
    """
    
    def __init__(self, search_space: Dict[str, Any], population_size: int = 50):
        self.search_space = search_space
        self.population_size = population_size
        self.generation = 0
        self.population = []
        self.fitness_history = []
        
        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = max(1, population_size // 10)
        
        # Multi-objective optimization
        self.objectives = ["performance", "efficiency", "robustness"]
        self.pareto_front = []
        
        self.logger = logging.getLogger(f'{__name__}.HyperparameterEvolution')
        
    def evolve(self, fitness_function: Callable, generations: int = 10) -> Dict[str, Any]:
        """
        Evolve hyperparameters over multiple generations.
        
        Args:
            fitness_function: Function that evaluates a configuration
            generations: Number of generations to evolve
            
        Returns:
            Best configuration found
        """
        # Initialize population if empty
        if not self.population:
            self._initialize_population()
        
        for gen in range(generations):
            self.logger.info(f"Evolution generation {self.generation + 1}")
            
            # Evaluate population
            fitness_scores = self._evaluate_population(fitness_function)
            self.fitness_history.append(fitness_scores)
            
            # Update Pareto front
            self._update_pareto_front(fitness_scores)
            
            # Select parents and create next generation
            new_population = self._create_next_generation(fitness_scores)
            self.population = new_population
            
            self.generation += 1
            
            # Log progress
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.logger.info(f"Generation {self.generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
        
        # Return best individual
        final_fitness = self._evaluate_population(fitness_function)
        best_idx = np.argmax(final_fitness)
        return self.population[best_idx]
    
    def _initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            individual = self._random_individual()
            self.population.append(individual)
    
    def _random_individual(self) -> Dict[str, Any]:
        """Generate random individual from search space."""
        individual = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'float':
                low = param_config.get('low', 0.0)
                high = param_config.get('high', 1.0)
                individual[param_name] = random.uniform(low, high)
            
            elif param_type == 'int':
                low = param_config.get('low', 1)
                high = param_config.get('high', 10)
                individual[param_name] = random.randint(low, high)
            
            elif param_type == 'categorical':
                choices = param_config.get('choices', ['default'])
                individual[param_name] = random.choice(choices)
            
            elif param_type == 'bool':
                individual[param_name] = random.choice([True, False])
        
        return individual
    
    def _evaluate_population(self, fitness_function: Callable) -> List[float]:
        """Evaluate fitness of entire population."""
        fitness_scores = []
        
        for individual in self.population:
            try:
                fitness = fitness_function(individual)
                fitness_scores.append(fitness)
            except Exception as e:
                self.logger.warning(f"Fitness evaluation failed: {e}")
                fitness_scores.append(0.0)
        
        return fitness_scores
    
    def _create_next_generation(self, fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Create next generation using selection, crossover, and mutation."""
        new_population = []
        
        # Elite selection
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = parent1.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform crossover."""
        offspring = {}
        
        for param_name in self.search_space:
            if random.random() < 0.5:
                offspring[param_name] = parent1.get(param_name)
            else:
                offspring[param_name] = parent2.get(param_name)
        
        return offspring
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate individual."""
        mutated = individual.copy()
        
        for param_name, param_config in self.search_space.items():
            if random.random() < 0.1:  # 10% mutation rate per parameter
                param_type = param_config.get('type', 'float')
                
                if param_type == 'float':
                    current = mutated[param_name]
                    mutation_strength = param_config.get('mutation_strength', 0.1)
                    mutation = random.gauss(0, mutation_strength * current)
                    
                    low = param_config.get('low', 0.0)
                    high = param_config.get('high', 1.0)
                    mutated[param_name] = max(low, min(high, current + mutation))
                
                elif param_type == 'int':
                    current = mutated[param_name]
                    mutation = random.randint(-1, 1)
                    
                    low = param_config.get('low', 1)
                    high = param_config.get('high', 10)
                    mutated[param_name] = max(low, min(high, current + mutation))
                
                elif param_type == 'categorical':
                    choices = param_config.get('choices', ['default'])
                    mutated[param_name] = random.choice(choices)
                
                elif param_type == 'bool':
                    mutated[param_name] = not mutated[param_name]
        
        return mutated
    
    def _update_pareto_front(self, fitness_scores: List[float]) -> None:
        """Update Pareto front for multi-objective optimization."""
        # Simplified Pareto front update
        # In practice, you'd have multiple objectives
        
        for i, (individual, fitness) in enumerate(zip(self.population, fitness_scores)):
            is_dominated = False
            
            for other_fitness in fitness_scores:
                if other_fitness > fitness:
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Add to Pareto front
                self.pareto_front.append({
                    "individual": individual.copy(),
                    "fitness": fitness,
                    "generation": self.generation
                })
        
        # Keep only recent Pareto front members
        self.pareto_front = self.pareto_front[-100:]  # Keep last 100
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary."""
        if not self.fitness_history:
            return {"status": "no_data"}
        
        current_fitness = self.fitness_history[-1] if self.fitness_history else []
        
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "best_fitness": max(current_fitness) if current_fitness else 0.0,
            "avg_fitness": np.mean(current_fitness) if current_fitness else 0.0,
            "pareto_front_size": len(self.pareto_front),
            "search_space_size": len(self.search_space),
            "convergence_rate": self._calculate_convergence_rate()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate."""
        if len(self.fitness_history) < 3:
            return 0.0
        
        recent_best = [max(gen_fitness) for gen_fitness in self.fitness_history[-3:]]
        improvement = recent_best[-1] - recent_best[0]
        
        return improvement / len(recent_best)