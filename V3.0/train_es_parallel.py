"""
V3.0 Evolution Strategy Training (Multiprocessing Version)
Uses multiple CPU cores for parallel candidate evaluation
Optimized for i9-13900 (24 cores)
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# ==========================================
# Policy Network (defined at module level for pickling)
# ==========================================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    
    def forward(self, state):
        return self.net(state)

# ==========================================
# Evaluation Functions (defined at module level for multiprocessing)
# ==========================================
def evaluate_single_candidate(args):
    """Evaluate a single candidate - runs in separate process"""
    child_state_dict, scenario_list, incomplete_penalty = args
    
    # Import here to avoid issues with multiprocessing
    from boiler_env import BoilerPhysics
    
    # Create model in CPU for this process
    model = Policy()
    model.load_state_dict(child_state_dict)
    model.eval()
    
    total_cost = 0
    total_incomplete = 0
    
    for scenario in scenario_list:
        physics = BoilerPhysics()
        physics.reset()
        
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        
        with torch.no_grad():
            for _ in range(2000):
                max_t, active, load, min_time = physics.get_system_state()
                if active == 0:
                    break
                
                rate = physics.boiler_temp - prev_temp
                prev_temp = physics.boiler_temp
                
                state = torch.tensor([[
                    physics.boiler_temp / 300.0,
                    max_t / 300.0,
                    active / 4.0,
                    rate,
                    load / 2000000.0,
                    min_time / 500.0
                ]], dtype=torch.float32)
                
                action = model(state).item()
                power = action * 100.0
                physics.step(power, dt=0.5)
        
        total_cost += physics.total_cost
        
        # Count incomplete tasks
        completed = sum(1 for u in physics.units.values() if u['state'] == 'FINISHED')
        incomplete = len(physics.units) - completed
        total_incomplete += incomplete
    
    # Add penalty
    fitness = total_cost + total_incomplete * incomplete_penalty
    
    return fitness, child_state_dict

def mutate_model(parent_state_dict, noise_std=0.01):
    """Create mutated copy of model parameters"""
    child_state_dict = {}
    for key, param in parent_state_dict.items():
        noise = torch.randn_like(param) * noise_std
        child_state_dict[key] = param + noise
    return child_state_dict

# ==========================================
# Main Evolution Strategy
# ==========================================
def evolution_strategy_parallel(
    initial_model_path,
    population_size=30,
    max_generations=500,
    noise_std=0.02,
    noise_decay=0.998,
    elite_ratio=0.1,
    num_workers=16,
    incomplete_penalty=100.0
):
    """Evolution Strategy with Parallel Evaluation"""
    
    from boiler_env import BoilerPhysics
    from benchmark import BENCHMARK_SCENARIOS
    from time_aware_sc import TimeAwareSC
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("[ES] Evolution Strategy Training (PARALLEL)")
    print(f"Population: {population_size} | Max Gen: {max_generations}")
    print(f"Noise: {noise_std} | Elite: {elite_ratio*100:.0f}%")
    print(f"Workers: {num_workers} | Device: {device}")
    print("="*70)
    
    # Load initial model
    parent = Policy().to(device)
    if os.path.exists(initial_model_path):
        parent.load_state_dict(torch.load(initial_model_path, map_location=device))
        print(f"Loaded parent: {initial_model_path}")
    else:
        print("ERROR: Initial model not found!")
        return None
    
    # Get SC baseline
    sc = TimeAwareSC()
    sc_cost = 0
    for scenario in BENCHMARK_SCENARIOS:
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        for _ in range(2000):
            _, active, _, _ = physics.get_system_state()
            if active == 0: break
            power = sc.decide(physics.boiler_temp, physics.units)
            physics.step(power, dt=0.5)
        sc_cost += physics.total_cost
    
    # Evaluate initial parent
    parent_state_cpu = {k: v.cpu() for k, v in parent.state_dict().items()}
    parent_fitness, _ = evaluate_single_candidate((parent_state_cpu, BENCHMARK_SCENARIOS, incomplete_penalty))
    
    print(f"\nInitial Parent Fitness: {parent_fitness:.2f}")
    print(f"SC Baseline Cost: {sc_cost:.2f}")
    print(f"Target: Find fitness < {sc_cost:.2f}")
    print("-"*70)
    
    best_cost = parent_fitness
    best_model_state = copy.deepcopy(parent_state_cpu)
    generations_without_improvement = 0
    
    for gen in range(max_generations):
        # Generate population
        population = []
        for _ in range(population_size):
            child_state = mutate_model(best_model_state, noise_std)
            population.append((child_state, BENCHMARK_SCENARIOS, incomplete_penalty))
        
        # Parallel evaluation using ProcessPoolExecutor
        fitness_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(evaluate_single_candidate, args) for args in population]
            for future in as_completed(futures):
                fitness_results.append(future.result())
        
        # Sort by fitness
        fitness_results.sort(key=lambda x: x[0])
        
        best_child_cost, best_child_state = fitness_results[0]
        
        # Check improvement
        improved = False
        if best_child_cost < best_cost:
            best_cost = best_child_cost
            best_model_state = copy.deepcopy(best_child_state)
            improved = True
            generations_without_improvement = 0
            
            # Save checkpoint
            torch.save(best_model_state, os.path.join(current_dir, "model_es_best.pth"))
        else:
            generations_without_improvement += 1
        
        # Adaptive noise
        noise_std *= noise_decay
        noise_std = max(0.001, noise_std)
        
        # Count wins
        wins = sum(1 for f, _ in fitness_results if f < sc_cost)
        
        # Status
        status = ">>>" if improved else "   "
        gap = best_cost - sc_cost
        print(f"{status} Gen {gen+1:3d}: Best={best_cost:.2f}, Child={best_child_cost:.2f}, "
              f"Gap={gap:+.2f}, Noise={noise_std:.4f}, Stall={generations_without_improvement}")
        
        # Termination conditions
        if wins >= 10:
            print(f"\n[ES] SUCCESS! 10/10 candidates beat SC!")
            break
        
        # Adaptive noise: increase every 100 stalled generations
        if generations_without_improvement >= 100 and generations_without_improvement % 100 == 0:
            old_noise = noise_std
            noise_std = min(0.05, noise_std * 2)
            print(f"    [Adaptive] Stall={generations_without_improvement}, Increasing noise: {old_noise:.4f} -> {noise_std:.4f}")
    
    # Save final model
    final_path = os.path.join(current_dir, "model_es_final.pth")
    torch.save(best_model_state, final_path)
    print(f"\n[ES] Final model saved: {final_path}")
    print(f"[ES] Best fitness achieved: {best_cost:.2f}")
    
    return best_model_state

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Windows multiprocessing fix
    mp.freeze_support()
    
    # Reduce worker count to avoid Windows memory paging issues
    # Each worker loads PyTorch (~2-3GB), 32GB RAM can't handle 16 workers
    num_cores = mp.cpu_count()
    num_workers = min(6, num_cores // 4)  # Conservative: 6 workers max
    
    print(f"[ES] Detected {num_cores} CPU cores, using {num_workers} workers (memory-safe)")
    
    # Find best starting model
    es_best_path = os.path.join(current_dir, "model_es_best.pth")
    es_final_path = os.path.join(current_dir, "model_es_final.pth")
    bc_path = os.path.join(current_dir, "model_bc.pth")
    
    if os.path.exists(es_best_path):
        start_path = es_best_path
        print("[ES] Continuing from previous best model...")
    elif os.path.exists(es_final_path):
        start_path = es_final_path
        print("[ES] Continuing from previous final model...")
    elif os.path.exists(bc_path):
        start_path = bc_path
        print("[ES] Starting from BC model...")
    else:
        print("ERROR: No model found. Run train_self_improve.py first.")
        sys.exit(1)
    
    # Run parallel ES
    evolution_strategy_parallel(
        initial_model_path=start_path,
        population_size=30,
        max_generations=3000,
        noise_std=0.02,
        noise_decay=0.998,
        elite_ratio=0.1,
        num_workers=num_workers,
        incomplete_penalty=100.0
    )
