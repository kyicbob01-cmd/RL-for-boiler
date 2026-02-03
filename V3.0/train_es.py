"""
V3.0 Evolution Strategy Training
Find better strategies than SC through evolutionary optimization
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool, cpu_count

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ES] Device: {device}")

# ==========================================
# Policy Network
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
# Evaluation Function
# ==========================================
def evaluate_model_on_scenario(model, scenario):
    """Run model on single scenario, return cost"""
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
            ]], dtype=torch.float32).to(device)
            
            action = model(state).item()
            power = action * 100.0
            physics.step(power, dt=0.5)
    
    # Count completed tasks
    completed = sum(1 for u in physics.units.values() if u['state'] == 'FINISHED')
    total = len(physics.units)
    
    return physics.total_cost, completed, total

def evaluate_model(model, penalize_incomplete=True):
    """Evaluate model on all benchmark scenarios
    
    Returns:
        total_fitness: cost + penalty for incomplete tasks
        costs: dict of scenario costs
    """
    model.eval()
    total_cost = 0
    total_penalty = 0
    costs = {}
    
    INCOMPLETE_PENALTY = 100.0  # Penalty per incomplete task
    
    for scenario in BENCHMARK_SCENARIOS:
        cost, completed, total = evaluate_model_on_scenario(model, scenario)
        incomplete = total - completed
        
        if penalize_incomplete and incomplete > 0:
            penalty = incomplete * INCOMPLETE_PENALTY
            total_penalty += penalty
        
        costs[scenario['name']] = cost
        total_cost += cost
    
    return total_cost + total_penalty, costs

# ==========================================
# Evolution Strategy Core
# ==========================================
def mutate_model(parent_state_dict, noise_std=0.01):
    """Create mutated copy of model parameters"""
    child_state_dict = {}
    for key, param in parent_state_dict.items():
        noise = torch.randn_like(param) * noise_std
        child_state_dict[key] = param + noise
    return child_state_dict

def evolution_strategy(
    initial_model_path,
    population_size=20,
    max_generations=500,
    noise_std=0.01,
    noise_decay=0.999,
    elite_ratio=0.2
):
    """Evolution Strategy Training"""
    
    print("\n" + "="*70)
    print("[ES] Evolution Strategy Training")
    print(f"Population: {population_size} | Max Gen: {max_generations}")
    print(f"Noise: {noise_std} | Elite: {elite_ratio*100:.0f}%")
    print("="*70)
    
    # Load initial model (BC)
    parent = Policy().to(device)
    if os.path.exists(initial_model_path):
        parent.load_state_dict(torch.load(initial_model_path, map_location=device))
        print(f"Loaded parent: {initial_model_path}")
    else:
        print("ERROR: Initial model not found!")
        return None
    
    # Evaluate initial parent
    parent_cost, parent_costs = evaluate_model(parent)
    print(f"\nInitial Parent Cost: {parent_cost:.2f}")
    
    # Get SC baseline for comparison
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
    
    print(f"SC Baseline Cost: {sc_cost:.2f}")
    print(f"Target: Find cost < {sc_cost:.2f}")
    print("-"*70)
    
    best_cost = parent_cost
    best_model_state = copy.deepcopy(parent.state_dict())
    generations_without_improvement = 0
    
    for gen in range(max_generations):
        # Generate population by mutating parent
        population = []
        parent_state = parent.state_dict()
        
        for _ in range(population_size):
            child_state = mutate_model(parent_state, noise_std)
            population.append(child_state)
        
        # Evaluate population
        fitness = []
        for i, child_state in enumerate(population):
            child = Policy().to(device)
            child.load_state_dict(child_state)
            cost, _ = evaluate_model(child)
            fitness.append((cost, child_state))
        
        # Sort by fitness (lower cost = better)
        fitness.sort(key=lambda x: x[0])
        
        # Select elites
        num_elites = max(1, int(population_size * elite_ratio))
        elites = fitness[:num_elites]
        
        best_child_cost, best_child_state = elites[0]
        
        # Check improvement
        improved = False
        if best_child_cost < best_cost:
            best_cost = best_child_cost
            best_model_state = copy.deepcopy(best_child_state)
            parent.load_state_dict(best_child_state)
            improved = True
            generations_without_improvement = 0
            
            # Save checkpoint
            torch.save(best_model_state, os.path.join(current_dir, "model_es_best.pth"))
        else:
            generations_without_improvement += 1
        
        # Adaptive noise
        noise_std *= noise_decay
        noise_std = max(0.001, noise_std)  # Minimum noise
        
        # Status
        status = ">>>" if improved else "   "
        gap = best_cost - sc_cost
        print(f"{status} Gen {gen+1:3d}: Best={best_cost:.2f}, Child={best_child_cost:.2f}, "
              f"Gap={gap:+.2f}, Noise={noise_std:.4f}, Stall={generations_without_improvement}")
        
        # Count victories for current best
        best_model = Policy().to(device)
        best_model.load_state_dict(best_model_state)
        _, best_costs = evaluate_model(best_model)
        
        wins = 0
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
            if best_costs[scenario['name']] < physics.total_cost:
                wins += 1
        
        # === DUAL TERMINATION CONDITIONS ===
        # Condition 1: 10/10 victories
        if wins == 10:
            print(f"\n[ES] SUCCESS! Achieved 10/10 victories over SC!")
            print(f"[ES] Best cost: {best_cost:.2f} vs SC: {sc_cost:.2f}")
            break
        
        # Condition 2: 100 generations without improvement (converged)
        if generations_without_improvement >= 100:
            print(f"\n[ES] CONVERGED: 100 generations without improvement")
            print(f"[ES] Best cost: {best_cost:.2f}, Victories: {wins}/10")
            break
        
        # Adaptive: Increase noise if stuck for 50 generations
        if generations_without_improvement >= 50 and generations_without_improvement % 50 == 0:
            old_noise = noise_std
            noise_std = min(0.05, noise_std * 2)
            print(f"    [Adaptive] Increasing noise: {old_noise:.4f} -> {noise_std:.4f}")
    
    # Save final model
    final_path = os.path.join(current_dir, "model_es_final.pth")
    torch.save(best_model_state, final_path)
    print(f"\n[ES] Final model saved: {final_path}")
    print(f"[ES] Best cost achieved: {best_cost:.2f}")
    
    # Final comparison
    print("\n" + "="*70)
    print("[ES] Final Comparison")
    print("="*70)
    
    final_model = Policy().to(device)
    final_model.load_state_dict(best_model_state)
    _, final_costs = evaluate_model(final_model)
    
    print(f"{'Scenario':<20} | {'SC Cost':>10} | {'ES Cost':>10} | {'Diff':>10}")
    print("-"*55)
    
    sc = TimeAwareSC()
    total_sc = 0
    total_es = 0
    wins = 0
    
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
        sc_cost = physics.total_cost
        total_sc += sc_cost
        
        es_cost = final_costs[scenario['name']]
        total_es += es_cost
        
        diff = ((sc_cost - es_cost) / sc_cost) * 100
        winner = "ES" if es_cost < sc_cost else "SC"
        if es_cost < sc_cost:
            wins += 1
        
        print(f"{scenario['name']:<20} | {sc_cost:>10.2f} | {es_cost:>10.2f} | {diff:>+9.1f}%")
    
    print("-"*55)
    improvement = ((total_sc - total_es) / total_sc) * 100
    print(f"{'TOTAL':<20} | {total_sc:>10.2f} | {total_es:>10.2f} | {improvement:>+9.1f}%")
    print(f"{'VICTORIES':<20} |            | {wins:>10}/10 |")
    
    return final_model

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Prefer the best ES model if available (for continued training)
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
    
    # Run Evolution Strategy
    evolution_strategy(
        initial_model_path=start_path,
        population_size=30,      # Number of candidates per generation
        max_generations=500,     # Maximum generations
        noise_std=0.02,          # Initial mutation strength
        noise_decay=0.998,       # Noise decay per generation
        elite_ratio=0.1          # Top 10% become parents
    )
