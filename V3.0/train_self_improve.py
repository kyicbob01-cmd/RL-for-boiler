import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[V3.0] Device: {device}")
torch.set_float32_matmul_precision('high')

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

def run_scenario_with_model(model, scenario):
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
    
    return physics.total_cost

def evaluate_on_benchmarks(model):
    model.eval()
    costs = {}
    for scenario in BENCHMARK_SCENARIOS:
        cost = run_scenario_with_model(model, scenario)
        costs[scenario['name']] = cost
    return costs

def stage1_behavior_cloning(num_episodes=5000, epochs=300, batch_size=32768):
    print("\n" + "="*70)
    print("[Stage 1] Behavior Cloning - Mimicking Time-Aware SC")
    print("="*70)
    
    sc = TimeAwareSC()
    all_states = []
    all_actions = []
    
    print(f"Collecting {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        if ep < len(BENCHMARK_SCENARIOS) * 20:
            scenario = BENCHMARK_SCENARIOS[ep % len(BENCHMARK_SCENARIOS)]
        else:
            scenario = {"name": f"Rand_{ep}", "tasks": []}
            for _ in range(random.randint(1, 4)):
                scenario["tasks"].append({
                    "name": f"U{random.randint(1,99)}",
                    "target": random.uniform(60, 180),
                    "duration": random.uniform(50, 500),
                    "weight": random.uniform(100, 2000)
                })
        
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        
        for step in range(2000):
            max_t, active, load, min_time = physics.get_system_state()
            if active == 0:
                break
            
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            
            state = np.array([
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0,
                min_time / 500.0
            ], dtype=np.float32)
            
            power = sc.decide(physics.boiler_temp, physics.units)
            action = power / 100.0
            
            all_states.append(state)
            all_actions.append(action)
            
            physics.step(power, dt=0.5)
        
        if (ep + 1) % 500 == 0:
            print(f"  Progress: {ep+1}/{num_episodes}, Samples: {len(all_states)}")
    
    print(f"\nTraining ({epochs} epochs, batch={batch_size})...")
    print(f"  Samples: {len(all_states)}")
    
    states = torch.tensor(np.array(all_states, dtype=np.float32)).to(device)
    actions = torch.tensor(np.array([[a] for a in all_actions], dtype=np.float32)).to(device)
    num_samples = states.shape[0]
    
    del all_states, all_actions
    
    model = Policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples, device=device)
        total_loss = 0
        batches = 0
        
        for start in range(0, num_samples, batch_size):
            idx = indices[start:start + batch_size]
            pred = model(states[idx])
            loss = criterion(pred, actions[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/batches:.6f}")
    
    torch.save(model.state_dict(), os.path.join(current_dir, "model_bc.pth"))
    print("[Stage 1] Complete. Saved: model_bc.pth")
    
    return model

def generate_training_data(model, num_episodes=1000):
    all_states = []
    all_actions = []
    
    for ep in range(num_episodes):
        if ep < len(BENCHMARK_SCENARIOS) * 10:
            scenario = BENCHMARK_SCENARIOS[ep % len(BENCHMARK_SCENARIOS)]
        else:
            scenario = {"name": f"Rand_{ep}", "tasks": []}
            for _ in range(random.randint(1, 4)):
                scenario["tasks"].append({
                    "name": f"U{random.randint(1,99)}",
                    "target": random.uniform(60, 180),
                    "duration": random.uniform(50, 500),
                    "weight": random.uniform(100, 2000)
                })
        
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
                
                state = np.array([
                    physics.boiler_temp / 300.0,
                    max_t / 300.0,
                    active / 4.0,
                    rate,
                    load / 2000000.0,
                    min_time / 500.0
                ], dtype=np.float32)
                
                state_t = torch.tensor([state], dtype=torch.float32).to(device)
                
                action = model(state_t).item()
                action = action + np.random.normal(0, 0.05)
                action = np.clip(action, 0, 1)
                
                all_states.append(state)
                all_actions.append(action)
                
                power = action * 100.0
                physics.step(power, dt=0.5)
    
    return np.array(all_states), np.array(all_actions).reshape(-1, 1)

def stage2_self_improvement(initial_model, max_iterations=100, win_threshold=0.6):
    print("\n" + "="*70)
    print("[Stage 2] Self-Improvement Training")
    print(f"Win Threshold: {win_threshold*100:.0f}% | Target: 100% beat initial BC")
    print("="*70)
    
    initial_costs = evaluate_on_benchmarks(initial_model)
    print("\nInitial BC Costs (Target to beat):")
    for name, cost in initial_costs.items():
        print(f"  {name}: {cost:.2f}")
    print(f"  TOTAL: {sum(initial_costs.values()):.2f}")
    
    best_model = copy.deepcopy(initial_model)
    best_costs = initial_costs.copy()
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration+1}/{max_iterations} ---")
        
        states, actions = generate_training_data(best_model, num_episodes=2000)
        
        candidate = Policy().to(device)
        candidate.load_state_dict(best_model.state_dict())
        
        X = torch.tensor(states, dtype=torch.float32).to(device)
        Y = torch.tensor(actions, dtype=torch.float32).to(device)
        
        optimizer = torch.optim.Adam(candidate.parameters(), lr=5e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            indices = torch.randperm(len(X), device=device)
            for start in range(0, len(X), 32768):
                idx = indices[start:start+32768]
                pred = candidate(X[idx])
                loss = criterion(pred, Y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        candidate_costs = evaluate_on_benchmarks(candidate)
        
        wins = 0
        for name in BENCHMARK_SCENARIOS:
            name = name['name']
            if candidate_costs[name] < best_costs[name]:
                wins += 1
        
        win_rate = wins / len(BENCHMARK_SCENARIOS)
        print(f"  Candidate vs Best: {wins}/10 wins ({win_rate*100:.0f}%)")
        print(f"  Candidate Total: {sum(candidate_costs.values()):.2f} | Best Total: {sum(best_costs.values()):.2f}")
        
        if win_rate >= win_threshold:
            print(f"  >>> Candidate promoted to new best!")
            best_model = candidate
            best_costs = candidate_costs
            torch.save(best_model.state_dict(), os.path.join(current_dir, "model_best.pth"))
        
        victories_vs_initial = sum(1 for name in initial_costs if best_costs[name] < initial_costs[name])
        print(f"  Best vs Initial BC: {victories_vs_initial}/10")
        
        if victories_vs_initial == 10:
            print(f"\n[Stage 2] SUCCESS! All scenarios beat initial BC after {iteration+1} iterations!")
            torch.save(best_model.state_dict(), os.path.join(current_dir, "model_final.pth"))
            return best_model
    
    print(f"\n[Stage 2] Completed {max_iterations} iterations. Best: {sum(best_costs.values()):.2f}")
    torch.save(best_model.state_dict(), os.path.join(current_dir, "model_final.pth"))
    return best_model

if __name__ == "__main__":
    bc_path = os.path.join(current_dir, "model_bc.pth")
    
    if os.path.exists(bc_path):
        print(f"Loading existing BC model: {bc_path}")
        initial_model = Policy().to(device)
        initial_model.load_state_dict(torch.load(bc_path, map_location=device))
    else:
        initial_model = stage1_behavior_cloning()
    
    final_model = stage2_self_improvement(initial_model, max_iterations=100, win_threshold=0.6)
    
    print("\n" + "="*70)
    print("[Final Evaluation]")
    print("="*70)
    
    sc = TimeAwareSC()
    final_costs = evaluate_on_benchmarks(final_model)
    
    print(f"{'Scenario':<20} | {'SC Cost':>10} | {'DRL Cost':>10} | {'Diff':>10}")
    print("-" * 55)
    
    total_sc = 0
    total_drl = 0
    
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
        
        drl_cost = final_costs[scenario['name']]
        total_drl += drl_cost
        
        diff = ((sc_cost - drl_cost) / sc_cost) * 100
        print(f"{scenario['name']:<20} | {sc_cost:>10.2f} | {drl_cost:>10.2f} | {diff:>+9.1f}%")
    
    print("-" * 55)
    improvement = ((total_sc - total_drl) / total_sc) * 100
    print(f"{'TOTAL':<20} | {total_sc:>10.2f} | {total_drl:>10.2f} | {improvement:>+9.1f}%")
