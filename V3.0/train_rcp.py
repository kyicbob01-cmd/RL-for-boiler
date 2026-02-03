"""
V3.0 RCP Training
Train Return-Conditioned Policy using Time-Aware SC trajectories
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC
from policy import RCPolicy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RCP Training] Device: {device}")

# GPU Optimization
torch.set_float32_matmul_precision('high')

# ==========================================
# Data Collection with Episode Costs
# ==========================================
def collect_rcp_data(num_episodes=10000):
    """Collect (state, action, episode_cost) triplets from Time-Aware SC"""
    print(f"[RCP] Collecting Data ({num_episodes} episodes)...")
    
    sc = TimeAwareSC()
    all_states = []
    all_actions = []
    all_costs = []  # Cost for each sample (episode total cost)
    
    for ep in range(num_episodes):
        # Scenario selection (mix of benchmark and random)
        if ep < len(BENCHMARK_SCENARIOS) * 50:
            scenario = BENCHMARK_SCENARIOS[ep % len(BENCHMARK_SCENARIOS)]
        else:
            import random
            scenario = {"name": f"Rand_{ep}", "tasks": []}
            for _ in range(random.randint(1, 4)):
                scenario["tasks"].append({
                    "name": f"U{random.randint(1,99)}",
                    "target": random.uniform(60, 180),
                    "duration": random.uniform(50, 500),
                    "weight": random.uniform(100, 2000)
                })
        
        # Run simulation and collect trajectory
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        trajectory = []
        
        for step in range(2000):
            max_t, active, load, min_time = physics.get_system_state()
            if active == 0:
                break
            
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            
            # 6D State
            state = np.array([
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0,
                min_time / 500.0
            ], dtype=np.float32)
            
            # Expert action
            power = sc.decide(physics.boiler_temp, physics.units)
            action = power / 100.0
            
            trajectory.append((state, action))
            physics.step(power, dt=0.5)
        
        # Get episode total cost
        episode_cost = physics.total_cost
        
        # Add all samples with their episode cost
        for state, action in trajectory:
            all_states.append(state)
            all_actions.append(action)
            all_costs.append(episode_cost)
        
        if (ep + 1) % 1000 == 0:
            print(f"  Progress: {ep+1}/{num_episodes}, Samples: {len(all_states)}")
    
    states = np.array(all_states)
    actions = np.array(all_actions, dtype=np.float32).reshape(-1, 1)
    costs = np.array(all_costs, dtype=np.float32).reshape(-1, 1)
    
    print(f"[RCP] Data Collection Complete: {len(states)} samples")
    print(f"[RCP] Cost Range: {costs.min():.2f} - {costs.max():.2f}")
    
    return states, actions, costs

# ==========================================
# Training
# ==========================================
def train_rcp(states, actions, costs, epochs=500, batch_size=32768, lr=1e-3):
    """Train RCP via Behavior Cloning with cost conditioning"""
    print(f"\n[RCP] Training ({epochs} epochs, batch={batch_size})...")
    
    # Prepare data
    X_state = torch.tensor(states, dtype=torch.float32)
    X_cost = torch.tensor(costs, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)
    dataset = TensorDataset(X_state, X_cost, Y)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Model
    model = RCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_state, batch_cost, batch_y in loader:
            batch_state = batch_state.to(device, non_blocking=True)
            batch_cost = batch_cost.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            pred = model(batch_state, batch_cost)
            loss = criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    # Save model
    model_path = os.path.join(current_dir, "model_rcp.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[RCP] Model saved: {model_path}")
    
    return model

# ==========================================
# Validation
# ==========================================
def validate(model):
    """Compare RCP (with target = SC*0.9) vs SC"""
    print(f"\n[RCP] Validation...")
    model.eval()
    sc = TimeAwareSC()
    
    print(f"{'Scenario':<20} | {'SC Cost':>10} | {'RCP Cost':>10} | {'Target':>10} | {'Winner':>8}")
    print("-" * 70)
    
    total_sc = 0
    total_rcp = 0
    victories = 0
    
    for scenario in BENCHMARK_SCENARIOS:
        # Run SC first to get baseline cost
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
        
        # Run RCP with target = SC * 0.9 (ask for 10% lower cost)
        target_cost = sc_cost * 0.9
        
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        
        with torch.no_grad():
            for _ in range(2000):
                max_t, active, load, min_time = physics.get_system_state()
                if active == 0: break
                
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
                
                target = torch.tensor([[target_cost]], dtype=torch.float32).to(device)
                
                action = model(state, target).item()
                power = action * 100.0
                physics.step(power, dt=0.5)
        
        rcp_cost = physics.total_cost
        total_rcp += rcp_cost
        
        winner = "RCP" if rcp_cost < sc_cost else "SC"
        if rcp_cost < sc_cost:
            victories += 1
        
        print(f"{scenario['name']:<20} | {sc_cost:>10.2f} | {rcp_cost:>10.2f} | {target_cost:>10.2f} | {winner:>8}")
    
    print("-" * 70)
    improvement = ((total_sc - total_rcp) / total_sc) * 100
    print(f"{'TOTAL':<20} | {total_sc:>10.2f} | {total_rcp:>10.2f} |            |")
    print(f"{'VICTORIES':<20} |            | {victories:>10}/10 |            |")
    print(f"{'IMPROVEMENT':<20} |            | {improvement:>+9.1f}% |            |")
    print("=" * 70)
    
    return victories, total_rcp, total_sc

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # 1. Collect Data
    states, actions, costs = collect_rcp_data(10000)
    
    # 2. Train RCP
    model = train_rcp(states, actions, costs, epochs=500)
    
    # 3. Validate
    validate(model)
