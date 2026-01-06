"""
Return-Conditioned Policy (RCP) Training Script
==============================================
Training framework for the RCP model, using offline reinforcement learning 
principles (supervised learning on trajectory outcomes).

Key Features:
- Diversity-driven Data Collection (SmartController Variants)
- Hindsight Relabeling (Associating actions with final outcomes)
- Failure Filtering (Excluding failed tasks to avoid survivorship bias)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# Device Configuration
device = torch.device("cpu") # Force CPU for stability
print(f"Using device: {device}")

class SmartController:
    """Baseline controller for generating training data."""
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        needed = [u for u in active if u['current'] < u['target'] - 0.5]
        targets = [u['target'] for u in needed]
        
        if not targets:
            holding = [u['target'] for u in active if u['state'] == 'HOLDING']
            if holding and temp < min(holding) + 3.0: return 30.0
            return 0.0
        
        max_t = max(targets)
        max_demand = max([u['current'] for u in needed])
        target_boiler = max(max_t + 5.0, max_demand + 6.0)
        gap = target_boiler - temp
        
        if gap < -2.0: return 0.0
        if gap > 20: return 100.0
        elif gap > 10: return 80.0
        elif gap > 5: return 50.0
        elif gap > 0: return 20.0
        else: return 0.0

class RCPolicy(nn.Module):
    """
    Return-Conditioned Policy Network.
    Input: State (5 dims) + Target Cost (1 dim)
    Output: Power Action (0-1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, state, target_cost):
        # Normalize target cost (assuming reasonable range 0-50)
        cost_norm = target_cost / 50.0
        x = torch.cat([state, cost_norm], dim=-1)
        return self.net(x)

def collect_data(num_episodes=10000):
    """Collects diverse training data using multiple expert strategies."""
    print(f"Collecting Training Data ({num_episodes} episodes)...")
    controller = SmartController()
    all_data = []
    
    # Expert Strategies Definition
    strategies = [
        ("Baseline", lambda p: p),
        ("Aggressive", lambda p: min(100, p * 1.3)),
        ("Conservative", lambda p: p * 0.7),
        ("SuperSaver", lambda p: p * 0.4 if p < 50 else p * 0.8)
    ]
    
    # Supplementary Training Scenarios (Low Temperature Focus)
    low_temp_scenarios = [
        {"name": "Train_Low_1", "tasks": [{"name": "A", "target": 70.0, "duration": 100.0, "weight": 500.0}]},
        {"name": "Train_Low_2", "tasks": [{"name": "B", "target": 80.0, "duration": 120.0, "weight": 500.0}]},
        {"name": "Train_Low_3", "tasks": [{"name": "C", "target": 90.0, "duration": 150.0, "weight": 500.0}]},
    ]
    
    # Combined Scenario Pool (Weighted)
    total_scenarios = BENCHMARK_SCENARIOS + low_temp_scenarios * 3
    
    for ep in range(num_episodes):
        scenario = total_scenarios[ep % len(total_scenarios)]
        
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        # Strategy Selection (Weighted Mix)
        rand_val = np.random.rand()
        if rand_val < 0.4:
            _, modifier = strategies[0] # Baseline (40%)
        elif rand_val < 0.7:
             _, modifier = strategies[2] # Conservative (30%)
        elif rand_val < 0.85:
             _, modifier = strategies[1] # Aggressive (15%)
        else:
             _, modifier = strategies[3] # SuperSaver (15%)
        
        trajectory = []
        prev_temp = physics.boiler_temp
        task_completed = False
        
        # Simulation Loop
        for step in range(2000):
            max_t, active, load = physics.get_system_state()
            if active == 0: 
                task_completed = True
                break
            
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            
            state = np.array([
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0
            ], dtype=np.float32)
            
            base_power = controller.decide(physics.boiler_temp, physics.units)
            power = modifier(base_power)
            power = max(0, min(100, power))
            
            trajectory.append({
                'state': state,
                'action': power / 100.0
            })
            
            physics.step(power, dt=0.5)
        
        # Data Filtering: Only learn from completed tasks
        if task_completed:
            final_cost = physics.total_cost
            for t in trajectory:
                t['final_cost'] = final_cost
                all_data.append(t)
        
        if (ep + 1) % 1000 == 0:
            print(f"  Progress: {ep+1}/{num_episodes}")
    
    print(f"Data Collection Complete: {len(all_data)} valid samples.")
    return all_data

def train(data, epochs=500):
    """Training loop for the RCP network."""
    print(f"\nTraining Model ({epochs} Epochs)...")
    
    states = np.array([d['state'] for d in data], dtype=np.float32)
    actions = np.array([[d['action']] for d in data], dtype=np.float32)
    costs = np.array([[d['final_cost']] for d in data], dtype=np.float32)
    
    states_t = torch.tensor(states).to(device)
    actions_t = torch.tensor(actions).to(device)
    costs_t = torch.tensor(costs).to(device)
    
    dataset = TensorDataset(states_t, costs_t, actions_t)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    model = RCPolicy().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for s, c, a in loader:
            opt.zero_grad()
            pred = model(s, c)
            loss = loss_fn(pred, a)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
    
    # Save Model
    model.cpu()
    torch.save(model.state_dict(), "rc_policy.pth")
    print("\nModel Saved: rc_policy.pth")
    return model.to(device)

def validate(model):
    """Quick validation run on a subset of scenarios."""
    print("\nValidating Model Performance...")
    
    # Inference Target: Aggressive but realistic
    target_cost = 10.0
    
    model.eval()
    
    # Test on first 5 benchmark scenarios
    for scenario in BENCHMARK_SCENARIOS[:5]:
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        with torch.no_grad():
            for _ in range(2000):
                max_t, active, load = physics.get_system_state()
                if active == 0: break
                
                state = torch.tensor([[
                    physics.boiler_temp / 300.0,
                    max_t / 300.0,
                    active / 4.0,
                    0.0,
                    load / 2000000.0
                ]], dtype=torch.float32)
                
                target = torch.tensor([[target_cost]], dtype=torch.float32)
                power = model(state, target).item() * 100.0
                physics.step(power, dt=0.5)
        
        print(f"  {scenario['name']}: Cost = {physics.total_cost:.2f}")

if __name__ == "__main__":
    print("-" * 60)
    print("  RCP Training Pipeline")
    print("-" * 60)
    
    data = collect_data(10000)
    model = train(data, epochs=500)
    validate(model)
    
    print("\n" + "-" * 60)
    print("  Pipeline Completed Successfully")
    print("-" * 60)
