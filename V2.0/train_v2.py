"""
Return-Conditioned Policy Training (V2)
Trains RCP model using offline RL on varied trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SmartController:
    # Baseline controller for data generation
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
    
    def forward(self, state, target_cost):
        cost_norm = target_cost / 50.0
        x = torch.cat([state, cost_norm], dim=-1)
        return self.net(x)

def collect_data(num_episodes=10000):
    print(f"Collecting Data ({num_episodes})...")
    controller = SmartController()
    all_data = []
    
    strategies = [
        ("Baseline", lambda p: p),
        ("Aggressive", lambda p: min(100, p * 1.3)),
        ("Conservative", lambda p: p * 0.7),
        ("SuperSaver", lambda p: p * 0.4 if p < 50 else p * 0.8)
    ]
    
    low_temp_scenarios = [
        {"name": "Train_Low_1", "tasks": [{"name": "A", "target": 70.0, "duration": 100.0, "weight": 500.0}]},
        {"name": "Train_Low_2", "tasks": [{"name": "B", "target": 80.0, "duration": 120.0, "weight": 500.0}]},
        {"name": "Train_Low_3", "tasks": [{"name": "C", "target": 90.0, "duration": 150.0, "weight": 500.0}]},
    ]
    total_scenarios = BENCHMARK_SCENARIOS + low_temp_scenarios * 3
    
    for ep in range(num_episodes):
        scenario = total_scenarios[ep % len(total_scenarios)]
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        # Strategy Mixing
        rand_val = np.random.rand()
        if rand_val < 0.25: _, modifier = strategies[0]
        elif rand_val < 0.65: _, modifier = strategies[1]
        elif rand_val < 0.90: _, modifier = strategies[2]
        else: _, modifier = strategies[3]
        
        trajectory = []
        prev_temp = physics.boiler_temp
        task_completed = False
        
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
            
            base = controller.decide(physics.boiler_temp, physics.units)
            power = max(0, min(100, modifier(base)))
            
            trajectory.append({'state': state, 'action': power / 100.0})
            physics.step(power, dt=0.5)
        
        if task_completed:
            cost = physics.total_cost
            for t in trajectory:
                t['final_cost'] = cost
                all_data.append(t)
        
        if (ep + 1) % 1000 == 0: print(f"  Progress: {ep+1}")
    
    print(f"Data Collection Complete: {len(all_data)} samples.")
    return all_data

def train(data, epochs=500):
    print(f"\nTraining ({epochs} Epochs)...")
    
    states = torch.tensor(np.array([d['state'] for d in data], dtype=np.float32)).to(device)
    actions = torch.tensor(np.array([[d['action']] for d in data], dtype=np.float32)).to(device)
    costs = torch.tensor(np.array([[d['final_cost']] for d in data], dtype=np.float32)).to(device)
    
    BATCH_SIZE = 32768
    num_samples = states.shape[0]
    
    model = RCPolicy().to(device)
    torch.set_float32_matmul_precision('high')

    opt = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples, device=device)
        total_loss, batches = 0, 0
        
        for start in range(0, num_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            pred = model(states[idx], costs[idx])
            loss = loss_fn(pred, actions[idx])
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            batches += 1
        
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/batches:.4f}")
    
    model.cpu()
    torch.save(model.state_dict(), "model.pth")
    print("Model Saved.")
    return model.to(device)

def validate(model):
    print("\nValidating...")
    model.eval()
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
                ]], dtype=torch.float32).to(device)
                
                target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
                power = model(state, target).item() * 100.0
                physics.step(power, dt=0.5)
        
        print(f"  {scenario['name']}: {physics.total_cost:.2f}")

if __name__ == "__main__":
    data = collect_data(10000)
    model = train(data, epochs=500)
    validate(model)
