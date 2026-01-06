"""
Return-Conditioned Policy V2.0 (Self-Evolution)
Teacher: V1.0 Model (Ambitious Mode)
Student: V2.0 Model (Learns from Elite Trajectories)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 0. Model Architecture (Shared)
# ==========================================
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

# ==========================================
# 1. Teacher Agent (V1.0)
# ==========================================
class RCPTeacher:
    def __init__(self):
        self.model = RCPolicy().to(device)
        # Load V1.0 Model
        v1_path = os.path.join(current_dir, "..", "V1.0", "model.pth")
        if os.path.exists(v1_path):
            self.model.load_state_dict(torch.load(v1_path, map_location=device))
            self.model.eval()
            print(f">>> Teacher Loaded: {v1_path}")
        else:
            raise FileNotFoundError(f"Teacher model not found at {v1_path}")

    def decide(self, physics, noise_scale=0.05):
        """
        Teacher Decision with Exploration Noise
        noise_scale: Standard deviation of Gaussian noise added to action
        """
        max_t, active, load = physics.get_system_state()
        
        obs = torch.tensor([[
            physics.boiler_temp / 300.0,
            max_t / 300.0,
            active / 4.0,
            0.0, # Rate assumption
            load / 2000000.0
        ]], dtype=torch.float32).to(device)
        
        # Ambitious Target for Teacher
        target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            action = self.model(obs, target).item()
            
        # Add Exploration Noise
        if noise_scale > 0:
            action += np.random.normal(0, noise_scale)
            action = np.clip(action, 0.0, 1.0)
            
        return action * 100.0

# ==========================================
# 2. Data Collection (Self-Play)
# ==========================================
def collect_data(num_episodes=20000):
    print(f"Collecting Self-Play Data ({num_episodes} episodes)...")
    teacher = RCPTeacher()
    all_data = []
    
    # Extended Scenarios (Benchmark + Random Variations)
    scenarios = BENCHMARK_SCENARIOS * 5 
    
    for ep in range(num_episodes):
        # Scenario Selection
        if ep < len(scenarios):
            base_scenario = scenarios[ep]
        else:
            # Generate Random Scenario for Diversity
            import random
            base_scenario = {"name": f"Rand_{ep}", "tasks": []}
            for _ in range(random.randint(1, 4)):
                base_scenario["tasks"].append({
                    "name": "Unit", 
                    "target": random.uniform(80, 180), 
                    "duration": random.uniform(50, 400), 
                    "weight": random.uniform(100, 2000)
                })

        physics = BoilerPhysics()
        physics.reset()
        for task in base_scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        trajectory = []
        prev_temp = physics.boiler_temp
        task_completed = False
        
        noise = 0.05 
        
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
            
            # Teacher Acts
            power = teacher.decide(physics, noise_scale=noise)
            
            trajectory.append({
                'state': state,
                'action': power / 100.0
            })
            
            physics.step(power, dt=0.5)
        
        if task_completed:
            final_cost = physics.total_cost
            for t in trajectory:
                t['final_cost'] = final_cost
            
            all_data.append({
                'trajectory': trajectory,
                'cost': final_cost
            })
        
        if (ep + 1) % 2000 == 0:
            print(f"  Progress: {ep+1}/{num_episodes}")
            
    return all_data

# ==========================================
# 3. Elite Filtering (Distillation)
# ==========================================
def filter_elite_data(raw_data, quantile=0.3):
    """Keep top X% most efficient episodes"""
    costs = [d['cost'] for d in raw_data]
    threshold = np.quantile(costs, quantile)
    avg_cost = np.mean(costs)
    
    elite_samples = []
    for d in raw_data:
        if d['cost'] <= threshold:
            elite_samples.extend(d['trajectory'])
            
    print(f"\nFiltering Complete:")
    print(f"  Original Avg Cost: {avg_cost:.2f}")
    print(f"  Elite Threshold (Top {quantile*100}%): {threshold:.2f}")
    print(f"  Elite Samples: {len(elite_samples)} steps")
    
    return elite_samples

# ==========================================
# 4. Training Loop
# ==========================================
def train(data, epochs=500):
    print(f"\nTraining Student V2.0 ({epochs} Epochs)...")
    
    states = torch.tensor(np.array([d['state'] for d in data], dtype=np.float32)).to(device)
    actions = torch.tensor(np.array([[d['action']] for d in data], dtype=np.float32)).to(device)
    costs = torch.tensor(np.array([[d['final_cost']] for d in data], dtype=np.float32)).to(device)
    
    BATCH_SIZE = 32768
    num_samples = states.shape[0]
    
    student = RCPolicy().to(device)
    torch.set_float32_matmul_precision('high')

    opt = optim.Adam(student.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples, device=device)
        total_loss, batches = 0, 0
        
        for start in range(0, num_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            pred = student(states[idx], costs[idx])
            loss = loss_fn(pred, actions[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); batches += 1
        
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/batches:.4f}")
    
    student.cpu()
    torch.save(student.state_dict(), "model_v2.pth")
    print("Student V2.0 Saved: model_v2.pth")
    return student.to(device)

def validate(model):
    print("\nValidating Student V2.0...")
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
                obs = torch.tensor([[physics.boiler_temp/300.0, max_t/300.0, active/4.0, 0.0, load/2000000.0]], dtype=torch.float32).to(device)
                target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
                power = model(obs, target).item() * 100.0
                physics.step(power, dt=0.5)
        print(f"  {scenario['name']}: {physics.total_cost:.2f}")

if __name__ == "__main__":
    # 1. Self-Play
    raw_data = collect_data(20000)
    
    # 2. Distillation
    elite_data = filter_elite_data(raw_data, quantile=0.3)
    
    # 3. Train Student
    student = train(elite_data, epochs=500)
    
    # 4. Quick Check
    validate(student)
