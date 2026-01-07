"""
Return-Conditioned Policy V2.3 (Direct CPU Cloning)
Teacher: V1.0 Model (RCP)
Student: V2.3 Model (Unconditional - State Only)

Key Fix: Use original BoilerPhysics (CPU) for data collection to ensure
physics match between training and evaluation.
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
# 0. Model Architectures
# ==========================================
class RCPolicy(nn.Module):
    """V1.0 Architecture (for loading Teacher)"""
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

class UnconditionalPolicy(nn.Module):
    """V2.3 Architecture: State -> Action (No Cost Conditioning)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 1024), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )
    
    def forward(self, state):
        return self.net(state)

# ==========================================
# 1. Teacher Agent (V1.0)
# ==========================================
class RCPTeacher:
    def __init__(self):
        self.model = RCPolicy().to(device)
        v1_path = os.path.join(current_dir, "..", "V1.0", "model.pth")
        if os.path.exists(v1_path):
            self.model.load_state_dict(torch.load(v1_path, map_location=device))
            self.model.eval()
            print(f">>> Teacher Loaded: {v1_path}")
        else:
            raise FileNotFoundError(f"Teacher model not found at {v1_path}")

    def decide(self, physics):
        """Deterministic Teacher Decision (No Noise)"""
        max_t, active, load = physics.get_system_state()
        
        obs = torch.tensor([[
            physics.boiler_temp / 300.0,
            max_t / 300.0,
            active / 4.0,
            0.0,
            load / 2000000.0
        ]], dtype=torch.float32).to(device)
        
        target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            action = self.model(obs, target).item()
            
        return action * 100.0

# ==========================================
# 2. Data Collection (CPU-Based)
# ==========================================
def collect_data(num_episodes=5000):
    print(f"Collecting Data (CPU Physics, {num_episodes} episodes)...")
    teacher = RCPTeacher()
    all_data = []
    
    # Use BENCHMARK_SCENARIOS + random variations
    scenarios = BENCHMARK_SCENARIOS * (num_episodes // len(BENCHMARK_SCENARIOS) + 1)
    
    for ep in range(num_episodes):
        # Scenario Selection
        if ep < len(BENCHMARK_SCENARIOS) * 10:
            # First 100 episodes: Use benchmark scenarios
            scenario = scenarios[ep % len(BENCHMARK_SCENARIOS)]
        else:
            # Rest: Random scenario for diversity
            import random
            scenario = {"name": f"Rand_{ep}", "tasks": []}
            for _ in range(random.randint(1, 4)):
                scenario["tasks"].append({
                    "name": f"U{random.randint(1,99)}", 
                    "target": random.uniform(60, 180), 
                    "duration": random.uniform(50, 500), 
                    "weight": random.uniform(100, 2000)
                })

        # Run simulation with ORIGINAL CPU Physics
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        trajectory = []
        prev_temp = physics.boiler_temp
        
        for step in range(2000):
            max_t, active, load = physics.get_system_state()
            if active == 0: 
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
            
            # Teacher Decision (Deterministic)
            power = teacher.decide(physics)
            
            trajectory.append({
                'state': state,
                'action': power / 100.0
            })
            
            physics.step(power, dt=0.5)
        
        # Only keep completed episodes
        if physics.get_system_state()[1] == 0:
            all_data.extend(trajectory)
        
        if (ep + 1) % 500 == 0:
            print(f"  Progress: {ep+1}/{num_episodes}, Samples: {len(all_data)}")
            
    print(f"Data Collection Complete: {len(all_data)} samples")
    return all_data

# ==========================================
# 3. Training Loop (GPU)
# ==========================================
def train(data, epochs=500):
    print(f"\nTraining Student V2.3 (Unconditional, {epochs} Epochs)...")
    
    states = torch.tensor(np.array([d['state'] for d in data], dtype=np.float32)).to(device)
    actions = torch.tensor(np.array([[d['action']] for d in data], dtype=np.float32)).to(device)
    
    BATCH_SIZE = 32768
    num_samples = states.shape[0]
    print(f"  Samples: {num_samples}, Batch Size: {BATCH_SIZE}")
    
    student = UnconditionalPolicy().to(device)
    torch.set_float32_matmul_precision('high')

    opt = optim.Adam(student.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples, device=device)
        total_loss, batches = 0, 0
        
        for start in range(0, num_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            pred = student(states[idx])
            loss = loss_fn(pred, actions[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); batches += 1
        
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/batches:.6f}")
    
    student.cpu()
    model_save_path = os.path.join(current_dir, "model_v2.pth")
    torch.save(student.state_dict(), model_save_path)
    print(f"Student V2.3 Saved: {model_save_path}")
    return student.to(device)

# ==========================================
# 4. Validation
# ==========================================
def validate(model):
    print("\nValidating Student V2.3...")
    model.eval()
    for scenario in BENCHMARK_SCENARIOS[:5]:
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        with torch.no_grad():
            for _ in range(2000):
                max_t, active, load = physics.get_system_state()
                if active == 0: break
                rate = physics.boiler_temp - prev_temp
                prev_temp = physics.boiler_temp
                obs = torch.tensor([[
                    physics.boiler_temp/300.0, 
                    max_t/300.0, 
                    active/4.0, 
                    rate, 
                    load/2000000.0
                ]], dtype=torch.float32).to(device)
                power = model(obs).item() * 100.0
                physics.step(power, dt=0.5)
        print(f"  {scenario['name']}: {physics.total_cost:.2f}")

if __name__ == "__main__":
    # 1. Collect Data (CPU Physics)
    data = collect_data(5000)
    
    # 2. Train Student (GPU)
    student = train(data, epochs=500)
    
    # 3. Quick Check
    validate(student)
