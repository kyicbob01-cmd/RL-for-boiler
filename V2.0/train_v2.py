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
# 2. Optimized Data Collection (GPU Batched)
# ==========================================
class BatchedBoilerPhysics:
    def __init__(self, num_envs, device):
        self.n = num_envs
        self.device = device
        
        # Physics Constants
        self.dt = 0.5
        self.heat_cap_boiler = 1500.0
        self.transfer_coeff = 30.0
        self.max_power = 100.0
        self.T_env = 25.0
        
        # State Tensors (N, )
        self.boiler_temp = torch.ones(self.n, device=self.device) * 25.0
        self.total_cost = torch.zeros(self.n, device=self.device)
        self.finished = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        
        # Unit State Tensors (N, 4) - Max 4 units
        self.num_units = 4
        # Features
        self.u_active = torch.zeros((self.n, self.num_units), dtype=torch.bool, device=self.device)
        self.u_target = torch.zeros((self.n, self.num_units), device=self.device)
        self.u_current = torch.ones((self.n, self.num_units), device=self.device) * 25.0
        self.u_weight = torch.zeros((self.n, self.num_units), device=self.device)
        self.u_duration = torch.zeros((self.n, self.num_units), device=self.device)
        self.u_thermal_mass = torch.ones((self.n, self.num_units), device=self.device) # Avoid div by zero
        
        # 0: Heating, 1: Holding, 2: Finished
        self.u_state = torch.zeros((self.n, self.num_units), dtype=torch.long, device=self.device) 

    def reset_random(self):
        # Vectorized random initialization
        self.boiler_temp[:] = 25.0
        self.total_cost[:] = 0.0
        self.finished[:] = False
        self.u_state[:] = 0 # Heating
        self.u_current[:] = 25.0
        
        # Randomize Tasks
        # Num tasks: Random 1-4
        n_tasks = torch.randint(1, 5, (self.n,), device=self.device)
        
        # Mask active units
        col_idx = torch.arange(4, device=self.device).expand(self.n, 4)
        self.u_active = col_idx < n_tasks.unsqueeze(1)
        
        # Random params
        self.u_target = torch.rand((self.n, 4), device=self.device) * (180 - 60) + 60
        self.u_duration = torch.rand((self.n, 4), device=self.device) * (500 - 50) + 50
        self.u_weight = torch.rand((self.n, 4), device=self.device) * (2000 - 100) + 100
        self.u_thermal_mass = self.u_weight * 1.2
        
        # Zero out inactive
        self.u_target[~self.u_active] = 0
        
    def get_state(self):
        # Calculate derived features for observation
        # Max Target (N, 1)
        # Active Count (N, 1)
        # Total Load (N, 1)
        
        max_t = self.u_target.max(dim=1)[0]
        
        # Active: State != 2 (Finished) AND Is Active Unit
        is_working = (self.u_state != 2) & self.u_active
        active_count = is_working.sum(dim=1).float()
        
        # Gap = Target - Current
        gap = torch.clamp(self.u_target - self.u_current, min=0)
        load = (gap * self.u_weight * is_working.float()).sum(dim=1)
        
        return self.boiler_temp, max_t, active_count, load

    def step(self, power_pct):
        # Input power_pct: (N,)
        current_kw = (power_pct / 100.0) * self.max_power
        heat_in_rate = current_kw * 50.0 * 0.9
        
        # Cost check
        kwh = current_kw * (self.dt / 3600.0)
        self.total_cost += kwh * 4.5
        
        # === Unit Physics ===
        # 1. State Transitions
        # Heating -> Holding if current >= target - 0.5
        to_hold = (self.u_state == 0) & (self.u_current >= self.u_target - 0.5)
        self.u_state[to_hold] = 1
        
        # Holding -> Finished if duration <= 0 (Logic: duration decrement)
        # Decrement duration if State == 1 (Holding) AND Temp >= Target - 3.0
        holding_ok = (self.u_state == 1) & (self.u_current >= self.u_target - 3.0)
        self.u_duration[holding_ok] -= self.dt
        
        to_finish = (self.u_state == 1) & (self.u_duration <= 0)
        self.u_state[to_finish] = 2
        
        # 2. Valve Logic
        # Open if (Heating OR Holding) AND (Current < Target - 0.5) AND (Boiler > Current)
        # Close if (Current > Target + 0.5)
        # Simplify: Standard thermostat
        needs_heat = (self.u_state <= 1) & self.u_active
        valve_open = needs_heat & (self.u_current < self.u_target - 0.5) & (self.boiler_temp.unsqueeze(1) > self.u_current)
        
        # 3. Heat Transfer
        # Transfer = Delta * Coeff
        delta = torch.clamp(self.boiler_temp.unsqueeze(1) - self.u_current, min=0)
        transfer_rate = delta * self.transfer_coeff * valve_open.float()
        
        # Loss
        loss_rate = (self.u_current - self.T_env) * 0.3 # While valve open
        loss_rate_closed = (self.u_current - self.T_env) * 0.2
        
        # Update Unit Temp
        # If open: (Transfer - Loss) / Mass
        # If closed: (-Loss) / Mass
        net_heat = torch.where(valve_open, transfer_rate - loss_rate, -loss_rate_closed)
        self.u_current += (net_heat / self.u_thermal_mass) * self.dt
        
        # 4. Boiler Physics
        total_transfer_out = transfer_rate.sum(dim=1)
        loss_boiler = (self.boiler_temp - self.T_env) * 4.0
        
        b_change = (heat_in_rate - loss_boiler - total_transfer_out) / self.heat_cap_boiler
        self.boiler_temp += b_change * self.dt
        
        # Check global finish
        # Finished if ALL active units are state 2
        # If a unit is NOT active, treat as finished.
        # So we check if ANY (Active AND State!=2) exists.
        any_working = ((self.u_state != 2) & self.u_active).any(dim=1)
        self.finished = ~any_working

        return self.finished

def collect_data(num_episodes=20000):
    BATCH_SIZE = 2048 # Maximize GPU
    iterations = (num_episodes // BATCH_SIZE) + 1
    
    print(f"Collecting Data on GPU (Batch={BATCH_SIZE}, Iters={iterations})...")
    
    teacher = RCPTeacher()
    all_data = [] # List of dicts (CPU) - Try to keep on CPU to save GPU VRAM
    
    env = BatchedBoilerPhysics(BATCH_SIZE, device)
    
    for it in range(iterations):
        env.reset_random()
        
        # Trajectory Buffers
        # We need to store (S, A, C) for 2000 steps for B batches
        # Pre-allocate large tensors on Host memory to avoid GPU OOM
        # Actually 2048 * 2000 * floats is fine (16MB).
        
        # We'll accumulate lists and stack later
        states_list = []
        actions_list = []
        
        active_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device=device)
        traj_lengths = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        
        prev_temp = env.boiler_temp.clone()
        noise = 0.05
        
        for step in range(2000):
            # 1. Get State
            b_temp, max_t, active, load = env.get_state()
            rate = b_temp - prev_temp
            prev_temp = b_temp.clone()
            
            # (N, 5)
            obs = torch.stack([
                b_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0
            ], dim=1)
            
            # 2. Teacher Inference (Batched)
            # Ambitious Target
            target_val = torch.ones((BATCH_SIZE, 1), device=device) * 5.0
            with torch.no_grad():
                 # Teacher model expects (N, 5)
                 # Teacher decide fn was single item. We invoke model directly.
                 # RCPModel forward: cat(state, target/50)
                 preds = teacher.model(obs, target_val)
                 
                 # Add noise
                 noise_t = torch.randn_like(preds) * noise
                 actions = torch.clamp(preds + noise_t, 0.0, 1.0)
                 
            power_vals = actions.squeeze() * 100.0
            power_vals = torch.where(active_mask, power_vals, torch.zeros_like(power_vals))
            
            # 3. Step
            dones = env.step(power_vals)
            
            # 4. Store
            states_list.append(obs.cpu()) # Move to CPU to save VRAM? Or keep GPU. CPU is safer for lists.
            actions_list.append(actions.cpu())
            
            # Update Active Mask
            # If done, mask out. But we must continue stepping others.
            # Just set power to 0 for done envs (handled above).
            # But we record length.
            just_finished = dones & active_mask
            traj_lengths[just_finished] = step + 1
            active_mask = active_mask & ~dones
            
            if (~active_mask).all():
                break
                
        # End of Episode
        # Process results
        final_costs = env.total_cost.cpu().numpy()
        lens = traj_lengths.cpu().numpy()
        
        # Convert lists of tensors to Tensor (Steps, B, F)
        S = torch.stack(states_list).numpy() # (T, B, 5)
        A = torch.stack(actions_list).numpy() # (T, B, 1)
        
        # Unpack into episodes
        for i in range(BATCH_SIZE):
            length = lens[i]
            if length == 0: length = 2000 # Max steps if never finished
            
            traj = []
            c = final_costs[i]
            
            s_ep = S[:length, i, :]
            a_ep = A[:length, i, :]
            
            for t in range(length):
                traj.append({
                    'state': s_ep[t],
                    'action': a_ep[t][0], # scalar
                    'final_cost': c
                })
            
            all_data.append({
                'trajectory': traj,
                'cost': c
            })
            
        print(f"  Batch {it+1}/{iterations} Done. Total Samples: {len(all_data)}")
        
    return all_data[:num_episodes]

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
