import torch
import torch.nn as nn
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class SmartController:
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

def collect_data(num_episodes=10000, max_steps=2000):
    print(f"Collecting Data ({num_episodes} episodes)...")
    all_data = []
    sc = SmartController()
    
    for ep in range(num_episodes):
        if ep < len(BENCHMARK_SCENARIOS) * 20:
            scenario = BENCHMARK_SCENARIOS[ep % len(BENCHMARK_SCENARIOS)]
            physics = BoilerPhysics()
            physics.reset()
            for task in scenario["tasks"]:
                physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        else:
            physics = BoilerPhysics()
            physics.reset_random_scenario()
        
        if not physics.units: continue
        
        trajectory = []
        prev_temp = physics.boiler_temp
        
        for step in range(max_steps):
            max_t, active, load = physics.get_system_state()
            if active == 0: break
            
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            
            state = np.array([
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0
            ], dtype=np.float32)
            
            power = sc.decide(physics.boiler_temp, physics.units)
            
            trajectory.append({
                'state': state,
                'action': power / 100.0,
                'power_raw': power
            })
            
            physics.step(power, dt=0.5)
        
        if physics.get_system_state()[1] == 0:
            final_cost = physics.total_cost
            for i, sample in enumerate(trajectory):
                sample['episode_cost'] = final_cost
            all_data.extend(trajectory)
        
        if (ep + 1) % 1000 == 0:
            print(f"  Progress: {ep+1}/{num_episodes}, Samples: {len(all_data)}")

    print(f"Data Collection Complete: {len(all_data)} samples")
    return all_data

def train(data, epochs=500):
    print(f"\nTraining V1.0 RCP ({epochs} epochs)...")
    
    states = torch.tensor(np.array([d['state'] for d in data], dtype=np.float32)).to(device)
    episode_costs = torch.tensor(np.array([[d['episode_cost']] for d in data], dtype=np.float32)).to(device)
    actions = torch.tensor(np.array([[d['action']] for d in data], dtype=np.float32)).to(device)
    
    BATCH_SIZE = 32768
    num_samples = states.shape[0]
    print(f"  Samples: {num_samples}, Batch Size: {BATCH_SIZE}")
    
    model = RCPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        indices = torch.randperm(num_samples, device=device)
        total_loss, batches = 0, 0
        
        for start in range(0, num_samples, BATCH_SIZE):
            idx = indices[start : start + BATCH_SIZE]
            pred = model(states[idx], episode_costs[idx])
            loss = loss_fn(pred, actions[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); batches += 1
        
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/batches:.6f}")
    
    model.cpu()
    model_save_path = os.path.join(current_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved: {model_save_path}")
    return model.to(device)

def validate(model):
    print("\nValidating V1.0 RCP...")
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
                obs = torch.tensor([[
                    physics.boiler_temp/300.0, 
                    max_t/300.0, 
                    active/4.0, 
                    0.0, 
                    load/2000000.0
                ]], dtype=torch.float32).to(device)
                target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
                power = model(obs, target).item() * 100.0
                physics.step(power, dt=0.5)
        print(f"  {scenario['name']}: {physics.total_cost:.2f}")

if __name__ == "__main__":
    data = collect_data(10000)
    model = train(data, epochs=500)
    validate(model)
