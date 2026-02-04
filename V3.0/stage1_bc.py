import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 1] Device: {device}")

def collect_expert_data(num_episodes=5000):
    print(f"[Stage 1] Collecting Expert Data ({num_episodes} episodes)...")
    
    sc = TimeAwareSC()
    all_states = []
    all_actions = []
    
    for ep in range(num_episodes):
        if ep < len(BENCHMARK_SCENARIOS) * 20:
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
    
    states = np.array(all_states)
    actions = np.array(all_actions, dtype=np.float32).reshape(-1, 1)
    
    print(f"[Stage 1] Data Collection Complete: {len(states)} samples")
    return states, actions

def train_bc(states, actions, epochs=300, batch_size=32768, lr=1e-3):
    print(f"\n[Stage 1] Training Behavior Cloning ({epochs} epochs, batch={batch_size})...")
    
    X = torch.tensor(states, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)
    dataset = TensorDataset(X, Y)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    model = Policy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    model_path = os.path.join(current_dir, "model_bc.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[Stage 1] Model saved: {model_path}")
    
    return model

def validate(model):
    print(f"\n[Stage 1] Validation...")
    model.eval()
    sc = TimeAwareSC()
    
    results = []
    
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
                
                action = model(state).item()
                power = action * 100.0
                physics.step(power, dt=0.5)
        
        bc_cost = physics.total_cost
        
        diff_pct = abs(bc_cost - sc_cost) / sc_cost * 100
        match = "OK" if diff_pct < 5.0 else "DIFF"
        
        results.append({
            'name': scenario['name'],
            'sc_cost': sc_cost,
            'bc_cost': bc_cost,
            'diff_pct': diff_pct,
            'match': match
        })
        
        print(f"  {scenario['name']:<20}: SC={sc_cost:.2f}, BC={bc_cost:.2f}, Diff={diff_pct:.1f}% [{match}]")
    
    ok_count = sum(1 for r in results if r['match'] == 'OK')
    print(f"\n[Stage 1] Validation Complete: {ok_count}/10 scenarios within 5%")
    
    if ok_count >= 8:
        print("[Stage 1] SUCCESS: Ready for Stage 2 PPO Fine-tuning!")
        return True
    else:
        print("[Stage 1] WARNING: BC model may need more training")
        return False

if __name__ == "__main__":
    states, actions = collect_expert_data(5000)
    model = train_bc(states, actions, epochs=300)
    validate(model)
