"""
V2.0 Showdown: Student vs Teacher vs Rules
hypothesis: V2.0 (Student) > V1.0 (Teacher) > SmartController
"""

import torch
import torch.nn as nn
import os
import sys
import numpy as np

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Shared Architecture
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
# Controllers
# ==========================================
class SmartControllerRules:
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

def load_model(path):
    if not os.path.exists(path):
        print(f"Warning: Model not found at {path}")
        return None
    model = RCPolicy().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# ==========================================
# Simulation
# ==========================================
def run_simulation(controller_type, model=None, scenario=None):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    sc = SmartControllerRules()
    
    for _ in range(2000):
        max_t, active, load = physics.get_system_state()
        if active == 0: break
        
        power = 0.0
        if controller_type == "SC":
            power = sc.decide(physics.boiler_temp, physics.units)
        else: # RCP
            obs = torch.tensor([[
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                0.0, load / 2000000.0
            ]], dtype=torch.float32).to(device)
            target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
            with torch.no_grad():
                power = model(obs, target).item() * 100.0
        
        physics.step(power, dt=0.5)
        
    return physics.total_cost

# ==========================================
# Main Evaluation
# ==========================================
def evaluate_all():
    print("="*80)
    print("V2.0 FINAL SHOWDOWN: Student vs Teacher vs Rules")
    print("="*80)
    
    # Load Models
    teacher_path = os.path.join(current_dir, "..", "V1.0", "model.pth")
    student_path = os.path.join(current_dir, "model_v2.pth")
    
    teacher = load_model(teacher_path)
    student = load_model(student_path)
    
    if student is None:
        print("CRITICAL: model_v2.pth NOT FOUND. Please run train_v2.py first.")
        return

    print(f"{'Scenario':<20} | {'SC (Rules)':<10} | {'V1.0 (Teacher)':<15} | {'V2.0 (Student)':<15} | {'Winner':<10}")
    print("-" * 85)
    
    scores = {"SC": 0, "V1.0": 0, "V2.0": 0}
    wins = {"SC": 0, "V1.0": 0, "V2.0": 0}
    
    for scenario in BENCHMARK_SCENARIOS:
        c_sc = run_simulation("SC", scenario=scenario)
        c_v1 = run_simulation("V1.0", model=teacher, scenario=scenario) if teacher else 999
        c_v2 = run_simulation("V2.0", model=student, scenario=scenario)
        
        # Determine Winner (Lowest Cost)
        results = [("SC", c_sc), ("V1.0", c_v1), ("V2.0", c_v2)]
        results.sort(key=lambda x: x[1])
        winner = results[0][0]
        
        # Update Stats
        scores["SC"] += c_sc
        scores["V1.0"] += c_v1
        scores["V2.0"] += c_v2
        wins[winner] += 1
        
        print(f"{scenario['name']:<20} | {c_sc:>10.2f} | {c_v1:>15.2f} | {c_v2:>15.2f} | {winner:<10}")
        
    print("-" * 85)
    print(f"TOTAL COST           | {scores['SC']:>10.2f} | {scores['V1.0']:>15.2f} | {scores['V2.0']:>15.2f}")
    print(f"TOTAL WINS           | {wins['SC']:>10} | {wins['V1.0']:>15} | {wins['V2.0']:>15}")
    print("="*80)

    # Validation Logic
    if scores['V2.0'] < scores['V1.0'] and scores['V2.0'] < scores['SC']:
        print("✅ SUCCESS: V2.0 (Student) is the new State-of-the-Art!")
    elif scores['V2.0'] < scores['SC']:
        print("⚠️ PARTIAL: V2.0 beat Rules but failed to beat Teacher.")
    else:
        print("❌ FAILURE: V2.0 failed to improve.")

if __name__ == "__main__":
    evaluate_all()
