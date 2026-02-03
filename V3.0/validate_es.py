"""
V3.0 ES Model Validation (Fixed)
Verify that the ES model completes all tasks correctly
"""

import os
import sys
import torch
import torch.nn as nn

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC

# ==========================================
# Policy Network (must match train_es.py)
# ==========================================
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

# ==========================================
# Validation Function (matches training evaluation)
# ==========================================
def validate_scenario(model, scenario):
    """Run scenario - using same logic as training"""
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    prev_temp = physics.boiler_temp
    power_history = []
    temp_history = []
    
    with torch.no_grad():
        for _ in range(2000):
            max_t, active, load, min_time = physics.get_system_state()
            if active == 0:
                break
            
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            
            # CPU-only, same as training!
            state = torch.tensor([[
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0,
                min_time / 500.0
            ]], dtype=torch.float32)
            
            action = model(state).item()
            power = action * 100.0
            
            power_history.append(power)
            temp_history.append(physics.boiler_temp)
            
            physics.step(power, dt=0.5)
    
    # Build unit status
    unit_status = {}
    for uid, unit in physics.units.items():
        unit_status[unit['name']] = {
            'target': unit['target'],
            'hold_req': unit['duration_total'],
            'hold_left': unit['duration_left'],
            'completed': unit['state'] == 'FINISHED',
            'state': unit['state']
        }
    
    return {
        'cost': physics.total_cost,
        'units': unit_status,
        'avg_power': sum(power_history) / len(power_history) if power_history else 0,
        'max_temp': max(temp_history) if temp_history else 0
    }

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Model priority: final > best
    es_final = os.path.join(current_dir, "model_es_final.pth")
    es_best = os.path.join(current_dir, "model_es_best.pth")
    
    if os.path.exists(es_final):
        model_path = es_final
    elif os.path.exists(es_best):
        model_path = es_best
    else:
        print("ERROR: No ES model found!")
        sys.exit(1)
    
    print("="*80)
    print("V3.0 ES MODEL VALIDATION")
    print("="*80)
    
    # Load model on CPU (same as training)
    model = Policy()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded: {model_path}")
    
    # Run SC for comparison
    sc = TimeAwareSC()
    
    total_es = 0
    total_sc = 0
    all_pass = True
    
    print(f"\n{'Scenario':<15} | {'ES Cost':>10} | {'SC Cost':>10} | {'Diff':>8} | {'Status':<15}")
    print("-"*70)
    
    for scenario in BENCHMARK_SCENARIOS:
        # ES evaluation
        es_result = validate_scenario(model, scenario)
        
        # SC evaluation
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
        
        total_es += es_result['cost']
        total_sc += sc_cost
        
        # Check all units completed
        all_completed = all(u['completed'] for u in es_result['units'].values())
        status = "PASS" if all_completed else "FAIL"
        if not all_completed:
            all_pass = False
        
        diff = ((sc_cost - es_result['cost']) / sc_cost) * 100
        print(f"{scenario['name']:<15} | {es_result['cost']:>10.2f} | {sc_cost:>10.2f} | {diff:>+7.1f}% | [{status}]")
    
    print("-"*70)
    improvement = ((total_sc - total_es) / total_sc) * 100
    print(f"{'TOTAL':<15} | {total_es:>10.2f} | {total_sc:>10.2f} | {improvement:>+7.1f}%")
    
    print("\n" + "="*80)
    if all_pass:
        print("[PASS] All tasks completed successfully!")
        print(f"ES achieves {improvement:+.1f}% improvement over SC")
    else:
        print("[FAIL] Some tasks were not completed!")
    print("="*80)
