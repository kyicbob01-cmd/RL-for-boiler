import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from train import RCPolicy, SmartController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation(controller_type, model=None, scenario=None):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    sc = SmartController()
    
    for _ in range(2000):
        max_t, active, load = physics.get_system_state()
        if active == 0: break
        
        if controller_type == "SC":
            power = sc.decide(physics.boiler_temp, physics.units)
        else:
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

def evaluate_all():
    print("="*80)
    print("V1.0 RCP EVALUATION: SmartController vs RCPolicy")
    print("="*80)
    
    model_path = os.path.join(current_dir, "model.pth")
    
    if not os.path.exists(model_path):
        print("CRITICAL: model.pth NOT FOUND. Please run train.py first.")
        return
    
    model = RCPolicy().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded: {model_path}\n")

    print(f"{'Scenario':<20} | {'SC (Rules)':<10} | {'RCP':<10} | {'Winner':<10}")
    print("-" * 60)
    
    scores = {"SC": 0, "RCP": 0}
    wins = {"SC": 0, "RCP": 0}
    
    for scenario in BENCHMARK_SCENARIOS:
        c_sc = run_simulation("SC", scenario=scenario)
        c_rcp = run_simulation("RCP", model=model, scenario=scenario)
        
        results = [("SC", c_sc), ("RCP", c_rcp)]
        results.sort(key=lambda x: x[1])
        winner = results[0][0]
        
        scores["SC"] += c_sc
        scores["RCP"] += c_rcp
        wins[winner] += 1
        
        print(f"{scenario['name']:<20} | {c_sc:>10.2f} | {c_rcp:>10.2f} | {winner:<10}")
        
    print("-" * 60)
    print(f"TOTAL COST           | {scores['SC']:>10.2f} | {scores['RCP']:>10.2f}")
    print(f"TOTAL WINS           | {wins['SC']:>10} | {wins['RCP']:>10}")
    print("="*80)

    if scores['RCP'] < scores['SC']:
        print("SUCCESS: RCPolicy is better than SmartController!")
    else:
        print("FAILURE: RCPolicy failed to improve.")

if __name__ == "__main__":
    evaluate_all()
