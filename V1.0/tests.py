import random
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from boiler_env import BoilerPhysics
from train import RCPolicy, SmartController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_scenario():
    scenario = {"name": f"Random_{random.randint(1000, 9999)}", "tasks": []}
    for i in range(random.randint(1, 4)):
        scenario["tasks"].append({
            "name": f"Task_{i}",
            "target": random.uniform(60, 180),
            "duration": random.uniform(50, 500),
            "weight": random.uniform(100, 2000)
        })
    return scenario

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
    
    completed = all(u['state'] == 'FINISHED' for u in physics.units.values())
    return physics.total_cost, completed

def stress_test(num_tests=100):
    print("="*80)
    print(f"V1.0 RCP STRESS TEST: {num_tests} Random Scenarios")
    print("="*80)
    
    model_path = os.path.join(current_dir, "model.pth")
    
    if not os.path.exists(model_path):
        print("CRITICAL: model.pth NOT FOUND. Please run train.py first.")
        return
    
    model = RCPolicy().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded: {model_path}\n")
    
    rcp_wins = 0
    sc_wins = 0
    rcp_total = 0
    sc_total = 0
    rcp_failures = 0
    sc_failures = 0
    
    for i in range(num_tests):
        scenario = generate_random_scenario()
        
        c_sc, sc_ok = run_simulation("SC", scenario=scenario)
        c_rcp, rcp_ok = run_simulation("RCP", model=model, scenario=scenario)
        
        if not sc_ok: sc_failures += 1
        if not rcp_ok: rcp_failures += 1
        
        sc_total += c_sc
        rcp_total += c_rcp
        
        if c_rcp < c_sc:
            rcp_wins += 1
        else:
            sc_wins += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_tests}, RCP Wins: {rcp_wins}, SC Wins: {sc_wins}")
    
    print("\n" + "="*80)
    print("RESULTS:")
    print(f"  RCP Wins: {rcp_wins}/{num_tests} ({100*rcp_wins/num_tests:.1f}%)")
    print(f"  SC Wins:  {sc_wins}/{num_tests} ({100*sc_wins/num_tests:.1f}%)")
    print(f"  RCP Total Cost: {rcp_total:.2f}")
    print(f"  SC Total Cost:  {sc_total:.2f}")
    print(f"  RCP Failures:   {rcp_failures}")
    print(f"  SC Failures:    {sc_failures}")
    
    improvement = ((sc_total - rcp_total) / sc_total) * 100
    print(f"\n  Overall Improvement: {improvement:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    stress_test(100)
