import torch
import numpy as np
import random
import os
from boiler_env import BoilerPhysics
from train import RCPolicy
from evaluate import SmartControllerRules

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_scenario(id):
    """Generates a completely random industrial task."""
    num_tasks = random.choice([1, 2, 3, 4])
    tasks = []
    
    names = ["Unit_A", "Unit_B", "Unit_C", "Unit_D"]
    
    for i in range(num_tasks):
        # Randomize physics parameters
        # Target Temp: 60 - 200 (Wide range)
        target = random.uniform(60.0, 200.0)
        # Duration: 50 - 500 (Short to Long)
        duration = random.uniform(50.0, 500.0)
        # Weight (Heat Loss factor): 200 - 3000 (Easy to Extreme)
        weight = random.uniform(200.0, 3000.0)
        
        tasks.append({
            "name": names[i],
            "target": target,
            "duration": duration,
            "weight": weight
        })
        
    complexity = "Low"
    if num_tasks > 2 or any(t['weight'] > 2000 for t in tasks):
        complexity = "High"
    elif num_tasks == 2:
        complexity = "Medium"
        
    return {
        "name": f"Rand_{id:03d}_{complexity}",
        "tasks": tasks
    }

def run_simulation(controller_type, scenario, model=None, sc_rules=None):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
    # Run Simulation
    for _ in range(3000): # Max 3000 steps
        max_t, active, load = physics.get_system_state()
        if active == 0: break
        
        power = 0.0
        if controller_type == "SC":
            power = sc_rules.decide(physics.boiler_temp, physics.units)
            
        elif controller_type == "RCP":
            # Ambitious Strategy: Target = 5.0
            obs = torch.tensor([[
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                0.0,
                load / 2000000.0
            ]], dtype=torch.float32).to(device)
            
            target = torch.tensor([[5.0]], dtype=torch.float32).to(device)
            with torch.no_grad():
                power = model(obs, target).item() * 100.0
                
        physics.step(power, dt=0.5)
        
    return physics.total_cost

def stress_test(num_episodes=100):
    print(f"Starting Stress Test: {num_episodes} Random Scenarios...")
    print("-" * 60)
    
    if not os.path.exists("model.pth"):
         print("Error: model.pth not found. Train first.")
         return

    # Load Models
    model = RCPolicy().to(device)
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    sc = SmartControllerRules()
    
    wins = 0
    losses = 0
    ties = 0
    
    total_cost_sc = 0.0
    total_cost_rcp = 0.0
    
    results = []
    
    print(f"{'ID':<10} | {'Type':<10} | {'SC Cost':<10} | {'RCP Cost':<10} | {'Winner':<10}")
    print("-" * 60)
    
    for i in range(num_episodes):
        scenario = generate_random_scenario(i+1)
        
        cost_sc = run_simulation("SC", scenario, sc_rules=sc)
        cost_rcp = run_simulation("RCP", scenario, model=model)
        
        total_cost_sc += cost_sc
        total_cost_rcp += cost_rcp
        
        diff = cost_rcp - cost_sc
        winner = "Tie"
        if diff < -0.1: 
            winner = "RCP"
            wins += 1
        elif diff > 0.1: 
            winner = "SC"
            losses += 1
        else:
            ties += 1
            
        print(f"{scenario['name']:<10} | {len(scenario['tasks'])} Units   | {cost_sc:<10.2f} | {cost_rcp:<10.2f} | {winner:<10}")
        
    print("-" * 60)
    print("Final Statistics:")
    print(f"Total Scenarios: {num_episodes}")
    print(f"RCP Wins: {wins} ({wins/num_episodes*100:.1f}%)")
    print(f"SC Wins : {losses}")
    print(f"Ties    : {ties}")
    print(f"Total Cost SC : {total_cost_sc:.2f}")
    print(f"Total Cost RCP: {total_cost_rcp:.2f}")
    
    eff = ((total_cost_sc - total_cost_rcp) / total_cost_sc) * 100.0
    print(f"Overall Efficiency Improvement: {eff:+.2f}%")

if __name__ == "__main__":
    stress_test(50) # Run 50 random tests
