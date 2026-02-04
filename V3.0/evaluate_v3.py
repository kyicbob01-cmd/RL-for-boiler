import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC
from policy import RCPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sc(scenario):
    sc = TimeAwareSC()
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    for _ in range(2000):
        _, active, _, _ = physics.get_system_state()
        if active == 0: break
        
        power = sc.decide(physics.boiler_temp, physics.units)
        physics.step(power, dt=0.5)
    
    return physics.total_cost

def run_rcp(scenario, model, target_multiplier=0.9):
    sc_cost = run_sc(scenario)
    target_cost = sc_cost * target_multiplier
    
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
            
            target = torch.tensor([[target_cost]], dtype=torch.float32).to(device)
            
            action = model(state, target).item()
            power = action * 100.0
            physics.step(power, dt=0.5)
    
    return physics.total_cost, sc_cost, target_cost

def evaluate_with_targets():
    print("=" * 90)
    print("V3.0 RCP EVALUATION: Testing Multiple Target Cost Multipliers")
    print("=" * 90)
    
    model = RCPolicy().to(device)
    rcp_path = os.path.join(current_dir, "model_rcp.pth")
    
    if not os.path.exists(rcp_path):
        print("ERROR: model_rcp.pth not found. Run train_rcp.py first.")
        return
    
    model.load_state_dict(torch.load(rcp_path, map_location=device))
    model.eval()
    print(f"Loaded: {rcp_path}\n")
    
    multipliers = [1.0, 0.95, 0.9, 0.85, 0.8]
    
    for mult in multipliers:
        print(f"\n{'='*80}")
        print(f"TARGET MULTIPLIER: {mult} (asking for cost = SC * {mult})")
        print(f"{'='*80}")
        
        print(f"{'Scenario':<20} | {'SC Cost':>10} | {'RCP Cost':>10} | {'Target':>10} | {'Diff':>8}")
        print("-" * 70)
        
        total_sc = 0
        total_rcp = 0
        victories = 0
        
        for scenario in BENCHMARK_SCENARIOS:
            rcp_cost, sc_cost, target = run_rcp(scenario, model, mult)
            
            total_sc += sc_cost
            total_rcp += rcp_cost
            
            diff = ((sc_cost - rcp_cost) / sc_cost) * 100
            if rcp_cost < sc_cost:
                victories += 1
            
            print(f"{scenario['name']:<20} | {sc_cost:>10.2f} | {rcp_cost:>10.2f} | {target:>10.2f} | {diff:>+7.1f}%")
        
        print("-" * 70)
        improvement = ((total_sc - total_rcp) / total_sc) * 100
        print(f"{'TOTAL':<20} | {total_sc:>10.2f} | {total_rcp:>10.2f} |            | {improvement:>+7.1f}%")
        print(f"{'VICTORIES':<20} |            | {victories:>10}/10 |            |")

def evaluate_standard():
    print("=" * 80)
    print("V3.0 RCP FINAL EVALUATION: Time-Aware SC vs RCP (target = 0.9x)")
    print("=" * 80)
    
    model = RCPolicy().to(device)
    rcp_path = os.path.join(current_dir, "model_rcp.pth")
    
    if not os.path.exists(rcp_path):
        print("ERROR: model_rcp.pth not found. Run train_rcp.py first.")
        return
    
    model.load_state_dict(torch.load(rcp_path, map_location=device))
    model.eval()
    print(f"Loaded: {rcp_path}\n")
    
    print(f"{'Scenario':<20} | {'SC Cost':>10} | {'RCP Cost':>10} | {'Improvement':>12} | {'Winner':>8}")
    print("-" * 75)
    
    total_sc = 0
    total_rcp = 0
    victories = 0
    
    for scenario in BENCHMARK_SCENARIOS:
        sc_cost = run_sc(scenario)
        rcp_cost, _, _ = run_rcp(scenario, model, 0.9)
        
        total_sc += sc_cost
        total_rcp += rcp_cost
        
        improvement = ((sc_cost - rcp_cost) / sc_cost) * 100
        winner = "RCP" if rcp_cost < sc_cost else "SC"
        
        if rcp_cost < sc_cost:
            victories += 1
        
        print(f"{scenario['name']:<20} | {sc_cost:>10.2f} | {rcp_cost:>10.2f} | {improvement:>+11.1f}% | {winner:>8}")
    
    print("-" * 75)
    total_improvement = ((total_sc - total_rcp) / total_sc) * 100
    print(f"{'TOTAL':<20} | {total_sc:>10.2f} | {total_rcp:>10.2f} | {total_improvement:>+11.1f}% |")
    print(f"{'VICTORIES':<20} |            | {victories:>10}/10 |            |")
    print("=" * 80)
    
    if victories == 10:
        print("SUCCESS: V3.0 RCP beats Time-Aware SC on ALL scenarios!")
    elif victories >= 6:
        print(f"GOOD: V3.0 RCP beats Time-Aware SC on {victories}/10 scenarios.")
    else:
        print(f"NEEDS IMPROVEMENT: V3.0 RCP only beats SC on {victories}/10 scenarios.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        evaluate_with_targets()
    else:
        evaluate_standard()
