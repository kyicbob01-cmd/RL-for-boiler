"""
SC vs RCP - Performance Comparison Script
=========================================
Benchmarks the Return-Conditioned Policy (RCP) against the Rule-Based 
SmartController (SC) across 10 standard scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# ==========================================
# Model Definition
# ==========================================
class RCPolicy(nn.Module):
    """Return-Conditioned Policy Network (Inference Mode)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, state, target_cost):
        cost_norm = target_cost / 50.0
        x = torch.cat([state, cost_norm], dim=-1)
        return self.net(x)

# ==========================================
# Rule-Based Controller
# ==========================================
class SmartControllerRules:
    """Legacy rule-based logic for comparison baseline."""
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

# ==========================================
# Simulation Runner
# ==========================================
def run_simulation(controller_type, model=None, scenario=None):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    sc_rules = SmartControllerRules()
    history_temp = []
    current_time = 0.0
    
    for _ in range(2000): # Max 1000s
        max_t, active, load = physics.get_system_state()
        if active == 0: break
        
        rate = 0.0
        if len(history_temp) > 0:
            rate = physics.boiler_temp - history_temp[-1]
        history_temp.append(physics.boiler_temp)
        
        power = 0.0
        if controller_type == "SC":
            power = sc_rules.decide(physics.boiler_temp, physics.units)
            
        elif controller_type == "RCP":
            if model:
                obs = torch.tensor([[
                    physics.boiler_temp / 300.0,
                    max_t / 300.0,
                    active / 4.0,
                    0.0,
                    load / 2000000.0
                ]], dtype=torch.float32)
                
                # Inference Target: 10.0 TWD
                target = torch.tensor([[10.0]], dtype=torch.float32)
                
                with torch.no_grad():
                    power = model(obs, target).item() * 100.0
        
        physics.step(power, dt=0.5)
        current_time += 0.5
        
    return {
        "time": current_time,
        "cost": physics.total_cost
    }

# ==========================================
# Main Execution
# ==========================================
def compare_all():
    print("="*80)
    print("  SC vs RCP - Performance Benchmark")
    print("="*80)
    
    if not os.path.exists("rc_policy.pth"):
        print("Error: rc_policy.pth not found! Run train_rcp.py first.")
        return

    rcp_model = RCPolicy()
    try:
        rcp_model.load_state_dict(torch.load("rc_policy.pth", map_location='cpu'))
        rcp_model.eval()
        print(">>> RCP Model Loaded Successfully")
    except Exception as e:
        print(f"Error loading RCP model: {e}")
        return

    results = []
    print(f"{'Scenario':<25} | {'SC Cost':<10} | {'RCP Cost':<10} | {'Diff':<10} | {'Winner':<10}")
    print("-" * 80)
    
    sc_wins = 0
    rcp_wins = 0
    total_sc_cost = 0
    total_rcp_cost = 0

    for scenario in BENCHMARK_SCENARIOS:
        res_sc = run_simulation("SC", scenario=scenario)
        res_rcp = run_simulation("RCP", model=rcp_model, scenario=scenario)
        
        diff = res_sc['cost'] - res_rcp['cost']
        if diff > 0.5:
            winner = "RCP 🏆"
            rcp_wins += 1
        elif diff < -0.5:
            winner = "SC"
            sc_wins += 1
        else:
            winner = "Tie"
            
        print(f"{scenario['name']:<25} | {res_sc['cost']:>8.2f}   | {res_rcp['cost']:>8.2f}   | {diff:>+8.2f}   | {winner}")
        
        total_sc_cost += res_sc['cost']
        total_rcp_cost += res_rcp['cost']

    print("-" * 80)
    print(f"Total Wins: SC={sc_wins}, RCP={rcp_wins}")
    print(f"Total Cost: SC={total_sc_cost:.2f}, RCP={total_rcp_cost:.2f} (Diff: {total_sc_cost - total_rcp_cost:+.2f})")
    print("="*80)

if __name__ == "__main__":
    compare_all()
