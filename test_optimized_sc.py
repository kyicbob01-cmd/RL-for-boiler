"""
Optimized SmartController Test
Compares Original SC vs Physics-Based Optimized SC
"""

import sys
import os

# Add V1.0 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# ==========================================
# Original SC (for reference)
# ==========================================
class OriginalSC:
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
# Optimized SC (Physics-Based)
# ==========================================
class OptimizedSC:
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        heating = [u for u in active if u['state'] == 'HEATING']
        holding = [u for u in active if u['state'] == 'HOLDING']
        
        # Phase 1: RUSH - Any unit in heating state
        if heating:
            max_target = max([u['target'] for u in heating])
            target_boiler = max_target + 15.0  # Aggressive margin
            
            gap = target_boiler - temp
            
            if gap > 5.0:
                return 100.0  # Full speed
            elif gap > 0:
                return 60.0   # Approaching target
            elif gap < -3.0:
                return 0.0    # Overshoot
            else:
                return 30.0   # Fine-tune
        
        # Phase 2: MAINTAIN - Only holding units
        if holding:
            max_holding_target = max([u['target'] for u in holding])
            target_boiler = max_holding_target + 3.0  # Minimal margin
            
            # Calculate required power based on losses
            # Boiler loss: (T - 25) * 4.0 kJ/s
            boiler_loss = (temp - 25.0) * 4.0
            
            # Unit transfer demand (approximate)
            unit_demand = 0.0
            for u in holding:
                if u['current'] < u['target'] - 0.5:
                    delta = temp - u['current']
                    unit_demand += delta * 0.5  # Reduced transfer coefficient estimate
            
            total_demand = boiler_loss + unit_demand
            required_kw = total_demand / (50.0 * 0.9)  # Convert to kW
            required_pct = (required_kw / 100.0) * 100.0 * 1.2  # 20% margin
            
            # Clamp and also consider if boiler is below target
            gap = target_boiler - temp
            if gap > 5.0:
                return min(100, max(required_pct, 50.0))
            elif gap > 0:
                return min(100, max(required_pct, 20.0))
            elif gap < -2.0:
                return 0.0
            else:
                return min(100, max(required_pct, 10.0))
        
        return 0.0

# ==========================================
# Test Runner
# ==========================================
def run_simulation(controller, scenario):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    for _ in range(2000):
        _, active, _, _ = physics.get_system_state()
        if active == 0: break
        
        power = controller.decide(physics.boiler_temp, physics.units)
        physics.step(power, dt=0.5)
    
    return physics.total_cost

def main():
    print("=" * 80)
    print("SmartController Benchmark: Original vs Optimized")
    print("=" * 80)
    
    original = OriginalSC()
    optimized = OptimizedSC()
    
    print(f"{'Scenario':<20} | {'Original SC':>12} | {'Optimized SC':>12} | {'Improvement':>12}")
    print("-" * 65)
    
    total_orig = 0
    total_opt = 0
    wins_orig = 0
    wins_opt = 0
    
    for scenario in BENCHMARK_SCENARIOS:
        cost_orig = run_simulation(original, scenario)
        cost_opt = run_simulation(optimized, scenario)
        
        total_orig += cost_orig
        total_opt += cost_opt
        
        if cost_orig < cost_opt:
            wins_orig += 1
            winner = "Original"
        else:
            wins_opt += 1
            winner = "Optimized"
        
        improvement = ((cost_orig - cost_opt) / cost_orig) * 100
        
        print(f"{scenario['name']:<20} | {cost_orig:>12.2f} | {cost_opt:>12.2f} | {improvement:>+11.1f}%")
    
    print("-" * 65)
    print(f"{'TOTAL':<20} | {total_orig:>12.2f} | {total_opt:>12.2f} | {((total_orig - total_opt) / total_orig) * 100:>+11.1f}%")
    print(f"{'WINS':<20} | {wins_orig:>12} | {wins_opt:>12} |")
    print("=" * 80)
    
    if total_opt < total_orig:
        print("SUCCESS: Optimized SC outperforms Original SC!")
    else:
        print("NEEDS WORK: Optimized SC did not beat Original SC.")

if __name__ == "__main__":
    main()
