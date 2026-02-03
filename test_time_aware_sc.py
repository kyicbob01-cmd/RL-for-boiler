"""
Time-Aware SC Benchmark
Compares Current SC vs Time-Aware SC (with remaining time sensing)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# ==========================================
# Current Adaptive SC (no time awareness)
# ==========================================
class CurrentAdaptiveSC:
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        heating = [u for u in active if u['state'] == 'HEATING']
        holding = [u for u in active if u['state'] == 'HOLDING']
        
        max_target = max([u['target'] for u in active])
        
        if max_target < 100:
            heat_margin, hold_margin = 8, 1
        elif max_target < 150:
            heat_margin, hold_margin = 20, 2
        else:
            heat_margin, hold_margin = 40, 2
        
        if heating:
            target_boiler = max([u['target'] for u in heating]) + heat_margin
            return 100.0 if temp < target_boiler else 0.0
        
        if holding:
            hold_target = max([u['target'] for u in holding])
            target_boiler = hold_target + hold_margin
            gap = target_boiler - temp
            
            boiler_loss = (temp - 25.0) * 4.0
            required_kw = boiler_loss / (50.0 * 0.9)
            required_pct = (required_kw / 100.0) * 100.0 * 1.2
            
            if gap > 3: return min(100, max(40, required_pct))
            elif gap > 0: return min(100, max(15, required_pct))
            elif gap < -2: return 0.0
            else: return min(100, max(10, required_pct))
        
        return 0.0

# ==========================================
# Time-Aware SC (with finishing optimization)
# ==========================================
class TimeAwareSC:
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        heating = [u for u in active if u['state'] == 'HEATING']
        holding = [u for u in active if u['state'] == 'HOLDING']
        
        max_target = max([u['target'] for u in active])
        
        if max_target < 100:
            heat_margin, hold_margin = 8, 1
        elif max_target < 150:
            heat_margin, hold_margin = 20, 2
        else:
            heat_margin, hold_margin = 40, 2
        
        # Phase 1: HEATING (unchanged)
        if heating:
            target_boiler = max([u['target'] for u in heating]) + heat_margin
            return 100.0 if temp < target_boiler else 0.0
        
        # Phase 2: HOLDING (with time awareness)
        if holding:
            hold_target = max([u['target'] for u in holding])
            min_remaining = min([u['duration_left'] for u in holding])
            
            # TIME-AWARE FINISHING LOGIC
            if min_remaining <= 15:
                # Last 15 seconds: stop heating, coast to target
                return 0.0
            elif min_remaining <= 30:
                # Last 30 seconds: minimal power, just prevent undershoot
                target_boiler = hold_target  # Exact target, no margin
                gap = target_boiler - temp
                if gap > 1: return 15.0
                else: return 0.0
            else:
                # Normal holding
                target_boiler = hold_target + hold_margin
                gap = target_boiler - temp
                
                boiler_loss = (temp - 25.0) * 4.0
                required_kw = boiler_loss / (50.0 * 0.9)
                required_pct = (required_kw / 100.0) * 100.0 * 1.2
                
                if gap > 3: return min(100, max(40, required_pct))
                elif gap > 0: return min(100, max(15, required_pct))
                elif gap < -2: return 0.0
                else: return min(100, max(10, required_pct))
        
        return 0.0

# ==========================================
# Benchmark Runner
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
    
    return physics.total_cost, physics.boiler_temp

def main():
    print("=" * 85)
    print("Time-Aware SC Benchmark: Current vs Time-Aware")
    print("=" * 85)
    
    current = CurrentAdaptiveSC()
    time_aware = TimeAwareSC()
    
    print(f"\n{'Scenario':<20} | {'Current SC':>12} | {'Time-Aware SC':>14} | {'Improvement':>12} | {'Final Temp':>10}")
    print("-" * 85)
    
    total_current = 0
    total_aware = 0
    wins_current = 0
    wins_aware = 0
    
    for scenario in BENCHMARK_SCENARIOS:
        cost_current, _ = run_simulation(current, scenario)
        cost_aware, final_temp = run_simulation(time_aware, scenario)
        
        total_current += cost_current
        total_aware += cost_aware
        
        improvement = ((cost_current - cost_aware) / cost_current) * 100
        
        if cost_current < cost_aware:
            wins_current += 1
            winner = "Current"
        else:
            wins_aware += 1
            winner = "Time-Aware"
        
        print(f"{scenario['name']:<20} | {cost_current:>12.2f} | {cost_aware:>14.2f} | {improvement:>+11.1f}% | {final_temp:>9.1f}C")
    
    print("-" * 85)
    improvement_total = ((total_current - total_aware) / total_current) * 100
    print(f"{'TOTAL':<20} | {total_current:>12.2f} | {total_aware:>14.2f} | {improvement_total:>+11.1f}% |")
    print(f"{'WINS':<20} | {wins_current:>12} | {wins_aware:>14} |")
    print("=" * 85)
    
    if total_aware < total_current:
        print("SUCCESS: Time-Aware SC outperforms Current SC!")
    else:
        print("NEEDS WORK: Time-Aware SC did not improve overall.")

if __name__ == "__main__":
    main()
