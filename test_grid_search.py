"""
Comprehensive SC Strategy Grid Search
Explores all combinations of heating and holding phase parameters
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# ==========================================
# Configurable SC with separate heating/holding params
# ==========================================
class ConfigurableSC:
    def __init__(self, heat_high=15, heat_low=0, hold_margin=3):
        # Heating phase: cycle between (target + heat_low) and (target + heat_high)
        self.heat_high = heat_high
        self.heat_low = heat_low
        # Holding phase: maintain at (target + hold_margin)
        self.hold_margin = hold_margin
    
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        heating = [u for u in active if u['state'] == 'HEATING']
        holding = [u for u in active if u['state'] == 'HOLDING']
        
        # Phase 1: HEATING - Use cycling strategy
        if heating:
            max_target = max([u['target'] for u in heating])
            low_bound = max_target + self.heat_low
            high_bound = max_target + self.heat_high
            
            if temp < low_bound:
                return 100.0  # Rush up
            elif temp > high_bound:
                return 0.0    # Coast down
            else:
                # In the middle - use hysteresis
                mid = (low_bound + high_bound) / 2
                if temp < mid:
                    return 100.0
                else:
                    return 0.0
        
        # Phase 2: HOLDING - Use constant strategy
        if holding:
            max_target = max([u['target'] for u in holding])
            target_boiler = max_target + self.hold_margin
            gap = target_boiler - temp
            
            # Calculate required power based on losses
            boiler_loss = (temp - 25.0) * 4.0
            required_kw = boiler_loss / (50.0 * 0.9)
            required_pct = (required_kw / 100.0) * 100.0 * 1.2
            
            if gap > 3.0:
                return min(100, max(required_pct, 40.0))
            elif gap > 0:
                return min(100, max(required_pct, 15.0))
            elif gap < -2.0:
                return 0.0
            else:
                return min(100, max(required_pct, 10.0))
        
        return 0.0

# ==========================================
# Grid Search
# ==========================================
def run_benchmark(controller):
    total_cost = 0
    for scenario in BENCHMARK_SCENARIOS:
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        for _ in range(2000):
            _, active, _ = physics.get_system_state()
            if active == 0: break
            power = controller.decide(physics.boiler_temp, physics.units)
            physics.step(power, dt=0.5)
        
        total_cost += physics.total_cost
    return total_cost

def main():
    print("=" * 80)
    print("Comprehensive SC Strategy Grid Search")
    print("=" * 80)
    
    # Parameter ranges
    heat_high_options = [10, 15, 20, 25, 30]  # Heating phase: high margin
    heat_low_options = [0, 5]                  # Heating phase: low margin  
    hold_margin_options = [2, 3, 5, 8]         # Holding phase: constant margin
    
    results = []
    
    print(f"\nTesting {len(heat_high_options) * len(heat_low_options) * len(hold_margin_options)} configurations...\n")
    
    for heat_high in heat_high_options:
        for heat_low in heat_low_options:
            for hold_margin in hold_margin_options:
                controller = ConfigurableSC(heat_high, heat_low, hold_margin)
                cost = run_benchmark(controller)
                results.append({
                    'heat_high': heat_high,
                    'heat_low': heat_low,
                    'hold_margin': hold_margin,
                    'cost': cost
                })
    
    # Sort by cost
    results.sort(key=lambda x: x['cost'])
    
    # Print top 10
    print("TOP 10 CONFIGURATIONS:")
    print("-" * 70)
    print(f"{'Rank':>4} | {'Heat High':>10} | {'Heat Low':>9} | {'Hold Margin':>11} | {'Total Cost':>12}")
    print("-" * 70)
    
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>4} | +{r['heat_high']:>8}C | +{r['heat_low']:>7}C | +{r['hold_margin']:>9}C | {r['cost']:>12.2f}")
    
    print("-" * 70)
    print(f"\nBest Configuration:")
    best = results[0]
    print(f"  Heating Phase: Cycle from +{best['heat_low']}C to +{best['heat_high']}C")
    print(f"  Holding Phase: Constant at +{best['hold_margin']}C")
    print(f"  Total Cost: {best['cost']:.2f}")
    
    # Compare to baseline
    print("\n" + "-" * 70)
    print("Comparison to Original SC (174.61):")
    print(f"  Best Optimized: {best['cost']:.2f}")
    print(f"  Improvement: {((174.61 - best['cost']) / 174.61 * 100):.1f}%")
    
    # Also test extreme values
    print("\n" + "=" * 80)
    print("EXTREME VALUES TEST:")
    print("-" * 70)
    
    extreme_configs = [
        (40, 0, 2, "Very aggressive heating (+40C)"),
        (50, 0, 2, "Ultra aggressive heating (+50C)"),
        (20, 10, 3, "Narrow heating band (+10 to +20C)"),
        (25, 0, 1, "Tight holding (+1C)"),
    ]
    
    for heat_high, heat_low, hold_margin, desc in extreme_configs:
        controller = ConfigurableSC(heat_high, heat_low, hold_margin)
        cost = run_benchmark(controller)
        print(f"{desc:<40} | {cost:>12.2f}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
