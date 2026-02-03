"""
Adaptive SmartController with Per-Scenario Optimization
Tests many parameter combinations to find optimal settings per scenario type
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# ==========================================
# Adaptive SC - adjusts based on task difficulty
# ==========================================
class AdaptiveSC:
    def __init__(self, 
                 lo_heat_margin=3, lo_hold_margin=1,    # Low temp (< 100C)
                 mid_heat_margin=15, mid_hold_margin=2,  # Mid temp (100-150C)
                 hi_heat_margin=30, hi_hold_margin=2):   # High temp (> 150C)
        self.lo_heat_margin = lo_heat_margin
        self.lo_hold_margin = lo_hold_margin
        self.mid_heat_margin = mid_heat_margin
        self.mid_hold_margin = mid_hold_margin
        self.hi_heat_margin = hi_heat_margin
        self.hi_hold_margin = hi_hold_margin
    
    def get_margins(self, max_target):
        """Select strategy based on maximum target temperature"""
        if max_target < 100:
            return self.lo_heat_margin, self.lo_hold_margin
        elif max_target < 150:
            return self.mid_heat_margin, self.mid_hold_margin
        else:
            return self.hi_heat_margin, self.hi_hold_margin
    
    def decide(self, temp, units):
        active = [u for u in units.values() if u['state'] != 'FINISHED']
        if not active: return 0.0
        
        heating = [u for u in active if u['state'] == 'HEATING']
        holding = [u for u in active if u['state'] == 'HOLDING']
        
        # Get maximum target across all active units
        all_targets = [u['target'] for u in active]
        max_target = max(all_targets)
        heat_margin, hold_margin = self.get_margins(max_target)
        
        # Phase 1: HEATING (Bang-Bang with adaptive margin)
        if heating:
            heat_target = max([u['target'] for u in heating])
            target_boiler = heat_target + heat_margin
            
            if temp < target_boiler:
                return 100.0  # Full power
            else:
                return 0.0    # Coast
        
        # Phase 2: HOLDING (Constant with adaptive margin)
        if holding:
            hold_target = max([u['target'] for u in holding])
            target_boiler = hold_target + hold_margin
            gap = target_boiler - temp
            
            # Calculate required power
            boiler_loss = (temp - 25.0) * 4.0
            required_kw = boiler_loss / (50.0 * 0.9)
            required_pct = (required_kw / 100.0) * 100.0 * 1.2
            
            if gap > 3:
                return min(100, max(40, required_pct))
            elif gap > 0:
                return min(100, max(15, required_pct))
            elif gap < -2:
                return 0.0
            else:
                return min(100, max(10, required_pct))
        
        return 0.0

# ==========================================
# Test Runner
# ==========================================
def run_scenario(controller, scenario):
    physics = BoilerPhysics()
    physics.reset()
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    for _ in range(2000):
        _, active, _ = physics.get_system_state()
        if active == 0: break
        power = controller.decide(physics.boiler_temp, physics.units)
        physics.step(power, dt=0.5)
    
    return physics.total_cost

def run_all_scenarios(controller):
    """Returns dict of scenario_name -> cost"""
    results = {}
    for scenario in BENCHMARK_SCENARIOS:
        cost = run_scenario(controller, scenario)
        results[scenario['name']] = cost
    return results

# ==========================================
# Comprehensive Grid Search
# ==========================================
def main():
    print("=" * 90)
    print("Adaptive SC Comprehensive Grid Search")
    print("=" * 90)
    
    # Parameter options
    lo_heat_options = [2, 3, 5, 8]
    lo_hold_options = [1, 2, 3]
    mid_heat_options = [10, 15, 20, 25]
    mid_hold_options = [2, 3, 5]
    hi_heat_options = [20, 30, 40, 50]
    hi_hold_options = [2, 3, 5]
    
    total_configs = (len(lo_heat_options) * len(lo_hold_options) * 
                     len(mid_heat_options) * len(mid_hold_options) *
                     len(hi_heat_options) * len(hi_hold_options))
    
    print(f"\nTesting {total_configs} configurations...")
    print("(This may take a few minutes)\n")
    
    results = []
    count = 0
    
    for lo_h in lo_heat_options:
        for lo_d in lo_hold_options:
            for mid_h in mid_heat_options:
                for mid_d in mid_hold_options:
                    for hi_h in hi_heat_options:
                        for hi_d in hi_hold_options:
                            controller = AdaptiveSC(
                                lo_heat_margin=lo_h, lo_hold_margin=lo_d,
                                mid_heat_margin=mid_h, mid_hold_margin=mid_d,
                                hi_heat_margin=hi_h, hi_hold_margin=hi_d
                            )
                            scenario_costs = run_all_scenarios(controller)
                            total = sum(scenario_costs.values())
                            
                            results.append({
                                'lo_h': lo_h, 'lo_d': lo_d,
                                'mid_h': mid_h, 'mid_d': mid_d,
                                'hi_h': hi_h, 'hi_d': hi_d,
                                'total': total,
                                'details': scenario_costs
                            })
                            
                            count += 1
                            if count % 100 == 0:
                                print(f"  Progress: {count}/{total_configs}")
    
    # Sort by total cost
    results.sort(key=lambda x: x['total'])
    
    # Print top 10
    print("\n" + "=" * 90)
    print("TOP 10 CONFIGURATIONS:")
    print("-" * 90)
    print(f"{'Rank':>4} | {'Lo Heat':>8} | {'Lo Hold':>8} | {'Mid Heat':>9} | {'Mid Hold':>9} | {'Hi Heat':>8} | {'Hi Hold':>8} | {'Total':>10}")
    print("-" * 90)
    
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>4} | +{r['lo_h']:>6}C | +{r['lo_d']:>6}C | +{r['mid_h']:>7}C | +{r['mid_d']:>7}C | +{r['hi_h']:>6}C | +{r['hi_d']:>6}C | {r['total']:>10.2f}")
    
    # Show best config details
    best = results[0]
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION DETAILS:")
    print("-" * 90)
    print(f"Low Temp (<100C):  Heat +{best['lo_h']}C, Hold +{best['lo_d']}C")
    print(f"Mid Temp (100-150C): Heat +{best['mid_h']}C, Hold +{best['mid_d']}C")
    print(f"High Temp (>150C): Heat +{best['hi_h']}C, Hold +{best['hi_d']}C")
    print(f"\nPer-Scenario Results:")
    for name, cost in best['details'].items():
        print(f"  {name:<20}: {cost:.2f}")
    print(f"\nTOTAL: {best['total']:.2f}")
    
    # Compare to baselines
    print("\n" + "-" * 90)
    print("COMPARISON TO BASELINES:")
    print(f"  Original SC:     174.61")
    print(f"  Previous Best:   163.53 (Uniform +50C)")
    print(f"  Adaptive Best:   {best['total']:.2f}")
    print(f"  Improvement:     {((174.61 - best['total']) / 174.61 * 100):.1f}% vs Original")
    print("=" * 90)

if __name__ == "__main__":
    main()
