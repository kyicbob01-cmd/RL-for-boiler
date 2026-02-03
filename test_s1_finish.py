"""
S1_Low Deep Analysis - Why Human Beats SC?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

S1 = BENCHMARK_SCENARIOS[0]

def run_with_detailed_log(strategy_name, decide_fn):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in S1["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    phases = []
    
    for step in range(2000):
        _, active, _, min_time = physics.get_system_state()
        if active == 0: break
        
        unit = list(physics.units.values())[0]
        power = decide_fn(physics)
        
        if step % 50 == 0 or step < 20:
            phases.append({
                'step': step,
                'boiler': physics.boiler_temp,
                'unit': unit['current'],
                'state': unit['state'],
                'power': power,
                'remaining': unit.get('duration_left', 0)
            })
        
        physics.step(power, dt=0.5)
    
    return physics.total_cost, physics.boiler_temp, phases

# Strategy: Optimized SC (current HMI version)
def optimized_sc(physics):
    temp = physics.boiler_temp
    units = physics.units
    
    active = [u for u in units.values() if u['state'] != 'FINISHED']
    if not active: return 0.0
    
    heating = [u for u in active if u['state'] == 'HEATING']
    holding = [u for u in active if u['state'] == 'HOLDING']
    
    all_targets = [u['target'] for u in active]
    max_target = max(all_targets)
    
    # Low temp: +8C heating, +1C holding
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
        target_boiler = max([u['target'] for u in holding]) + hold_margin
        gap = target_boiler - temp
        
        boiler_loss = (temp - 25.0) * 4.0
        required_kw = boiler_loss / (50.0 * 0.9)
        required_pct = (required_kw / 100.0) * 100.0 * 1.2
        
        if gap > 3: return min(100, max(40, required_pct))
        elif gap > 0: return min(100, max(15, required_pct))
        elif gap < -2: return 0.0
        else: return min(100, max(10, required_pct))
    
    return 0.0

# Strategy: Anticipatory (reduce power near end)
def anticipatory_sc(physics):
    temp = physics.boiler_temp
    units = physics.units
    
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
        min_remaining = min([u['duration_left'] for u in holding])
        
        # KEY CHANGE: Reduce target temperature as task nears completion
        if min_remaining < 10:  # Last 10 seconds
            target_boiler = hold_target - 1  # Let it coast below target
        elif min_remaining < 30:  # Last 30 seconds
            target_boiler = hold_target  # Exact target
        else:
            target_boiler = hold_target + hold_margin  # Normal margin
        
        gap = target_boiler - temp
        
        if gap > 3: return 40.0
        elif gap > 0: return 15.0
        elif gap < -3: return 0.0
        else: return 5.0  # Minimal maintenance
    
    return 0.0

# Strategy: Minimal Finish (stop heating 20s before end)
def minimal_finish_sc(physics):
    temp = physics.boiler_temp
    units = physics.units
    
    active = [u for u in units.values() if u['state'] != 'FINISHED']
    if not active: return 0.0
    
    heating = [u for u in active if u['state'] == 'HEATING']
    holding = [u for u in active if u['state'] == 'HOLDING']
    
    max_target = max([u['target'] for u in active])
    
    if heating:
        target_boiler = max([u['target'] for u in heating]) + 8
        return 100.0 if temp < target_boiler else 0.0
    
    if holding:
        hold_target = max([u['target'] for u in holding])
        min_remaining = min([u['duration_left'] for u in holding])
        
        # Stop all heating in last 20 seconds - coast to finish
        if min_remaining < 20:
            return 0.0
        
        # Normal holding logic
        target_boiler = hold_target + 1
        gap = target_boiler - temp
        
        if gap > 3: return 40.0
        elif gap > 0: return 15.0
        elif gap < -2: return 0.0
        else: return 10.0
    
    return 0.0

print("=" * 70)
print("S1_Low Deep Analysis")
print(f"Task: {S1['tasks'][0]}")
print("=" * 70)

strategies = [
    ("Current Optimized SC", optimized_sc),
    ("Anticipatory SC (reduce near end)", anticipatory_sc),
    ("Minimal Finish (stop 20s before)", minimal_finish_sc),
]

print(f"\n{'Strategy':<40} | {'Cost':>8} | {'Final Boiler':>12}")
print("-" * 65)

for name, fn in strategies:
    cost, final_temp, phases = run_with_detailed_log(name, fn)
    print(f"{name:<40} | {cost:>8.3f} | {final_temp:>11.1f}C")

print("-" * 65)
print(f"Human achieved: 4.33 with final boiler ~80.1C")
print("=" * 70)

# Show detailed phase for Minimal Finish
print("\n[Minimal Finish SC - Detailed Log]")
_, final, phases = run_with_detailed_log("Minimal Finish", minimal_finish_sc)
print(f"{'Step':>6} | {'Boiler':>8} | {'Unit':>8} | {'State':>10} | {'Power':>6} | {'Remaining':>10}")
print("-" * 65)
for p in phases:
    print(f"{p['step']:>6} | {p['boiler']:>7.1f}C | {p['unit']:>7.1f}C | {p['state']:>10} | {p['power']:>5.0f}% | {p['remaining']:>9.1f}s")
