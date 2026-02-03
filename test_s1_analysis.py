"""
S1_Low Deep Analysis
Why can human achieve 4.45 but SC can only get ~4.5+?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS

# Get S1 scenario
S1 = BENCHMARK_SCENARIOS[0]
print(f"S1 Scenario: {S1}")
print(f"  Tasks: {S1['tasks']}")

# ==========================================
# Detailed Simulation with Logging
# ==========================================
def run_with_log(power_strategy, max_steps=2000):
    physics = BoilerPhysics()
    physics.reset()
    
    for task in S1["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
    
    log = []
    
    for step in range(max_steps):
        _, active, _ = physics.get_system_state()
        if active == 0: break
        
        power = power_strategy(physics)
        
        unit = list(physics.units.values())[0]
        log.append({
            'step': step,
            'time': step * 0.5,
            'boiler': physics.boiler_temp,
            'unit': unit['current'],
            'state': unit['state'],
            'power': power,
            'cost': physics.total_cost
        })
        
        physics.step(power, dt=0.5)
    
    return physics.total_cost, log

# ==========================================
# Different Strategies
# ==========================================
def original_sc_strategy(physics):
    temp = physics.boiler_temp
    units = physics.units
    
    active = [u for u in units.values() if u['state'] != 'FINISHED']
    if not active: return 0.0
    
    needed = [u for u in active if u['current'] < u['target'] - 0.5]
    targets = [u['target'] for u in needed]
    
    if not targets:
        holding = [u['target'] for u in active if u['state'] == 'HOLDING']
        if holding and temp < min(holding) + 3.0: return 30.0
        return 0.0
    
    max_t = max(targets)
    target_boiler = max_t + 5.0
    gap = target_boiler - temp
    
    if gap < -2.0: return 0.0
    if gap > 20: return 100.0
    elif gap > 10: return 80.0
    elif gap > 5: return 50.0
    elif gap > 0: return 20.0
    else: return 0.0

def aggressive_strategy(physics):
    """New aggressive SC"""
    temp = physics.boiler_temp
    units = physics.units
    
    active = [u for u in units.values() if u['state'] != 'FINISHED']
    if not active: return 0.0
    
    heating = [u for u in active if u['state'] == 'HEATING']
    holding = [u for u in active if u['state'] == 'HOLDING']
    
    if heating:
        max_target = max([u['target'] for u in heating])
        target_high = max_target + 30
        target_low = max_target + 5
        
        if temp < target_low:
            return 100.0
        elif temp > target_high:
            return 0.0
        else:
            return 100.0 if temp < (target_low + target_high) / 2 else 0.0
    
    if holding:
        max_target = max([u['target'] for u in holding])
        target_boiler = max_target + 2
        gap = target_boiler - temp
        
        if gap > 3: return 30.0
        elif gap > 0: return 15.0
        elif gap < -2: return 0.0
        else: return 10.0
    
    return 0.0

def minimal_strategy(physics):
    """Minimal power - just enough to complete task"""
    temp = physics.boiler_temp
    units = physics.units
    
    active = [u for u in units.values() if u['state'] != 'FINISHED']
    if not active: return 0.0
    
    max_target = max([u['target'] for u in active])
    
    # Very tight control - just above target
    target_boiler = max_target + 1.0
    gap = target_boiler - temp
    
    if gap > 10: return 80.0
    elif gap > 5: return 50.0
    elif gap > 2: return 30.0
    elif gap > 0: return 15.0
    elif gap < -1: return 0.0
    else: return 5.0

def bang_bang_strategy(physics):
    """Simple on/off control"""
    temp = physics.boiler_temp
    units = physics.units
    
    active = [u for u in units.values() if u['state'] != 'FINISHED']
    if not active: return 0.0
    
    heating = [u for u in active if u['state'] == 'HEATING']
    holding = [u for u in active if u['state'] == 'HOLDING']
    
    if heating:
        max_target = max([u['target'] for u in heating])
        target = max_target + 3  # Very close
        if temp < target:
            return 100.0
        else:
            return 0.0
    
    if holding:
        max_target = max([u['target'] for u in holding])
        target = max_target + 1  # Even tighter for holding
        if temp < target:
            return 30.0
        else:
            return 0.0
    
    return 0.0

# ==========================================
# Run Tests
# ==========================================
print("\n" + "=" * 60)
print("S1_Low Strategy Comparison")
print("=" * 60)

strategies = [
    ("Original SC (+5C)", original_sc_strategy),
    ("Aggressive (+30C cycling)", aggressive_strategy),
    ("Minimal (+1C tight)", minimal_strategy),
    ("Bang-Bang (+3C on/off)", bang_bang_strategy),
]

best_cost = 999
best_name = ""

for name, strategy in strategies:
    cost, log = run_with_log(strategy)
    print(f"{name:<30}: Cost = {cost:.3f}, Steps = {len(log)}")
    if cost < best_cost:
        best_cost = cost
        best_name = name

print(f"\nBest: {best_name} at {best_cost:.3f}")
print(f"Human achieved: 4.45")
print(f"Gap: {best_cost - 4.45:.3f}")

# ==========================================
# Analyze what human might have done
# ==========================================
print("\n" + "=" * 60)
print("Analysis: What might human have done differently?")
print("=" * 60)

# Check S1 task details
task = S1["tasks"][0]
print(f"\nS1 Task:")
print(f"  Target: {task['target']}°C")
print(f"  Duration: {task['duration']}s")
print(f"  Weight: {task['weight']}kg")
print(f"  Thermal Mass: {task['weight'] * 1.2} kJ/°C")

# Calculate theoretical minimum
print(f"\nTheoretical Minimum Energy:")
# Energy to heat unit from 25 to target
energy_unit = task['weight'] * 1.2 * (task['target'] - 25)
print(f"  Energy to heat unit: {energy_unit:.0f} kJ")
# Convert to kWh (1 kWh = 3600 kJ)
kwh_unit = energy_unit / 3600
print(f"  That's ~{kwh_unit:.2f} kWh")
# Plus losses during holding
hold_loss_rate = (task['target'] - 25) * 0.3  # kJ/s when valve open
hold_energy = hold_loss_rate * task['duration']
print(f"  Holding losses: ~{hold_energy:.0f} kJ ({hold_energy/3600:.2f} kWh)")
# Total minimum
min_kwh = kwh_unit + hold_energy/3600
min_cost = min_kwh * 4.5
print(f"  Theoretical minimum cost: ~{min_cost:.2f} TWD")
print(f"  (This ignores boiler losses and inefficiencies)")
