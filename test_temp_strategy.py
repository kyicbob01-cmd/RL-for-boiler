"""
Constant Temperature vs Cycling Temperature Strategy Comparison
Question: Is maintaining 180°C constant better than cycling 180-190°C?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "V1.0"))

from boiler_env import BoilerPhysics

# ==========================================
# Strategy A: Constant Temperature
# ==========================================
class ConstantTempSC:
    def __init__(self, margin=3.0):
        self.margin = margin
    
    def decide(self, temp, units, target_temp):
        gap = (target_temp + self.margin) - temp
        
        if gap > 2.0:
            return 50.0  # Gentle heating
        elif gap > 0:
            return 20.0  # Fine-tune
        elif gap < -1.0:
            return 0.0   # Too hot
        else:
            return 10.0  # Maintain

# ==========================================
# Strategy B: Cycling Temperature (Pulse)
# ==========================================
class CyclingTempSC:
    def __init__(self, low_margin=0.0, high_margin=10.0):
        self.low_margin = low_margin
        self.high_margin = high_margin
    
    def decide(self, temp, units, target_temp):
        low_bound = target_temp + self.low_margin
        high_bound = target_temp + self.high_margin
        
        # If below low bound, heat to high bound
        if temp < low_bound:
            return 100.0  # Full blast
        # If above high bound, coast down
        elif temp > high_bound:
            return 0.0
        # If coasting down, continue coasting until we hit low
        else:
            # Hysteresis: if we were heating, continue until high
            # if we were coasting, continue until low
            # For simplicity, use a threshold-based approach
            if temp < (low_bound + high_bound) / 2:
                return 100.0  # Heat up
            else:
                return 0.0    # Coast down

# ==========================================
# Test Scenario: Single Heavy Unit at 180°C
# ==========================================
def run_holding_test(controller_class, **kwargs):
    """Test holding phase energy consumption"""
    physics = BoilerPhysics()
    physics.reset()
    
    # Add a heavy unit at 180°C (simulates holding phase)
    # We'll manually set it to already be at target
    physics.add_unit("Heavy", 180.0, 300.0, 1500.0)  # 300s hold time
    
    # Pre-heat boiler and unit to target (instant setup)
    physics.boiler_temp = 185.0
    for u in physics.units.values():
        u['current'] = 179.0  # Just below target
        u['state'] = 'HOLDING'
    
    controller = controller_class(**kwargs)
    
    # Run simulation
    steps = 0
    for _ in range(2000):
        _, active, _ = physics.get_system_state()
        if active == 0: break
        
        power = controller.decide(physics.boiler_temp, physics.units, 180.0)
        physics.step(power, dt=0.5)
        steps += 1
    
    return physics.total_cost, steps

def run_heating_test(controller_class, **kwargs):
    """Test heating phase energy consumption"""
    physics = BoilerPhysics()
    physics.reset()
    
    # Add a heavy unit at 180°C (needs heating from cold)
    physics.add_unit("Heavy", 180.0, 100.0, 1500.0)  # Short hold to focus on heating
    
    controller = controller_class(**kwargs)
    
    # Run simulation
    steps = 0
    for _ in range(2000):
        _, active, _ = physics.get_system_state()
        if active == 0: break
        
        power = controller.decide(physics.boiler_temp, physics.units, 180.0)
        physics.step(power, dt=0.5)
        steps += 1
    
    return physics.total_cost, steps

def main():
    print("=" * 70)
    print("Constant vs Cycling Temperature Strategy Comparison")
    print("=" * 70)
    
    # Test 1: Holding phase (already at temperature)
    print("\n[Test 1: HOLDING PHASE - Unit already at 180C]")
    print("-" * 50)
    
    cost_const, steps_const = run_holding_test(ConstantTempSC, margin=3.0)
    cost_cycle_5, steps_cycle_5 = run_holding_test(CyclingTempSC, low_margin=0.0, high_margin=5.0)
    cost_cycle_10, steps_cycle_10 = run_holding_test(CyclingTempSC, low_margin=0.0, high_margin=10.0)
    cost_cycle_15, steps_cycle_15 = run_holding_test(CyclingTempSC, low_margin=0.0, high_margin=15.0)
    
    print(f"{'Strategy':<30} | {'Cost':>10} | {'Steps':>8}")
    print("-" * 55)
    print(f"{'Constant +3C':<30} | {cost_const:>10.2f} | {steps_const:>8}")
    print(f"{'Cycling +0 to +5C':<30} | {cost_cycle_5:>10.2f} | {steps_cycle_5:>8}")
    print(f"{'Cycling +0 to +10C':<30} | {cost_cycle_10:>10.2f} | {steps_cycle_10:>8}")
    print(f"{'Cycling +0 to +15C':<30} | {cost_cycle_15:>10.2f} | {steps_cycle_15:>8}")
    
    # Test 2: Heating phase (from cold)
    print("\n[Test 2: HEATING PHASE - Unit from cold to 180C]")
    print("-" * 50)
    
    cost_const, steps_const = run_heating_test(ConstantTempSC, margin=3.0)
    cost_cycle_5, steps_cycle_5 = run_heating_test(CyclingTempSC, low_margin=0.0, high_margin=5.0)
    cost_cycle_10, steps_cycle_10 = run_heating_test(CyclingTempSC, low_margin=0.0, high_margin=10.0)
    cost_cycle_15, steps_cycle_15 = run_heating_test(CyclingTempSC, low_margin=0.0, high_margin=15.0)
    
    print(f"{'Strategy':<30} | {'Cost':>10} | {'Steps':>8}")
    print("-" * 55)
    print(f"{'Constant +3C (rush then hold)':<30} | {cost_const:>10.2f} | {steps_const:>8}")
    print(f"{'Cycling +0 to +5C':<30} | {cost_cycle_5:>10.2f} | {steps_cycle_5:>8}")
    print(f"{'Cycling +0 to +10C':<30} | {cost_cycle_10:>10.2f} | {steps_cycle_10:>8}")
    print(f"{'Cycling +0 to +15C':<30} | {cost_cycle_15:>10.2f} | {steps_cycle_15:>8}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("- Lower cost = better strategy")
    print("- Cycling may help during HEATING (faster transfer)")
    print("- Cycling may hurt during HOLDING (extra energy to reach high bound)")
    print("=" * 70)

if __name__ == "__main__":
    main()
