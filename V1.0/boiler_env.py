"""
Boiler Physics Engine
Simulates thermodynamics of a central boiler and multiple consumer units.
"""
import random

class BoilerPhysics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time_step_base = 0.5
        self.T_env = 25.0
        self.boiler_temp = 25.0
        
        self.total_cost = 0.0
        self.total_kwh = 0.0
        self.current_kw = 0.0
        self.max_power = 100.0
        
        self.heat_cap_boiler = 1500.0
        self.transfer_coeff = 30.0
        
        self.units = {} 
        self.next_id = 1

    def add_unit(self, name, target, duration, weight):
        uid = self.next_id
        self.next_id += 1
        
        self.units[uid] = {
            'id': uid, 'name': name, 'target': target,
            'duration_total': duration, 'duration_left': duration,
            'weight': weight, 'current': self.T_env,
            'thermal_mass': weight * 1.2,
            'state': 'HEATING', 'valve_open': False, 'alert': False
        }
        return uid

    def reset_random_scenario(self, difficulty='normal'):
        self.reset()
        if difficulty == 'extreme':
            for i in range(3):
                self.add_unit(f"Task_{i}", 160.0, 200.0, random.uniform(1500.0, 2500.0))
        else:
            for i in range(random.randint(1, 4)):
                target = random.choice([60.0, 90.0, 100.0, 130.0, 150.0, 180.0])
                self.add_unit(f"Task_{i}", target, random.uniform(60.0, 500.0), random.uniform(50.0, 2000.0))

    def get_system_state(self):
        max_t = 0.0
        active_count = 0
        total_thermal_load = 0.0
        
        for u in self.units.values():
            if u['state'] != 'FINISHED':
                active_count += 1
                if u['target'] > max_t: max_t = u['target']
                gap = max(0, u['target'] - u['current'])
                total_thermal_load += gap * u['weight']
                
        return max_t, active_count, total_thermal_load

    def step(self, power_percent, dt=None):
        if dt is None: dt = self.time_step_base
        
        power_percent = max(0.0, min(100.0, power_percent))
        self.current_kw = (power_percent / 100.0) * self.max_power
        
        # Thermodynamics: 1 kW = ~50 kJ/s heat in (efficiency 0.9)
        heat_in = self.current_kw * 50.0 * 0.9 
        
        kwh = self.current_kw * (dt / 3600.0)
        self.total_kwh += kwh
        self.total_cost += kwh * 4.5
        
        total_heat_out = 0.0
        
        for u in self.units.values():
            # State Machine
            if u['state'] == 'HEATING':
                if u['current'] >= u['target'] - 0.5: u['state'] = 'HOLDING'
            elif u['state'] == 'HOLDING':
                if u['current'] >= u['target'] - 3.0:
                    u['duration_left'] = max(0, u['duration_left'] - dt)
                if u['duration_left'] <= 0: u['state'] = 'FINISHED'
            
            # Valve & Temperature Physics
            u['valve_open'] = False
            if u['state'] in ['HEATING', 'HOLDING']:
                if u['current'] < u['target'] - 0.5 and self.boiler_temp > u['current']:
                    u['valve_open'] = True
                elif u['current'] > u['target'] + 0.5:
                    u['valve_open'] = False
            
            if u['valve_open']:
                delta = max(0, self.boiler_temp - u['current'])
                transfer = delta * self.transfer_coeff
                loss = (u['current'] - self.T_env) * 0.3
                u['current'] += ((transfer - loss) / u['thermal_mass']) * dt
                total_heat_out += transfer
            else:
                loss = (u['current'] - self.T_env) * 0.2
                u['current'] -= (loss / u['thermal_mass']) * dt
                
            u['alert'] = (u['state'] == 'HOLDING' and u['current'] < u['target'] - 5.0)

        loss_boiler = (self.boiler_temp - self.T_env) * 4.0
        b_change = (heat_in - loss_boiler - total_heat_out) / self.heat_cap_boiler
        self.boiler_temp += b_change * dt
        
        return self.boiler_temp
