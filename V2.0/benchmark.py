import numpy as np
from boiler_env import BoilerPhysics
import os

BENCHMARK_SCENARIOS = [
    {"name": "S1_Low", "tasks": [{"name": "A", "target": 80.0, "duration": 120.0, "weight": 500.0}]},
    {"name": "S2_High", "tasks": [{"name": "A", "target": 180.0, "duration": 200.0, "weight": 800.0}]},
    {"name": "S3_Dual_Sim", "tasks": [{"name": "A", "target": 100.0, "duration": 150.0, "weight": 600.0}, {"name": "B", "target": 110.0, "duration": 150.0, "weight": 600.0}]},
    {"name": "S4_Conflict", "tasks": [{"name": "A", "target": 80.0, "duration": 100.0, "weight": 400.0}, {"name": "B", "target": 180.0, "duration": 250.0, "weight": 1200.0}]},
    {"name": "S5_Mix", "tasks": [{"name": "A", "target": 90.0, "duration": 100.0, "weight": 300.0}, {"name": "B", "target": 130.0, "duration": 180.0, "weight": 700.0}, {"name": "C", "target": 160.0, "duration": 220.0, "weight": 900.0}]},
    {"name": "S6_Heavy", "tasks": [{"name": "A", "target": 150.0, "duration": 300.0, "weight": 2500.0}]},
    {"name": "S7_Long", "tasks": [{"name": "A", "target": 120.0, "duration": 500.0, "weight": 600.0}]},
    {"name": "S8_Short", "tasks": [{"name": "A", "target": 100.0, "duration": 60.0, "weight": 200.0}, {"name": "B", "target": 120.0, "duration": 80.0, "weight": 300.0}]},
    {"name": "S9_Extreme", "tasks": [{"name": "A", "target": 160.0, "duration": 150.0, "weight": 800.0}, {"name": "B", "target": 170.0, "duration": 180.0, "weight": 900.0}, {"name": "C", "target": 180.0, "duration": 200.0, "weight": 1000.0}, {"name": "D", "target": 190.0, "duration": 220.0, "weight": 1100.0}]},
    {"name": "S10_Std", "tasks": [{"name": "A", "target": 100.0, "duration": 150.0, "weight": 600.0}, {"name": "B", "target": 150.0, "duration": 300.0, "weight": 1000.0}]}
]

def run_single_episode(model, scenario, max_steps=2000, dt=0.5):
    physics = BoilerPhysics()
    physics.reset()
    
    expected_time = 0.0
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        expected_time = max(expected_time, task["duration"])
    
    max_target = max(t["target"] for t in scenario["tasks"])
    expected_time += (max_target - 25.0) / 0.5
    
    total_time = 0.0
    done = False
    pass

def _get_obs(physics):
    rate = 0.0
    max_target, active_count, total_load = physics.get_system_state()
    return np.array([
        physics.boiler_temp / 300.0,
        max_target / 300.0,
        active_count / 4.0,
        rate,
        total_load / 2000000.0
    ], dtype=np.float32)
