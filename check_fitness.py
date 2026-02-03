"""Quick diagnostic to compare training fitness vs validation"""
import torch
import sys
sys.path.insert(0, 'V3.0')
from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, state):
        return self.net(state)

print("Loading model_es_final.pth...")
model = Policy()
model.load_state_dict(torch.load('V3.0/model_es_final.pth', map_location='cpu'))
model.eval()

total_cost = 0
for scenario in BENCHMARK_SCENARIOS:
    physics = BoilerPhysics()
    physics.reset()
    for task in scenario['tasks']:
        physics.add_unit(task['name'], task['target'], task['duration'], task['weight'])
    prev_temp = physics.boiler_temp
    with torch.no_grad():
        for _ in range(2000):
            max_t, active, load, min_time = physics.get_system_state()
            if active == 0:
                break
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            state = torch.tensor([[
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0,
                min_time / 500.0
            ]], dtype=torch.float32)
            action = model(state).item()
            physics.step(action * 100.0, dt=0.5)
    
    print(f"{scenario['name']}: {physics.total_cost:.2f}")
    total_cost += physics.total_cost

print(f"\nTOTAL COST: {total_cost:.2f}")
print(f"Training reported: 153.79")
print(f"Difference: {total_cost - 153.79:.2f}")
