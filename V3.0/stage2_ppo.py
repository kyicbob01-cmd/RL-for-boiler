import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from boiler_env import BoilerPhysics
from benchmark import BENCHMARK_SCENARIOS
from time_aware_sc import TimeAwareSC

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
        self.value_head = nn.Sequential(
            nn.Linear(6, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, state):
        return self.net(state)
    
    def get_action(self, state, deterministic=False):
        mean = self.forward(state)
        if deterministic:
            return mean, torch.zeros(1)
        std = torch.exp(self.log_std).clamp(0.01, 0.5)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0, 1)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def get_value(self, state):
        return self.value_head(state)
    
    def evaluate(self, state, action):
        mean = self.forward(state)
        std = torch.exp(self.log_std).clamp(0.01, 0.5)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value_head(state).squeeze(-1)
        return log_prob, entropy, value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 2] Device: {device}")

class PPOConfig:
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5
    ppo_epochs = 10
    batch_size = 32768
    rollout_steps = 65536

class RolloutBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.states = np.zeros((max_size, 6), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
    
    def add(self, state, action, reward, value, log_prob, done):
        if self.ptr < self.max_size:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.dones[self.ptr] = done
            self.ptr += 1
    
    def clear(self):
        self.ptr = 0
    
    def get(self):
        n = self.ptr
        return (
            torch.tensor(self.states[:n], dtype=torch.float32, device=device),
            torch.tensor(self.actions[:n], dtype=torch.float32, device=device),
            torch.tensor(self.rewards[:n], dtype=torch.float32, device=device),
            torch.tensor(self.values[:n], dtype=torch.float32, device=device),
            torch.tensor(self.log_probs[:n], dtype=torch.float32, device=device),
            torch.tensor(self.dones[:n], dtype=torch.float32, device=device)
        )

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = advantages + values.to(device)
    
    return advantages, returns

def evaluate_policy(model):
    model.eval()
    sc = TimeAwareSC()
    
    victories = 0
    total_drl = 0
    total_sc = 0
    
    for scenario in BENCHMARK_SCENARIOS:
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        for _ in range(2000):
            _, active, _, _ = physics.get_system_state()
            if active == 0: break
            power = sc.decide(physics.boiler_temp, physics.units)
            physics.step(power, dt=0.5)
        sc_cost = physics.total_cost
        total_sc += sc_cost
        
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        
        with torch.no_grad():
            for _ in range(2000):
                max_t, active, load, min_time = physics.get_system_state()
                if active == 0: break
                
                rate = physics.boiler_temp - prev_temp
                prev_temp = physics.boiler_temp
                
                state = torch.tensor([[
                    physics.boiler_temp / 300.0,
                    max_t / 300.0,
                    active / 4.0,
                    rate,
                    load / 2000000.0,
                    min_time / 500.0
                ]], dtype=torch.float32).to(device)
                
                action = model(state).item()
                power = action * 100.0
                physics.step(power, dt=0.5)
        
        drl_cost = physics.total_cost
        total_drl += drl_cost
        
        if drl_cost < sc_cost:
            victories += 1
    
    model.train()
    return victories, total_drl, total_sc

def collect_rollout(model, buffer, config):
    import random
    
    steps = 0
    episode_rewards = []
    
    while steps < config.rollout_steps:
        scenario = random.choice(BENCHMARK_SCENARIOS)
        
        physics = BoilerPhysics()
        physics.reset()
        for task in scenario["tasks"]:
            physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        
        prev_temp = physics.boiler_temp
        prev_cost = 0
        ep_reward = 0
        prev_load = physics.get_system_state()[2]
        
        for step in range(2000):
            max_t, active, load, min_time = physics.get_system_state()
            if active == 0:
                ep_reward += 10.0
                buffer.add(state, action_np, 10.0, value.cpu().item(), log_prob.cpu().item(), 1.0)
                break
            
            rate = physics.boiler_temp - prev_temp
            prev_temp = physics.boiler_temp
            
            state = np.array([
                physics.boiler_temp / 300.0,
                max_t / 300.0,
                active / 4.0,
                rate,
                load / 2000000.0,
                min_time / 500.0
            ], dtype=np.float32)
            
            state_t = torch.tensor([state], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action, log_prob = model.get_action(state_t, deterministic=False)
                value = model.get_value(state_t)
            
            action_np = action.cpu().numpy()[0, 0]
            power = action_np * 100.0
            
            physics.step(power, dt=0.5)
            
            cost_delta = physics.total_cost - prev_cost
            cost_reward = -cost_delta * 0.1
            prev_cost = physics.total_cost
            
            curr_load = physics.get_system_state()[2]
            progress = (prev_load - curr_load) / 10000.0
            prev_load = curr_load
            
            temp_gap = abs(physics.boiler_temp - max_t)
            temp_reward = -temp_gap / 100.0 if temp_gap > 5 else 0.1
            
            reward = cost_reward + progress + temp_reward
            
            done = False
            
            buffer.add(
                state,
                action_np,
                reward,
                value.cpu().item(),
                log_prob.cpu().item(),
                float(done)
            )
            
            ep_reward += reward
            steps += 1
            
            if steps >= config.rollout_steps:
                break
        else:
            if active > 0:
                penalty = -5.0 * active
                ep_reward += penalty
        
        episode_rewards.append(ep_reward)
    return np.mean(episode_rewards) if episode_rewards else 0

def ppo_update(model, optimizer, buffer, config):
    states, actions, rewards, values, old_log_probs, dones = buffer.get()
    
    advantages, returns = compute_gae(
        rewards.cpu().numpy(), 
        values, 
        dones.cpu().numpy(),
        config.gamma, 
        config.gae_lambda
    )
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    
    for _ in range(config.ppo_epochs):
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, config.batch_size):
            end = start + config.batch_size
            batch_idx = indices[start:end]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            log_probs, entropy, values_pred = model.evaluate(
                batch_states, 
                batch_actions.unsqueeze(-1)
            )
            
            ratio = torch.exp(log_probs - batch_old_log_probs)
            clip_ratio = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
            policy_loss = -torch.min(ratio * batch_advantages, clip_ratio * batch_advantages).mean()
            
            value_loss = nn.MSELoss()(values_pred, batch_returns)
            
            entropy_loss = -entropy.mean()
            
            loss = (
                policy_loss + 
                config.value_coef * value_loss + 
                config.entropy_coef * entropy_loss
            )
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    
    buffer.clear()
    
    return policy_loss.item(), value_loss.item()

def train_ppo(max_updates=1000, eval_interval=10, target_victories=10):
    print("[Stage 2] Loading BC-pretrained model...")
    
    torch.set_float32_matmul_precision('high')
    
    model = Policy().to(device)
    bc_path = os.path.join(current_dir, "model_bc.pth")
    
    if os.path.exists(bc_path):
        state_dict = torch.load(bc_path, map_location=device)
        model.net.load_state_dict(state_dict)
        print(f"[Stage 2] Loaded: {bc_path}")
    else:
        print("[Stage 2] WARNING: BC model not found, starting from scratch!")
    
    model.train()
    
    config = PPOConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    buffer = RolloutBuffer(max_size=config.rollout_steps + 10000)
    
    best_victories = 0
    consecutive_max = 0
    
    print(f"\n[Stage 2] PPO Training (Target: {target_victories}/10 victories)")
    print("-" * 60)
    
    for update in range(max_updates):
        avg_reward = collect_rollout(model, buffer, config)
        
        policy_loss, value_loss = ppo_update(model, optimizer, buffer, config)
        
        if (update + 1) % eval_interval == 0:
            victories, drl_total, sc_total = evaluate_policy(model)
            
            status = ">>>" if victories > best_victories else "   "
            print(f"{status} Update {update+1}: Victories={victories}/10, "
                  f"DRL={drl_total:.2f}, SC={sc_total:.2f}, "
                  f"Policy Loss={policy_loss:.4f}")
            
            if victories >= 6:
                if victories > best_victories:
                    best_victories = victories
                    best_path = os.path.join(current_dir, "model_best.pth")
                    torch.save(model.state_dict(), best_path)
                    print(f"    [Checkpoint] Saved best model ({victories} victories)")
            
            if victories >= target_victories:
                consecutive_max += 1
                if consecutive_max >= 3:
                    print(f"\n[Stage 2] SUCCESS! Achieved {victories}/10 victories (3x consecutive)")
                    final_path = os.path.join(current_dir, "model_final.pth")
                    torch.save(model.state_dict(), final_path)
                    print(f"[Stage 2] Final model saved: {final_path}")
                    return model
            else:
                consecutive_max = 0
    
    print(f"\n[Stage 2] Training completed. Best: {best_victories}/10 victories")
    return model

if __name__ == "__main__":
    bc_path = os.path.join(current_dir, "model_bc.pth")
    if not os.path.exists(bc_path):
        print("[Stage 2] ERROR: Run stage1_bc.py first to create model_bc.pth")
        sys.exit(1)
    
    model = train_ppo(max_updates=500, eval_interval=5, target_victories=10)
    
    print("\n" + "=" * 60)
    print("[Stage 2] Final Evaluation")
    print("=" * 60)
    
    victories, drl_total, sc_total = evaluate_policy(model)
    improvement = ((sc_total - drl_total) / sc_total) * 100
    
    print(f"DRL Total Cost: {drl_total:.2f}")
    print(f"SC Total Cost:  {sc_total:.2f}")
    print(f"Victories:      {victories}/10")
    print(f"Improvement:    {improvement:+.1f}%")
