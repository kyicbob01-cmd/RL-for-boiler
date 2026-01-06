import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# ==========================================
# 1. 統一物理引擎 (Unified Physics Engine)
# ==========================================
class BoilerPhysics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sim_speed = 1.0
        self.time_step_base = 0.5 # seconds per step
        self.T_env = 25.0
        self.boiler_temp = 25.0
        
        self.total_cost = 0.0
        self.total_kwh = 0.0
        self.current_kw = 0.0
        
        # 統一：GUI 原本是 100.0 kW，訓練是 500.0 kW
        # 決定：使用 100.0 kW (配合 GUI/真實模擬)
        self.max_power = 100.0
        
        self.heat_cap_boiler = 1500.0 # 鍋爐熱容
        self.transfer_coeff = 30.0
        
        self.units = {} 
        self.next_id = 1

    def add_unit(self, name, target, duration, weight):
        uid = self.next_id
        self.next_id += 1
        
        thermal_mass = weight * 1.2
        
        self.units[uid] = {
            'id': uid,
            'name': name,
            'target': target,
            'duration_total': duration,
            'duration_left': duration,
            'weight': weight,
            'current': self.T_env,
            'thermal_mass': thermal_mass,
            'state': 'HEATING', # HEATING, HOLDING, FINISHED
            'valve_open': False,
            'alert': False
        }
        return uid

    def remove_unit(self, uid):
        if uid in self.units:
            del self.units[uid]

    def reset_random_scenario(self, difficulty='normal'):
        """用於 RL 訓練的隨機場景生成"""
        self.reset()
        
        if difficulty == 'extreme':
            # 極端重工業情境
            num_tasks = 3
            for i in range(num_tasks):
                self.add_unit(f"Task_{i}", 160.0, 200.0, random.uniform(1500.0, 2500.0))
        else:
            # 一般隨機情境
            num_tasks = random.randint(1, 4)
            for i in range(num_tasks):
                target = random.choice([60.0, 90.0, 100.0, 130.0, 150.0, 180.0])
                weight = random.uniform(50.0, 2000.0)
                duration = random.uniform(60.0, 500.0)
                self.add_unit(f"Task_{i}", target, duration, weight)

    def get_system_state(self):
        """回傳給 RL 的觀察值輔助數據"""
        max_t = 0.0
        active_count = 0
        total_thermal_load = 0.0
        
        for uid, u in self.units.items():
            if u['state'] != 'FINISHED':
                active_count += 1
                if u['target'] > max_t: max_t = u['target']
                gap = max(0, u['target'] - u['current'])
                total_thermal_load += gap * u['weight']
                
        return max_t, active_count, total_thermal_load

    def step(self, power_percent, dt=None):
        """
        power_percent: 0.0 ~ 1.0 (或 0~100，統一這裡輸入為 0~100)
        dt: 步長 (秒)
        """
        if dt is None:
            dt = self.time_step_base
            
        # 輸入保護
        power_percent = max(0.0, min(100.0, power_percent))
        
        # 1. 鍋爐產熱
        self.current_kw = (power_percent / 100.0) * self.max_power
        
        # 效率係數 0.9，轉換係數 50.0 (kJ/s approx) -> 這是原本的 magic number
        # 保持原本物理特性的相對關係
        heat_in = self.current_kw * 50.0 * 0.9 
        
        kwh = self.current_kw * (dt / 3600.0)
        self.total_kwh += kwh
        self.total_cost += kwh * 4.5
        
        # 2. 機台運算
        total_heat_out = 0.0
        
        for uid, u in self.units.items():
            # 狀態機維護
            if u['state'] == 'HEATING':
                if u['current'] >= u['target'] - 0.5:
                    u['state'] = 'HOLDING'
            elif u['state'] == 'HOLDING':
                # 修正：只要溫度高於目標-3度，就算有效製程時間
                # 過熱也算（實際生產中過熱可能有問題，但至少製程在進行）
                if u['current'] >= u['target'] - 3.0:
                    u['duration_left'] = max(0, u['duration_left'] - dt)
                
                if u['duration_left'] <= 0:
                    u['state'] = 'FINISHED'
            
            # 閥門與熱傳遞
            u['valve_open'] = False
            if u['state'] in ['HEATING', 'HOLDING']:
                 # 簡單 Thermostat + Hysteresis
                if u['current'] < u['target'] - 0.5 and self.boiler_temp > u['current']:
                    u['valve_open'] = True
                elif u['current'] > u['target'] + 0.5:
                    u['valve_open'] = False
            
            if u['valve_open']:
                delta = max(0, self.boiler_temp - u['current'])
                transfer = delta * self.transfer_coeff
                
                # 散熱
                loss = (u['current'] - self.T_env) * 0.3
                
                # 升溫
                change = (transfer - loss) / u['thermal_mass']
                u['current'] += change * dt
                
                total_heat_out += transfer
            else:
                # 自然冷卻
                loss = (u['current'] - self.T_env) * 0.2
                u['current'] -= (loss / u['thermal_mass']) * dt
                
            # 警報 Check
            u['alert'] = (u['state'] == 'HOLDING' and u['current'] < u['target'] - 5.0)

        # 3. 鍋爐溫度更新
        loss_boiler = (self.boiler_temp - self.T_env) * 4.0
        b_change = (heat_in - loss_boiler - total_heat_out) / self.heat_cap_boiler
        self.boiler_temp += b_change * dt
        
        return self.boiler_temp

# ==========================================
# 2. Gymnasium Environment Wrapper
# ==========================================
class BoilerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.physics = BoilerPhysics()
        
        # Action: Power % (0.0 ~ 1.0) -> 會被 map 到 0~100%
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: [BoilerT, MaxTarget, ActiveCount, Rate, TotalLoad]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.max_steps = 2000 # 由於功率變小(500->100)，可能需要更多時間

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 決定難度
        difficulty = 'normal'
        if random.random() < 0.2: difficulty = 'extreme'
        
        self.physics.reset_random_scenario(difficulty)
        self.current_step = 0
        self.prev_temp = 25.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        p = self.physics
        rate = p.boiler_temp - self.prev_temp
        max_target, active_count, total_load = p.get_system_state()
        
        return np.array([
            p.boiler_temp / 300.0,
            max_target / 300.0,
            active_count / 4.0, 
            rate,
            total_load / 2000000.0
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        power_pct = float(action[0]) * 100.0
        self.prev_temp = self.physics.boiler_temp
        self.physics.step(power_pct, dt=0.5)
        
        obs = self._get_obs()
        max_target, active_count, _ = self.physics.get_system_state()
        
        # ==========================================
        # REWARD V6 - 進度導向 + 效率獎勵
        # ==========================================
        
        # 完成任務
        if active_count == 0:
            cost = self.physics.total_cost
            # 根據成本給獎勵 (目標: 比 SmartController 的 ~25 TWD 更低)
            if cost < 20.0:
                reward = 200.0  # 極佳
            elif cost < 25.0:
                reward = 150.0  # 優秀，接近人類水準
            elif cost < 30.0:
                reward = 100.0  # 良好
            elif cost < 40.0:
                reward = 50.0   # 及格
            else:
                reward = 10.0   # 至少完成了
            return obs, reward, True, False, {"cost": cost}
        
        # 超時
        if self.current_step >= self.max_steps:
            return obs, -50.0, False, True, {}
        
        # 過程中：鼓勵溫度控制
        reward = 0.0
        gap = self.physics.boiler_temp - max_target
        
        # 溫度在合理範圍內 (0 ~ +15) = 獎勵
        if 0 <= gap <= 15:
            reward = 0.5
        # 溫度過高 (> +20) = 懲罰 (浪費能源)
        elif gap > 20:
            reward = -0.2
        # 溫度不足但在努力加熱
        elif gap < 0 and power_pct > 50:
            reward = 0.1
        
        return obs, reward, False, False, {}





