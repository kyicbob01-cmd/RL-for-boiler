"""
Benchmark System for RL Boiler Control
======================================
10 個固定測試場景 + 4 項驗證標準
"""
import numpy as np
from boiler_env import BoilerEnv, BoilerPhysics
from stable_baselines3 import PPO
import os

# ==========================================
# 10 個固定 Benchmark 場景
# ==========================================
BENCHMARK_SCENARIOS = [
    # 1. 單一單元，低溫 (Easy)
    {"name": "S1_Single_Low", "tasks": [
        {"name": "Unit_A", "target": 80.0, "duration": 120.0, "weight": 500.0}
    ]},
    
    # 2. 單一單元，高溫 (Medium)
    {"name": "S2_Single_High", "tasks": [
        {"name": "Unit_A", "target": 180.0, "duration": 200.0, "weight": 800.0}
    ]},
    
    # 3. 兩單元，相近溫度 (Easy)
    {"name": "S3_Dual_Similar", "tasks": [
        {"name": "Unit_A", "target": 100.0, "duration": 150.0, "weight": 600.0},
        {"name": "Unit_B", "target": 110.0, "duration": 150.0, "weight": 600.0}
    ]},
    
    # 4. 兩單元，大溫差 (Hard)
    {"name": "S4_Dual_BigGap", "tasks": [
        {"name": "Unit_A", "target": 80.0, "duration": 100.0, "weight": 400.0},
        {"name": "Unit_B", "target": 180.0, "duration": 250.0, "weight": 1200.0}
    ]},
    
    # 5. 三單元，混合溫度 (Medium)
    {"name": "S5_Triple_Mix", "tasks": [
        {"name": "Unit_A", "target": 90.0, "duration": 100.0, "weight": 300.0},
        {"name": "Unit_B", "target": 130.0, "duration": 180.0, "weight": 700.0},
        {"name": "Unit_C", "target": 160.0, "duration": 220.0, "weight": 900.0}
    ]},
    
    # 6. 重負載 (Hard)
    {"name": "S6_Heavy_Load", "tasks": [
        {"name": "Unit_A", "target": 150.0, "duration": 300.0, "weight": 2500.0}
    ]},
    
    # 7. 長時間製程 (Medium)
    {"name": "S7_Long_Process", "tasks": [
        {"name": "Unit_A", "target": 120.0, "duration": 500.0, "weight": 600.0}
    ]},
    
    # 8. 短時間製程 (Medium)
    {"name": "S8_Short_Process", "tasks": [
        {"name": "Unit_A", "target": 100.0, "duration": 60.0, "weight": 200.0},
        {"name": "Unit_B", "target": 120.0, "duration": 80.0, "weight": 300.0}
    ]},
    
    # 9. 極端：4 單元全高溫 (Extreme)
    {"name": "S9_Extreme_4Units", "tasks": [
        {"name": "Unit_A", "target": 160.0, "duration": 150.0, "weight": 800.0},
        {"name": "Unit_B", "target": 170.0, "duration": 180.0, "weight": 900.0},
        {"name": "Unit_C", "target": 180.0, "duration": 200.0, "weight": 1000.0},
        {"name": "Unit_D", "target": 190.0, "duration": 220.0, "weight": 1100.0}
    ]},
    
    # 10. 標準生產 (Variable - 模擬 HMI 場景 2)
    {"name": "S10_Standard", "tasks": [
        {"name": "反應槽 A", "target": 100.0, "duration": 150.0, "weight": 600.0},
        {"name": "反應槽 B", "target": 150.0, "duration": 300.0, "weight": 1000.0}
    ]}
]

# ==========================================
# 驗證標準
# ==========================================
VALIDATION_CRITERIA = {
    "completion_rate": 0.95,      # ≥ 95% 完成率
    "max_time_ratio": 1.5,        # 時間 ≤ 預期的 150%
    "max_cost_vs_baseline": 1.0,  # 能耗 ≤ Rule-based
    "max_cost_std_ratio": 0.2     # 穩定性 σ < 20%
}

# Baseline 成本 (由 Rule-based Controller 測量，需要先跑一次取得)
BASELINE_COSTS = {}  # 會在 calibrate_baseline() 填入

# ==========================================
# 評估函數
# ==========================================
def run_single_episode(model, scenario, max_steps=2000, dt=0.5):
    """
    執行單一場景並返回結果
    """
    physics = BoilerPhysics()
    physics.reset()
    
    # 載入場景
    expected_time = 0.0
    for task in scenario["tasks"]:
        physics.add_unit(task["name"], task["target"], task["duration"], task["weight"])
        expected_time = max(expected_time, task["duration"])
    
    # 加上升溫時間估計 (最高溫度 / 升溫速率)
    max_target = max(t["target"] for t in scenario["tasks"])
    heat_up_time = (max_target - 25.0) / 0.5  # 假設 0.5°C/s 升溫
    expected_time += heat_up_time
    
    obs = _get_obs(physics)
    total_time = 0.0
    done = False
    
    for step in range(max_steps):
        # 模型預測
        action, _ = model.predict(obs, deterministic=True)
        power_pct = float(action[0]) * 100.0
        
        # 物理模擬
        physics.step(power_pct, dt=dt)
        total_time += dt
        
        # 檢查完成
        _, active_count, _ = physics.get_system_state()
        if active_count == 0:
            done = True
            break
        
        obs = _get_obs(physics)
    
    return {
        "scenario": scenario["name"],
        "done": done,
        "total_time": total_time,
        "expected_time": expected_time,
        "time_ratio": total_time / expected_time if expected_time > 0 else 999,
        "cost": physics.total_cost
    }

def _get_obs(physics):
    """從 Physics 建構觀測向量"""
    rate = 0.0  # 簡化：不追蹤歷史
    max_target, active_count, total_load = physics.get_system_state()
    return np.array([
        physics.boiler_temp / 300.0,
        max_target / 300.0,
        active_count / 4.0,
        rate,
        total_load / 2000000.0
    ], dtype=np.float32)

def evaluate_model(model_path, verbose=True):
    """
    對模型進行完整 Benchmark 評估
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return None
    
    model = PPO.load(model_path)
    results = []
    
    for scenario in BENCHMARK_SCENARIOS:
        result = run_single_episode(model, scenario)
        results.append(result)
        if verbose:
            status = "✅" if result["done"] else "❌"
            print(f"{status} {result['scenario']}: "
                  f"Time={result['total_time']:.0f}s ({result['time_ratio']:.1%}), "
                  f"Cost={result['cost']:.2f} TWD")
    
    # 計算總體指標
    completion_rate = sum(r["done"] for r in results) / len(results)
    avg_time_ratio = np.mean([r["time_ratio"] for r in results if r["done"]])
    costs = [r["cost"] for r in results if r["done"]]
    avg_cost = np.mean(costs) if costs else 999
    std_cost = np.std(costs) if len(costs) > 1 else 0
    
    summary = {
        "completion_rate": completion_rate,
        "avg_time_ratio": avg_time_ratio,
        "avg_cost": avg_cost,
        "cost_std_ratio": std_cost / avg_cost if avg_cost > 0 else 999,
        "results": results
    }
    
    if verbose:
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(f"完成率: {completion_rate:.1%} (標準: ≥95%)")
        print(f"平均時間比: {avg_time_ratio:.1%} (標準: ≤150%)")
        print(f"平均成本: {avg_cost:.2f} TWD")
        print(f"成本穩定性: σ/μ = {summary['cost_std_ratio']:.1%} (標準: <20%)")
    
    return summary

def is_model_valid(summary):
    """
    檢查模型是否通過所有驗證標準
    """
    c = VALIDATION_CRITERIA
    passed = (
        summary["completion_rate"] >= c["completion_rate"] and
        summary["avg_time_ratio"] <= c["max_time_ratio"] and
        summary["cost_std_ratio"] < c["max_cost_std_ratio"]
    )
    return passed

def compare_models(model_a_path, model_b_path):
    """
    比較兩個模型，返回 B 是否在所有場景優於 A
    """
    print(f"Comparing: {model_a_path} vs {model_b_path}")
    
    model_a = PPO.load(model_a_path)
    model_b = PPO.load(model_b_path)
    
    wins_b = 0
    for scenario in BENCHMARK_SCENARIOS:
        result_a = run_single_episode(model_a, scenario)
        result_b = run_single_episode(model_b, scenario)
        
        # B 獲勝條件：完成 + 成本更低 (或 A 未完成)
        b_wins = (result_b["done"] and not result_a["done"]) or \
                 (result_b["done"] and result_a["done"] and result_b["cost"] < result_a["cost"])
        
        if b_wins:
            wins_b += 1
            print(f"✅ {scenario['name']}: B wins ({result_b['cost']:.2f} < {result_a['cost']:.2f})")
        else:
            print(f"❌ {scenario['name']}: A wins or tie")
    
    print(f"\nResult: Model B wins {wins_b}/10 scenarios")
    return wins_b == 10

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python benchmark.py <model.zip>           # 評估單一模型")
        print("  python benchmark.py compare <A.zip> <B.zip>  # 比較兩個模型")
        sys.exit(1)
    
    if sys.argv[1] == "compare" and len(sys.argv) >= 4:
        compare_models(sys.argv[2], sys.argv[3])
    else:
        model_path = sys.argv[1]
        summary = evaluate_model(model_path)
        
        if summary and is_model_valid(summary):
            print("\n🎉 MODEL PASSED ALL CRITERIA!")
        else:
            print("\n⚠️ MODEL FAILED VALIDATION")
