# Industrial Boiler Control - V3.0 Release Notes

## 1. Overview
V3.0 implements multiple advanced training strategies to surpass the **Time-Aware SmartController (SC)** baseline:
- Evolution Strategy (ES)
- Behavior Cloning -> PPO Pipeline
- Self-Improvement Loop

## 2. Directory Structure

| File | Description |
|:-----|:------------|
| `policy.py` | Shared policy network definitions (`Policy`, `RCPolicy`) |
| `time_aware_sc.py` | Time-Aware SmartController baseline |
| `boiler_env.py` | Physics simulation engine |
| `benchmark.py` | 10 standard test scenarios |
| `stage1_bc.py` | Stage 1: Behavior Cloning (BC) training |
| `stage2_ppo.py` | Stage 2: PPO fine-tuning |
| `train_es.py` | Evolution Strategy training |
| `train_es_parallel.py` | ES with multiprocessing |
| `train_self_improve.py` | Self-improvement loop training |
| `train_rcp.py` | Return-Conditioned Policy training |
| `evaluate_v3.py` | RCP evaluation (multi-target) |
| `validate_es.py` | ES model validation |
| `model_bc.pth` | BC-trained model weights |
| `model_es_best.pth` | Best ES-evolved model |
| `model_es_final.pth` | Final ES model after training |

## 3. Training Methods

### A. Two-Stage Pipeline (BC -> PPO)
```bash
python stage1_bc.py   # Behavior Cloning warm-start
python stage2_ppo.py  # PPO refinement
```

### B. Evolution Strategy
```bash
python train_es.py    # Single-process ES
```

### C. Self-Improvement
```bash
python train_self_improve.py
```

## 4. Evaluation
```bash
python validate_es.py        # ES validation
python evaluate_v3.py        # RCP standard evaluation
python evaluate_v3.py --multi  # Multi-target sweep
```

## 5. State Vector (6D)
```
[
  boiler_temp / 300.0,
  max_target_temp / 300.0,
  active_units / 4.0,
  temperature_rate,
  total_load / 2000000.0,
  min_remaining_time / 500.0
]
```

## 6. Key Improvements over V2.0
- Time-Aware SC as stronger baseline (vs original SC)
- Population-based optimization (ES) for global search
- On-policy PPO for stable fine-tuning
- Self-improvement via trajectory-level comparison
