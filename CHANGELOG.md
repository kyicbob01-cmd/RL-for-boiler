# Changelog

All notable changes to this project are documented in this file.

---

## [V3.0] - 2026-02-03

### Added
- **Evolution Strategy (ES) Training** (`train_es.py`)
  - Population-based optimization
  - Mutation + Selection mechanism
  - Multi-generation training with elite preservation
  - Model outputs: `model_es_best.pth`, `model_es_final.pth`

- **Two-Stage Training Pipeline**
  - `stage1_bc.py`: Behavior Cloning from Time-Aware SC
  - `stage2_ppo.py`: PPO Fine-tuning to surpass SC

- **Self-Improvement Training** (`train_self_improve.py`)
  - Iterative self-evolution via trajectory comparison
  - Win-rate based model selection

- **Time-Aware SmartController** (`time_aware_sc.py`)
  - Enhanced baseline incorporating remaining time

- **RCP Evaluation** (`evaluate_v3.py`)
  - Multi-target cost evaluation (0.8x ~ 1.0x SC cost)
  - Comprehensive benchmark comparison

- **ES Validation** (`validate_es.py`)
  - Task completion verification
  - SC baseline comparison

### Changed
- Policy architecture refactored to standalone `policy.py`
- State vector extended to 6D: `[temp, max_target, active_units, rate, load, min_remaining_time]`

### Architecture
- `Policy`: 6D -> 512 -> 512 -> 256 -> 1 (Unconditional)
- `RCPolicy`: 7D -> 512 -> 512 -> 256 -> 1 (Return-Conditioned)

---

## [V2.0] - 2026-01-20

### Added
- **Teacher-Student Distillation** (`train_v2.py`)
  - V1.0 RCP as Teacher
  - `UnconditionalPolicy` as Student (no target_cost conditioning)
  - CPU-based data collection for physics consistency

- **3-Way Verification** (`evaluate_v2.py`)
  - Compares SmartController vs V1.0 vs V2.0

### Changed
- **UnconditionalPolicy Architecture**
  - 6D input (Time-Aware state): `[temp, max_target, active, rate, load, min_time]`
  - Larger network: 1024 -> 1024 -> 512 -> 1

- **Stratified Difficulty Filtering**
  - Fixes S9 (Heavy-Load) data loss issue

### Key Commits
| Commit | Description |
|--------|-------------|
| `8e746c3` | CPU-Based Direct Cloning with Original Physics |
| `70d1bd3` | Unconditional Elite Cloning Architecture |
| `88b521e` | Stratified Difficulty Filtering for S9 fix |
| `99aec2e` | Batched GPU Simulation (100x speedup) |
| `80582db` | Self-Training Loop (Teacher-Student Distillation) |

---

## [V1.0] - 2025-12-15

### Added
- **Return-Conditioned Policy (RCP)** (`train.py`)
  - Offline RL training from SmartController trajectories
  - 4-Layer MLP: 6D + 1 (target_cost) -> 512 -> 512 -> 256 -> 1
  - Multi-strategy data collection (Baseline, Aggressive, Conservative, SuperSaver)

- **SmartController Baseline**
  - Rule-based boiler control logic (gap-based power scheduling)

- **Production HMI** (`hmi.py`)
  - SCADA-style graphical interface for operators

- **Benchmark Suite** (`benchmark.py`)
  - 10 standard scenarios (S1 Cold Start ~ S10 Complex Multi-Phase)

- **Stress Testing** (`tests.py`)
  - 50 random scenario validation

### Model Performance
- Wins ~6/10 benchmark scenarios vs SmartController
- Wins >55% on random stress tests

### Architecture
- **RCPolicy**: 7D (state + target_cost) -> Action
- **Inference**: Ambitious Mode (target_cost = 5.0 TWD) for peak efficiency

---

## Version Summary

| Version | Method | Key Innovation | Based On |
|---------|--------|----------------|----------|
| V1.0 | RCP (Offline RL) | Return-Conditioned Policy | SmartController |
| V2.0 | Distillation | Teacher-Student Learning | V1.0 RCP |
| V3.0 | ES + BC + PPO | Population-based Evolution | Time-Aware SC |
