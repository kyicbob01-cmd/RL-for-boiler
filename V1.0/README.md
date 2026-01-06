# Industrial Boiler Control - V1.0 Release Notes

## 1. Project Overview
**Return-Conditioned Policy (RCP)** agent for optimizing industrial boiler control.
V1.0 achieves superior performance to rule-based systems (SmartController) in heavy-load and complex scenarios.

## 2. Directory Structure (V1.0)
| File | Description |
| :--- | :--- |
| `train.py` | **Main Training Script**. Generates data and trains methods. Saves to `model.pth`. |
| `evaluate.py` | **Benchmark Script**. Compares `model.pth` against rule-based baseline. |
| `hmi.py` | **Production Interface**. The graphical SCADA interface for operators. |
| `tests.py` | **Stress Test**. Runs 50 random scenarios to verify robustness. |
| `boiler_env.py` | **Physics Engine**. Core thermodynamics logic. |
| `model.pth` | **Trained Weights**. The active model file. |

## 3. Usage Instructions

### A. Training
```bash
python train.py
```
*   Generates 10,000 episodes of expert data.
*   Trains for 500 epochs on GPU (RTX 5060 Optimized).
*   Saves result to `model.pth`.

### B. Validation
```bash
python evaluate.py
```
*   Runs standard 10 scenarios.
*   Expected Result: RCP wins ~6/10 scenarios with higher overall efficiency.

### C. Stress Testing
```bash
python tests.py
```
*   Runs 50 completely random scenarios.
*   Expected Result: RCP wins >55% of cases.

### D. HMI Operation
```bash
python hmi.py
```
*   Launches the GUI.
*   Select scenarios from the top menu and click START.

## 4. Key Configurations
*   **Inference Strategy**: `Ambitious Mode` (Target Cost = 5.0 TWD). This forces the model to perform at peak efficiency.
*   **Model Architecture**: 4-Layer MLP (6 -> 512 -> 512 -> 256 -> 1) with Dropout(0.1).
