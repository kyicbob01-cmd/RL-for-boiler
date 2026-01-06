import matplotlib.pyplot as plt
import numpy as np
import os

# Data from the latest benchmark run (Final Victory)
scenarios = ['S1 (Low)', 'S2 (High)', 'S3 (Dual)', 'S4 (Gap)', 'S5 (Mix)', 
             'S6 (Heavy)', 'S7 (Long)', 'S8 (Short)', 'S9 (Extreme)', 'S10 (Std)']
sc_costs = [4.53, 17.46, 9.38, 22.51, 19.37, 26.05, 12.73, 7.11, 37.34, 18.12]
rcp_costs = [8.63, 16.43, 10.94, 20.74, 18.43, 23.35, 12.72, 7.86, 34.89, 17.35]

# Ensure directory exists
output_dir = "presentation_assets"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('ggplot')

# ==========================================
# Chart 1: Overall Cost Comparison (Bar)
# ==========================================
plt.figure(figsize=(12, 6))
x = np.arange(len(scenarios))
width = 0.35

plt.bar(x - width/2, sc_costs, width, label='SmartController (Rules)', color='#e74c3c')
plt.bar(x + width/2, rcp_costs, width, label='RCP Agent (AI)', color='#3498db')

plt.ylabel('Cost (TWD)')
plt.title('Benchmark Results: Rule-Based vs AI Control')
plt.xticks(x, scenarios, rotation=45)
plt.legend()

# Annotate wins
for i in range(len(scenarios)):
    diff = sc_costs[i] - rcp_costs[i]
    if diff > 1.0: # RCP significant win
        plt.text(x[i] + width/2, rcp_costs[i] + 0.5, '★', ha='center', color='blue', fontsize=14)
    elif diff < -1.0: # SC significant win
        plt.text(x[i] - width/2, sc_costs[i] + 0.5, '★', ha='center', color='red', fontsize=14)

plt.tight_layout()
plt.savefig(f"{output_dir}/chart_benchmark.png", dpi=100)
print(f"Generated {output_dir}/chart_benchmark.png")

# ==========================================
# Chart 2: Efficiency Analysis (Simple vs Complex)
# ==========================================
plt.figure(figsize=(8, 6))

# Classify scenarios
complex_indices = [3, 4, 5, 8, 9] # S4, S5, S6, S9, S10
simple_indices = [0, 1, 2, 6, 7]  # S1, S2, S3, S7, S8

avg_sc_complex = np.mean([sc_costs[i] for i in complex_indices])
avg_rcp_complex = np.mean([rcp_costs[i] for i in complex_indices])

avg_sc_simple = np.mean([sc_costs[i] for i in simple_indices])
avg_rcp_simple = np.mean([rcp_costs[i] for i in simple_indices])

categories = ['Simple Scenarios', 'Complex Scenarios']
values_sc = [avg_sc_simple, avg_sc_complex]
values_rcp = [avg_rcp_simple, avg_rcp_complex]

x_cat = np.arange(len(categories))
width = 0.35

plt.bar(x_cat - width/2, values_sc, width, label='SmartController', color='#e74c3c')
plt.bar(x_cat + width/2, values_rcp, width, label='RCP Agent', color='#3498db')

# Add percentage labels
gain_complex = (avg_sc_complex - avg_rcp_complex) / avg_sc_complex * 100
gain_simple = (avg_sc_simple - avg_rcp_simple) / avg_sc_simple * 100

plt.text(x_cat[1], max(values_sc[1], values_rcp[1]) + 1, f"AI +{gain_complex:.1f}% Efficiency", ha='center', fontweight='bold', color='green')
plt.text(x_cat[0], max(values_sc[0], values_rcp[0]) + 1, f"Rules Better ({gain_simple:.1f}%)", ha='center', fontweight='bold', color='red')

plt.ylabel('Average Cost (TWD)')
plt.title('Performance by Task Complexity')
plt.xticks(x_cat, categories)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/chart_efficiency.png", dpi=100)
print(f"Generated {output_dir}/chart_efficiency.png")

# ==========================================
# Chart 3: The "S1" Outlier Analysis
# ==========================================
plt.figure(figsize=(6, 4))
labels = ['S1 (Low Temp)', 'All Other 9 Scenarios']
sc_sum = [sc_costs[0], sum(sc_costs[1:])]
rcp_sum = [rcp_costs[0], sum(rcp_costs[1:])]

x_s1 = np.arange(len(labels))
plt.bar(x_s1 - width/2, sc_sum, width, label='SmartController', color='#e74c3c')
plt.bar(x_s1 + width/2, rcp_sum, width, label='RCP Agent', color='#3498db')

diff_others = sc_sum[1] - rcp_sum[1]
plt.text(1, max(sc_sum[1], rcp_sum[1]) + 5, f"AI Wins by {diff_others:.1f} TWD", ha='center', color='green', fontweight='bold')

plt.title('Impact of the S1 Outlier')
plt.xticks(x_s1, labels)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/chart_outlier.png", dpi=100)
print(f"Generated {output_dir}/chart_outlier.png")
