import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei'] # Use SimHei for Chinese characters if installed, else fallback
plt.rcParams['axes.unicode_minus'] = False

# 1. NDCG@20 Comparison
datasets = ['Grocery', 'MovieLens-1M']
bprmf_scores = [0.1671, 0.3191]
lightgcn_scores = [0.1730, 0.3293]
lightccf_scores = [0.3200, 0.3685]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width, bprmf_scores, width, label='BPRMF')
rects2 = ax.bar(x, lightgcn_scores, width, label='LightGCN')
rects3 = ax.bar(x + width, lightccf_scores, width, label='LightCCF')

ax.set_ylabel('NDCG@20')
ax.set_title('Performance Comparison (NDCG@20)')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('ndcg_comparison.png')
print("Saved ndcg_comparison.png")

# 2. Training Time Comparison (MovieLens-1M)
models = ['BPRMF', 'LightCCF', 'LightGCN']
times = [12.0, 20.0, 150.0] # Approximate average seconds per epoch

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(models, times, color=['#1f77b4', '#2ca02c', '#d62728'])

ax.set_ylabel('Time per Epoch (s)')
ax.set_title('Training Speed on MovieLens-1M')
ax.bar_label(bars, fmt='%.1f s')

plt.tight_layout()
plt.savefig('time_comparison.png')
print("Saved time_comparison.png")

# 3. LightCCF Loss Convergence
epochs_grocery = list(range(1, 21))
loss_grocery = [1.3363, 1.1830, 1.1356, 1.1079, 1.0819, 1.0495, 1.0048, 0.9503, 0.8941, 0.8412, 
                0.7908, 0.7422, 0.6952, 0.6515, 0.6107, 0.5760, 0.5462, 0.5223, 0.5015, 0.4839]

epochs_ml = list(range(1, 13))
loss_ml = [1.2852, 1.0516, 0.8329, 0.7662, 0.7369, 0.7198, 0.7066, 0.6968, 0.6889, 0.6837, 0.6764, 0.6727]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(epochs_grocery, loss_grocery, marker='o', label='Grocery (20 Epochs)')
ax.plot(epochs_ml, loss_ml, marker='s', label='MovieLens-1M (Early Stop at 12)')

ax.set_xlabel('Epoch')
ax.set_ylabel('Total Loss')
ax.set_title('LightCCF Training Loss Convergence')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('loss_curve.png')
print("Saved loss_curve.png")