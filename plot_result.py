import matplotlib.pyplot as plt
import numpy as np
import sys
import io

# === 配置: 解决中文乱码和负号显示问题 ===
# 优先使用 SimHei (黑体)，如果没有则回退到其他字体 (适配 Windows/Mac/Linux)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 解决 Windows 控制台输出乱码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置绘图风格
plt.style.use('ggplot')

# ==========================================
# 图表 1: NDCG@20 性能对比 (簇状柱形图)
# ==========================================
datasets = ['Grocery (稀疏)', 'MovieLens-1M (稠密)']  # 加上中文注释更清晰
bprmf_scores = [0.1671, 0.3191]
lightgcn_scores = [0.1730, 0.3293]
lightccf_scores = [0.3200, 0.3685]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width, bprmf_scores, width, label='BPRMF', color='#999999') # 灰色
rects2 = ax.bar(x, lightgcn_scores, width, label='LightGCN', color='#4c72b0')   # 蓝色
rects3 = ax.bar(x + width, lightccf_scores, width, label='LightCCF', color='#55a868') # 绿色 (突出)

ax.set_ylabel('NDCG@20')
ax.set_title('不同模型 NDCG@20 性能对比')
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
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('ndcg_comparison_cn.png', dpi=300)
print("已保存: ndcg_comparison_cn.png")

# ==========================================
# 图表 2: 训练时间对比 (ML-1M 数据集)
# ==========================================
models = ['BPRMF', 'LightCCF', 'LightGCN']
times = [12.0, 20.0, 150.0] # Approximate average seconds per epoch

fig, ax = plt.subplots(figsize=(8, 6))
# 颜色分配：LightCCF 用绿色突出，LightGCN 用红色警示慢，BPRMF 用蓝色
bars = ax.bar(models, times, color=['#1f77b4', '#55a868', '#d62728'])

ax.set_ylabel('单轮训练耗时 (秒)')
ax.set_title('MovieLens-1M 训练效率对比 (越低越好)')
ax.bar_label(bars, fmt='%.1f 秒', padding=3)

plt.tight_layout()
plt.savefig('time_comparison_cn.png', dpi=300)
print("已保存: time_comparison_cn.png")

# ==========================================
# 图表 3: LightCCF 训练 Loss 收敛曲线
# ==========================================
epochs_grocery = list(range(1, 21))
loss_grocery = [1.3363, 1.1830, 1.1356, 1.1079, 1.0819, 1.0495, 1.0048, 0.9503, 0.8941, 0.8412, 
                0.7908, 0.7422, 0.6952, 0.6515, 0.6107, 0.5760, 0.5462, 0.5223, 0.5015, 0.4839]

epochs_ml = list(range(1, 13))
loss_ml = [1.2852, 1.0516, 0.8329, 0.7662, 0.7369, 0.7198, 0.7066, 0.6968, 0.6889, 0.6837, 0.6764, 0.6727]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(epochs_grocery, loss_grocery, marker='o', label='Grocery (跑满20轮)', linewidth=2)
ax.plot(epochs_ml, loss_ml, marker='s', label='MovieLens-1M (12轮早停)', linewidth=2)

ax.set_xlabel('训练轮次 (Epoch)')
ax.set_ylabel('总损失 (Total Loss)')
ax.set_title('LightCCF 训练损失收敛曲线')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('loss_curve_cn.png', dpi=300)
print("已保存: loss_curve_cn.png")