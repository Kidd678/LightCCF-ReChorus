#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import sys
import io

# === 配置: 解决中文乱码和负号显示问题 ===
# 优先使用 SimHei (黑体)，如果没有则回退到其他字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 解决 Windows 控制台输出乱码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# === 配置路径 ===
dataset = 'Grocery_and_Gourmet_Food'  # 可选: 'Grocery_and_Gourmet_Food', 'MovieLens_1M'
base_path = 'log'
ground_truth_file = f'ReChorus/data/{dataset}/test.csv'

# === 1. 数据加载函数 ===
def find_csv_files(base_path, dataset):
    files = {}
    model_names = ['BPRMF', 'LightGCN', 'LightCCF', 'LightCCF_Imp']
    for model in model_names:
        model_path = os.path.join(base_path, model)
        if not os.path.exists(model_path): continue
        for folder in os.listdir(model_path):
            if dataset in folder:
                csv_file = os.path.join(model_path, folder, f'rec-{model}-test.csv')
                if os.path.exists(csv_file): files[model] = csv_file; break
    return files

def parse_numpy_list(x):
    if isinstance(x, str):
        import re
        numbers = re.findall(r'np\.int64\((\d+)\)', x)
        if numbers: return [int(n) for n in numbers]
        try: return ast.literal_eval(x)
        except: return []
    return x

print("正在加载全量数据...")
files = find_csv_files(base_path, dataset)
if not files:
    print("错误：未找到预测结果文件。")
    exit()

gt_df = pd.read_csv(ground_truth_file, sep='\t')
gt_dict = gt_df.groupby('user_id')['item_id'].apply(set).to_dict()

preds = {}
for model, path in files.items():
    print(f"读取 {model}...")
    df = pd.read_csv(path, sep='\t')
    preds[model] = dict(zip(df['user_id'], df['rec_items'].apply(parse_numpy_list)))

# === 2. 统计全量排名 (Global Rank Collection) ===
print("正在计算所有用户的预测排名...")
global_ranks = {m: [] for m in files.keys()}
MISS_RANK = 101 

for uid, true_items in gt_dict.items():
    target_item = list(true_items)[0] 
    
    for model in files:
        rec_list = preds[model].get(uid, [])
        if target_item in rec_list:
            r = rec_list.index(target_item) + 1
        else:
            r = MISS_RANK 
        global_ranks[model].append(r)

print(f"统计完成。样本总数: {len(list(global_ranks.values())[0])}")

# === 3. 绘图 1: 全量排名箱线图 (修复重合版) ===
def plot_global_boxplot(rank_data, output_name='global_rank_boxplot_fixed.png'):
    plt.figure(figsize=(10, 7)) #稍微增加高度
    
    models = list(rank_data.keys())
    data = [rank_data[m] for m in models]
    
    # 核心修改：预先计算好标签，包含换行符和平均值
    # 这样 Matplotlib 会自动预留空间，不再需要手动 text 定位
    new_labels = []
    for m in models:
        mean_r = np.mean(rank_data[m])
        # 使用 \n 换行，将平均值整合进 X 轴标签
        label = f"{m}\n(平均: {mean_r:.1f})"
        new_labels.append(label)

    # 绘制箱线图
    bp = plt.boxplot(data, patch_artist=True, labels=new_labels, showfliers=False,
                     medianprops={'color': 'red', 'linewidth': 1.5})
    
    colors = ['#e0e0e0', '#e0e0e0', '#a1c9f4', '#8de5a1'] 
    for patch, color in zip(bp['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    plt.title(f"全量测试集预测排名分布 ({dataset})\n(排名数值越小越好)", fontsize=14)
    plt.ylabel("预测排名位置 (Rank)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加辅助线
    plt.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Top-20 截断线')
    plt.legend(loc='upper right', prop={'size': 10}) 
    
    # 调整刻度字体大小
    plt.xticks(fontsize=11)
    
    # 自动调整布局，防止标签被切掉
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"箱线图已保存: {output_name}")

# === 4. 绘图 2: Recall@K 趋势曲线 (微调布局) ===
def plot_recall_curve(rank_data, output_name='global_recall_curve_fixed.png'):
    plt.figure(figsize=(10, 6))
    
    k_range = range(1, 101) 
    
    markers = ['o', 's', '^', '*']
    colors = ['gray', 'black', 'blue', 'green']
    
    for i, (model, ranks) in enumerate(rank_data.items()):
        ranks_arr = np.array(ranks)
        recalls = []
        for k in k_range:
            hit_count = np.sum(ranks_arr <= k)
            recall = hit_count / len(ranks_arr)
            recalls.append(recall)
            
        plt.plot(k_range, recalls, label=model, 
                 color=colors[i], linewidth=2, 
                 marker=markers[i], markevery=10)

    plt.title(f"全量测试集 Recall@K 趋势曲线 ({dataset})", fontsize=14)
    plt.xlabel("K (推荐列表长度)", fontsize=12)
    plt.ylabel("召回率 (Recall / Hit Ratio)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    
    # 重点标注 @20 位置
    plt.axvline(x=20, color='r', linestyle='--', alpha=0.3)
    # 稍微调整文字位置防止压线
    plt.text(20, 0.02, 'K=20', color='r', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"趋势图已保存: {output_name}")

# === 执行绘图 ===
plot_global_boxplot(global_ranks)
plot_recall_curve(global_ranks)