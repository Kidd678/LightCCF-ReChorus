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
# 优先尝试加载支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 解决 Windows 控制台输出乱码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# === 配置路径 (请确保与你的目录结构一致) ===
dataset = 'Grocery_and_Gourmet_Food'
base_path = 'log'
ground_truth_file = f'ReChorus/data/{dataset}/test.csv'

# === 1. 数据加载函数 (复用之前的逻辑) ===
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

print("正在加载数据...")
files = find_csv_files(base_path, dataset)
if not files:
    print("错误：未找到预测结果文件，请检查路径。")
    exit()

gt_df = pd.read_csv(ground_truth_file, sep='\t')
gt_dict = gt_df.groupby('user_id')['item_id'].apply(set).to_dict()

preds = {}
for model, path in files.items():
    print(f"读取 {model}...")
    df = pd.read_csv(path, sep='\t')
    preds[model] = dict(zip(df['user_id'], df['rec_items'].apply(parse_numpy_list)))

# === 2. 筛选典型案例 ===
print("正在筛选典型用户...")
candidates = []
# 用于统计绘图的数据列表
stats_data = {m: [] for m in files.keys()}

for uid, true_items in gt_dict.items():
    if uid not in preds.get('LightCCF', {}): continue
    target_item = list(true_items)[0] # 取一个目标物品
    
    # 获取各模型排名
    ranks = {}
    valid_sample = True
    for model in preds:
        rec_list = preds[model].get(uid, [])
        if target_item in rec_list:
            r = rec_list.index(target_item) + 1
        else:
            r = 101 # 没推荐出来，设为 101 (超过 Top-100)
        ranks[model] = r
    
    # 筛选逻辑：LightCCF 排进前 5，且 LightGCN 排在 20 名开外
    # 这种样本最能体现 LightCCF 对“难样本”的挖掘能力
    if ranks.get('LightCCF', 101) <= 5 and ranks.get('LightGCN', 0) > 20:
        candidates.append({'user': uid, 'item': target_item, 'ranks': ranks})
        # 收集数据用于统计图
        for m, r in ranks.items():
            stats_data[m].append(r)

print(f"共找到 {len(candidates)} 个 LightCCF 优势样本。")

# === 3. 绘图功能 1: 个体案例对比 (Bar Chart) ===
def plot_individual_cases(cases, output_name='case_study_bar.png'):
    num_cases = min(4, len(cases)) # 只画前4个，避免图太挤
    if num_cases == 0: return
    
    fig, axes = plt.subplots(1, num_cases, figsize=(4 * num_cases, 5), sharey=True)
    if num_cases == 1: axes = [axes]
    
    colors = ['#CCCCCC', '#999999', '#4c72b0', '#55a868'] # 灰, 灰, 蓝(LightCCF), 绿(Imp)
    model_order = ['BPRMF', 'LightGCN', 'LightCCF', 'LightCCF_Imp']
    # 确保模型存在
    model_order = [m for m in model_order if m in cases[0]['ranks']]
    
    for i, case in enumerate(cases[:num_cases]):
        ax = axes[i]
        ranks = [case['ranks'][m] for m in model_order]
        
        # 绘制柱状图
        bars = ax.bar(model_order, ranks, color=colors[:len(model_order)])
        
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            text = f'{height}' if height <= 100 else '>100'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    text, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        # === 中文替换区域 ===
        ax.set_title(f"用户 ID: {case['user']}\n目标物品 ID: {case['item']}")
        ax.set_xlabel("模型")
        if i == 0: ax.set_ylabel("推荐排名 (数值越小越好)")
        ax.set_ylim(0, 80) # 截断Y轴，突出前排差异
        
        # 画一条红线表示 Top-20 阈值
        ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Top-20 截断线')
        if i == 0: ax.legend()

    # === 中文标题 ===
    plt.suptitle(f"案例分析: 典型用户在不同模型下的推荐排名对比", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_name}")

# === 4. 绘图功能 2: 整体分布统计 (Box Plot) ===
def plot_statistics(stats_data, output_name='case_study_boxplot.png'):
    plt.figure(figsize=(8, 6))
    
    model_order = ['BPRMF', 'LightGCN', 'LightCCF', 'LightCCF_Imp']
    model_order = [m for m in model_order if m in stats_data]
    data_to_plot = [stats_data[m] for m in model_order]
    
    # 绘制箱线图
    bp = plt.boxplot(data_to_plot, patch_artist=True, labels=model_order, showfliers=False)
    
    # 美化颜色
    colors = ['#e0e0e0', '#e0e0e0', '#a1c9f4', '#8de5a1']
    for patch, color in zip(bp['boxes'], colors[:len(model_order)]):
        patch.set_facecolor(color)
        
    # === 中文替换区域 ===
    plt.title(f"{len(stats_data['LightCCF'])} 个典型“难样本”的排名分布\n(LightCCF 命中 Top-5 但 LightGCN 失败的用户)", fontsize=12)
    plt.ylabel("推荐排名 (数值越小越好)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加 Top-20 线
    plt.axhline(y=20, color='r', linestyle='--', label='Top-20 阈值')
    plt.legend()
    
    plt.savefig(output_name, dpi=300)
    print(f"图表已保存: {output_name}")

# === 执行绘图 ===
plot_individual_cases(candidates)
plot_statistics(stats_data)