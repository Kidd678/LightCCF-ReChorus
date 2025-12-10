# LightCCF 用 ReChorus 框架的复现

本项目使用 ReChorus 框架复现 LightCCF 算法，并与基线模型（BPRMF、LightGCN）进行比较。

## 1. 项目结构

```
├── LightCCF/                  # 论文官方原始代码（仅作参考）
├── ReChorus/
    ├── src/
    │   ├── models/
    │   │   ├── general/
    │   │   │   ├── LightCCF.py        # [核心] 复现的 LightCCF 模型
    │   │   │   ├── LightCCF_Imp.py    # [创新] 改进版 LightCCF 模型
    │   │   │   ├── LightGCN.py        # 基线模型
    │   │   │   └── BPRMF.py           # 基线模型
    │   ├── helpers/
    │   │   ├── BaseRunner.py     # 训练和评估循环
    │   │   └── BaseReader.py     # 数据加载
    │   └── main.py               # 程序入口
    ├── data/                     # 数据集
    │   ├── Grocery_and_Gourmet_Food/
    │   └── MovieLens_1M/
    ├── ndcg_comparison.png       # 结果可视化
    ├── time_comparison.png       # 结果可视化
    └── loss_curve.png            # 结果可视化
├── model
├── log                           # 实验日志和保存的结果
```

## 2. 环境要求

- Python 3.8+
- PyTorch
- Pandas、NumPy、Scikit-learn

安装依赖：
```bash
pip install -r requirements.txt
```

## 3. 运行方法

**重要提示**：Windows 用户需要添加 `--num_workers 0` 以避免多进程问题。

### 3.1 运行 LightCCF（目标模型）
```bash
# Grocery 数据集
python src/main.py --model_name LightCCF --dataset Grocery_and_Gourmet_Food --epoch 20 --emb_size 64 --lr 1e-3 --l2 0 --ssl_lambda 0.1 --tau 0.2 --num_workers 0

# MovieLens-1M 数据集
python src/main.py --model_name LightCCF --dataset MovieLens_1M --epoch 20 --emb_size 64 --lr 1e-3 --l2 0 --ssl_lambda 0.1 --tau 0.2 --num_workers 0
```

### 3.2 运行基线模型
```bash
# BPRMF
python src/main.py --model_name BPRMF --dataset Grocery_and_Gourmet_Food --epoch 20 --emb_size 64 --lr 1e-3 --l2 1e-4 --num_workers 0

# LightGCN
python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --epoch 20 --emb_size 64 --lr 1e-3 --l2 1e-4 --num_workers 0
```

### 3.3 运行改进模型 (LightCCF_Imp)
```bash
# 结合噪声增强与可学习温度
python src/main.py --model_name LightCCF_Imp --dataset Grocery_and_Gourmet_Food --epoch 20 --emb_size 64 --lr 1e-3 --l2 0 --ssl_lambda 0.1 --tau 0.2 --noise_eps 0.02 --learnable_tau 1 --num_workers 0
```

## 4. 实验结果

我们在两个数据集上进行了实验：`Grocery_and_Gourmet_Food` 和 `MovieLens-1M`。

### 4.1 性能对比（NDCG@20）

| 模型 | Grocery | MovieLens-1M |
|-------|---------|--------------|
| BPRMF | 0.1671  | 0.3191       |
| LightGCN| 0.1730 | 0.3293       |
| **LightCCF** | **0.3200** | **0.3685** |
| **LightCCF_Imp (创新)** | **0.3286** | - |

**分析**：
- **LightCCF** 在两个数据集上都明显优于 BPRMF 和 LightGCN。
- 在 `Grocery` 数据集上，LightCCF 的性能接近基线模型的 **2 倍**，展示了其在稀疏数据集上的优势。
- 在 `MovieLens-1M` 上，它的性能始终领先，相比 LightGCN 有约 12% 的提升。

> **结论**：
> 1. LightCCF 在稀疏数据集 (Grocery) 上实现了近 **2倍** 的性能提升。
> 2. 我们的改进模型 **LightCCF_Imp** 通过引入噪声扰动和自适应温度，在 LightCCF 的基础上进一步提升了 **2.7%** 的性能。

### 4.2 训练效率

MovieLens-1M 数据集上每个 epoch 的训练时间对比：
- **BPRMF**：~12s
- **LightCCF**：~20s
- **LightGCN**：~150s

**分析**：
- LightCCF 比 LightGCN **快 7.5 倍**。
- 通过移除昂贵的图卷积操作，并依赖对比学习进行邻域聚合，LightCCF 实现了更好的性能，同时计算效率与简单矩阵分解相当。

## 5. 实现细节

LightCCF 的实现位于 `src/models/general/LightCCF.py` 中。主要特性包括：
- **无 GCN 层**：纯嵌入向量架构。
- **邻域聚合（NA）损失**：显式地优化用户与其正样本间的相似度，同时推离其他样本，隐式地实现 GCN 的平滑效果。
- **超参数**：经实验验证，`ssl_lambda=0.1`（NA 损失权重）和 `tau=0.2`（温度参数）是最优的。


## 6. 创新与改进 (LightCCF_Imp)

为了进一步提升模型性能，我们在原论文基础上提出了两点改进，并实现了 `LightCCF_Imp` 模型：

1.  **Embedding 噪声扰动 (Noise Perturbation)**:
    *   **原理**: 在计算 NA Loss 时，对 User 和 Item 的 Embedding 施加微小的随机噪声（Noise）。
    *   **作用**: 这种机制类似于 SimGCL 中的数据增强，能够增加模型的鲁棒性，防止模型过拟合于特定的 Embedding 点，从而学到更平滑的邻域结构。
    *   **实现**: 引入超参数 `--noise_eps` 控制噪声强度。

2.  **自适应温度系数 (Adaptive Temperature)**:
    *   **原理**: 将对比学习中的温度参数 $\tau$ 设为可学习参数 (`nn.Parameter`)。
    *   **作用**: 允许模型根据训练进程自动调整对负样本的区分力度（Sharpness），避免手动调参的局限性。
    *   **实现**: 引入参数 `--learnable_tau`。

**验证结果**:
在 `Grocery` 数据集上，改进后的模型取得了 **NDCG@20 = 0.3286** 的成绩，优于原始 LightCCF 的 0.3200。