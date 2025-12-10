# LightCCF 用 ReChorus 框架的复现

本项目使用 ReChorus 框架复现 LightCCF 算法，并与基线模型（BPRMF、LightGCN）进行比较。

## 1. 项目结构

```
ReChorus/
├── src/
│   ├── models/
│   │   ├── general/
│   │   │   ├── LightCCF.py   # [核心] 复现的 LightCCF 模型
│   │   │   ├── LightGCN.py   # 基线模型
│   │   │   └── BPRMF.py      # 基线模型
│   ├── helpers/
│   │   ├── BaseRunner.py     # 训练和评估循环
│   │   └── BaseReader.py     # 数据加载
│   └── main.py               # 程序入口
├── data/                     # 数据集
│   ├── Grocery_and_Gourmet_Food/
│   └── MovieLens_1M/
├── log/                      # 实验日志和保存的结果
# LightCCF + ReChorus — 复现实验仓库

本仓库将 `LightCCF` 的实现与 `ReChorus` 实验框架放在同一项目中，便于复现实验并与基线模型（BPRMF、LightGCN）比较。仓库包含两部分：`LightCCF/`（轻量复现实现）和 `ReChorus/`（实验框架与数据）。

**快速概览**：
- 根目录包含 `LightCCF/`、`ReChorus/`、以及用于存放模型/日志的 `model/` 和 `log/`（ `model/` 与 `log/` 需要运行后才会产生）。

## 1. 仓库结构（简要）

```
./
├── LightCCF/                # LightCCF 的简洁实现（脚本、工具、数据说明）
├── ReChorus/                # ReChorus 实验框架（src/, data/, docs/ 等）
│   ├── src/
│   │   ├── models/          # 各类模型实现（general/ 包含 LightCCF, LightGCN, BPRMF）
│   │   └── main.py          # ReChorus 的程序入口（训练/评估）
│   └── data/                # 示例数据集（请根据需要下载或准备）
├── model/                   # 训练得到的模型权重（运行后产生）
├── log/                     # 训练日志与结果（运行后产生）
├── README.md                # 本文件
└── requirements.txt         # 依赖（位于 ReChorus/ ）
```

## 2. 环境要求

- Python 3.8+
- PyTorch（与 CUDA 版本匹配，可选 GPU）
- pandas, numpy, scikit-learn, tqdm 等常用库

安装依赖：

```powershell
`pip install -r ReChorus/requirements.txt`
```


## 3. 运行说明

说明：`ReChorus/src/main.py` 是实验框架的入口，用于训练与评估多种模型；`LightCCF/` 下也包含用于单独运行或演示的小脚本。

**重要（Windows）**：在 Windows 平台上运行 `ReChorus` 时，请添加 `--num_workers 0` 以避免多进程问题。

示例命令（在仓库根目录运行）：

```powershell
# 运行 ReChorus 中的 LightCCF（Grocery 数据集）
python ReChorus/src/main.py --model_name LightCCF --dataset Grocery_and_Gourmet_Food --epoch 20 --emb_size 64 --lr 1e-3 --l2 0 --ssl_lambda 0.1 --tau 0.2 --num_workers 0

# 运行基线：LightGCN
python ReChorus/src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --epoch 20 --emb_size 64 --lr 1e-3 --l2 1e-4 --num_workers 0

# 运行根目录下的 LightCCF 示例（如果需要）
python LightCCF/main.py --help
```

根据框架不同，命令行参数名可能略有差异，请查看 `ReChorus/src/main.py` 的参数说明或 `--help` 输出。



## 4. 实验结果与可视化

- 项目中包含用于绘制对比图表的脚本（例如 `plot_result.py`）。运行这些脚本会读取 `log/` 目录下的实验输出并生成图片（若 `log/` 本地存在）。

示例：
```powershell
python plot_result.py
```

（根据脚本位置调整路径，例如 `python ReChorus/plot_result.py`）

