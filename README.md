# 传播路径是否适合当作普通 Autoregressive Sequence？

2D 无线电传播路径玩具实验。核心问题不是再做一个工业级射线追踪器，而是：**把 IRT 风格的传播路径直接当成普通自回归序列来生成，是否合适？**

最低要求：用实验证据回答该问题。模型不必优于所有 baseline。

---

## 1. 问题

Altair WinProp **Intelligent Ray Tracing (IRT)** 把路径看成：

1. 在离散墙面/tile 上的 **可见性树搜索**（预处理可见性，预测时沿树展开，交互次数通常很少，文档建议最多约 3 次）；
2. 每个离散元件上的 **连续交互点**。

见 [WinProp IRT 用户指南](https://2021.help.altair.com/2021/winprop/topics/winprop/user_guide/propagation_methods/propgation_models_urban_ray_models_irt.htm)。

普通 AR next-token（Transformer/LSTM）假设：下一步主要由局部前缀决定。而镜面路径的第一跳其实被 **整条墙序列的镜像几何** 约束；后续跳也不是「再猜一个相邻 token」，而是树上下一条可见边，并且连续点由全局 Snell/镜像法闭合。

本仓库用可在 CPU 上跑完的 2D 玩具，把这个归纳偏置问题测清楚。

## 2. 领域启发（有意做小，不复现 SOTA）

| 来源 | 本玩具里怎么用 |
| --- | --- |
| [RadioUNet](https://github.com/BenediktLeible/RadioUNet) | 仅作 **场景上下文**：把障碍物占用栅格 + Tx/Rx 热斑送进小 CNN。不做完整无线电地图回归。 |
| [RadioDiff](https://github.com/UNIC-Lab/RadioDiff)（IEEE TCCN） | **借生成式想法，不重实现 RadioDiff**。论文把无线电地图从判别式 CNN 改成条件扩散。这里把同样的思路用在 **连续交互点**：在离散墙序列条件下，对墙参数 \(t\in(0,1)^K\) 做小 DDPM，而不是把连续几何也写成逐步 AR。 |
| ILO + PGD（扩散逆问题，可选） | 玩具版：采样时 **冻住被污染的 \(t_0\)**（replacement / inpaint），看后续点能否在错误第一交互下自洽。 |
| 3DGS（可选类比） | 离散元件 + 连续属性。本玩具的「墙 ID + 墙上的 \(t\)」是同一分层，不做 splatting。 |
| WinProp IRT | 路径 = 离散墙序列 + 连续点；精确几何用 **镜像法**，上限 2–3 次反射。 |

## 3. 方法

```
env/          轴对齐矩形障碍、稳定 wall ID、占用栅格
pathfind/     镜像法：LoS / 1-bounce / 少次多跳镜面路径
data/         JSONL+NPZ 样本（几何、token、连续点、合法性）
models/       小 AR Transformer、one-shot 序列头、联合回归 / AR-t / 小 DDPM
experiments/  训练 + 强制改第一跳的误差累积实验
```

**离散表示：** `TX → wall_3 → wall_7 → RX`（wall ID 在单个场景内稳定，模型看的是墙的几何特征，而不是全局可迁移的 token 语义）。

**连续表示：** 每个反射点在所属墙上的一维参数 \(t\in(0,1)\)，以及二维坐标。

**对照：**

- Teacher-forcing vs free-run AR
- **干预：** 强制错误的第一交互，再继续 AR，观察后续墙与连续点如何变差
- 更少 AR：one-shot 整段序列；oracle 第一 token + AR 其余
- 连续：联合回归 / 联合 DDPM vs AR 逐步预测 \(t\)；镜像法 oracle（离散结构已知时几何应闭合）

## 4. 怎么跑

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 tests/test_image_method.py
PYTHONPATH=. python3 tests/test_models_shapes.py
PYTHONPATH=. python3 -m experiments.run_all --seed 0 --out results
```

默认在 CPU 上：生成小数据集 → 训练很小的模型 → 写出 `results/metrics.json`、`results/findings.md` 和 `results/figures/`。

只生成数据：

```bash
PYTHONPATH=. python3 -c "from data.generate import generate_dataset; generate_dataset('data/generated', seed=0)"
```

样本 schema 见生成后的 `data/generated/schema_example.json`。一条 JSONL 记录包含场景矩形、Tx/Rx、token、墙上 \(t\)、折线点和 `valid`。

## 5. 关键发现

跑完 `experiments.run_all` 后，数字以 `results/findings.md` 与 `results/metrics.json` 为准（本玩具实测，不编造指标）。定性预期（是否成立必须看那次 run 的图）：

- Teacher-forcing 可以在局部 hop 上显得「还行」，free-run 与 **改第一跳再 AR** 会让后续墙序列和几何合法性一起垮。
- 这与 IRT 的树/搜索本质一致：第一交互选错等于走错了可见性树的根分支，后面的 next-token 没有局部可恢复性。
- 离散墙序列一旦给定，连续点几乎由镜像法决定（oracle 误差应接近 0）。连续几何更适合 **联合生成/回归**（RadioDiff 式），而不是普通 AR。

图（生成后）：

- `results/figures/gt_paths_example.png` — 镜像法 GT 路径
- `results/figures/example_*.png` — GT vs AR free-run vs 错误第一跳
- `results/figures/discrete_hop_accuracy.png` — hop-wise token 准确率
- `results/figures/discrete_exact_valid.png` — 精确匹配 vs 几何合法
- `results/figures/continuous_error_growth.png` — 连续点误差随 hop 增长

## 6. 许可

MIT。这是研究玩具，不是 WinProp / RadioDiff 的再实现，也不能当现场规划工具用。
