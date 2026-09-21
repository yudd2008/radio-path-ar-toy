# 传播路径是否适合当作普通 Autoregressive Sequence？

2D 无线电传播路径玩具实验。核心问题不是再做一个工业级射线追踪器，而是：**把 IRT 风格的传播路径直接当成普通自回归序列来生成，是否合适？**

最低要求：用实验证据回答该问题。模型不必优于所有 baseline。

**本次 seed=0 玩具 run 的回答：不适合。** 第一交互是可见性树上的分支选择；改错第一跳后，相对原 GT 的后续墙会垮掉。若错误第一跳仍落在树上，模型会走出**另一条**合法路径（换枝，不是 AR 纠错）；若第一跳是随机墙（离开树），几何合法率从 0.96 掉到 0.15。连续点在离散墙序列给定后由镜像法闭合（oracle 误差 0）。

---

## 1. 问题

Altair WinProp **Intelligent Ray Tracing (IRT)** 把路径看成：

1. 在离散墙面/tile 上的 **可见性树搜索**（预处理可见性，预测时沿树展开，交互次数通常很少，文档建议最多约 3 次）；
2. 每个离散元件上的 **连续交互点**。

见 [WinProp IRT 用户指南](https://2021.help.altair.com/2021/winprop/topics/winprop/user_guide/propagation_methods/propgation_models_urban_ray_models_irt.htm)。

普通 AR next-token（Transformer/LSTM）假设：下一步主要由局部前缀决定。镜面路径的第一跳其实被 **整条墙序列的镜像几何** 约束；后续跳也不是「再猜一个相邻 token」，而是树上下一条可见边。

## 2. 领域启发（有意做小，不复现 SOTA）

| 来源 | 本玩具里怎么用 |
| --- | --- |
| RadioUNet | 仅作 **场景上下文**：占用栅格 + Tx/Rx 热斑 → 小 CNN。不做完整无线电地图。 |
| [RadioDiff](https://github.com/UNIC-Lab/RadioDiff)（IEEE TCCN） | **借生成式想法，不重实现**。论文把无线电地图从判别式 CNN 改成条件扩散。这里把同样的思路用在 **连续交互点**：在离散墙序列条件下联合预测/去噪 \(t\in(0,1)^K\)。 |
| ILO + PGD（扩散逆问题，可选） | 玩具版：采样时冻住被污染的 \(t_0\)（replacement / inpaint）。 |
| 3DGS（可选类比） | 离散元件 + 连续属性；本玩具对应「墙 ID + 墙上的 \(t\)」。 |
| WinProp IRT | 路径 = 离散墙序列 + 连续点；精确几何用 **镜像法**，上限 2–3 次反射。 |

## 3. 方法

```
env/          轴对齐矩形障碍、稳定 wall ID、占用栅格
pathfind/     镜像法：LoS / 1-bounce / 少次多跳镜面路径
data/         JSONL+NPZ（几何、token、连续点、合法性）
models/       小 AR Transformer、one-shot 序列头、联合回归 / AR-t / 小 DDPM
experiments/  训练 + 强制改第一跳的误差累积实验
```

**离散表示：** `TX → wall_3 → wall_7 → RX`（wall ID 在单个场景内稳定；模型看墙的几何，而不是全局可迁移的 token 语义）。

**连续表示：** 每个反射点在所属墙上的 \(t\in(0,1)\)，以及二维坐标。

**对照：** teacher-forcing vs free-run；强制错误第一交互再 AR；one-shot 整段；oracle 第一 token + AR 其余；连续头的联合回归 / AR-t / 小 DDPM vs 镜像法 oracle。

## 4. 怎么跑

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 tests/test_image_method.py
PYTHONPATH=. python3 tests/test_models_shapes.py
PYTHONPATH=. python3 -m experiments.run_all --seed 0 --out results
```

默认 CPU：生成小数据集（约 240/50/70 个场景）→ 训练很小的模型 → 写出 `results/metrics.json`、`results/findings.md` 和 `results/figures/`。整段 demo 大约一分钟量级。

样本 schema 见生成后的 `data/generated/schema_example.json`。

## 5. 关键发现（seed=0，n=204 条 n_bounces≥2 的测试路径）

完整文字结论见 [`results/findings.md`](results/findings.md)，原始数字见 [`results/metrics.json`](results/metrics.json)。

镜像法 GT（同一峡谷场景里 LoS、两侧 1-bounce、2-bounce ping-pong 同时存在——这就是 IRT 树，不是单一序列）：

![GT image-method paths](results/figures/gt_paths_example.png)

强制改第一跳之后，模型走出**另一条**合法镜面路径（换枝），而不是沿着原 GT 续写：

![GT vs AR wrong first interaction](results/figures/example_0_sid20000.png)

| 设定 | 第二墙 vs 这条 GT | 整段 exact | 几何合法 | 落在该场景某条已枚举 GT 分支 |
| --- | ---: | ---: | ---: | ---: |
| AR free-run | 0.451 | 0.230 | 0.956 | （free-run 常是某条合法枝） |
| Oracle 第一墙 + AR 其余 | 0.858 | 0.461 | 0.961 | — |
| 第一跳 = 模型第二候选，再 AR | **0.020** | **0.000** | 0.956 | **0.941** |
| 第一跳 = 随机墙，再 AR | 0.358 | 0.000 | **0.147** | 0.147 |
| One-shot（非 AR） | 0.010 | 0.005 | 0.985 | 短合法枝，对不上这条多跳 GT |

![hop accuracy](results/figures/discrete_hop_accuracy.png)

![exact vs valid](results/figures/discrete_exact_valid.png)

连续点（**离散墙序列给定**）：镜像法 oracle 误差为 0。联合回归 hop1 \(\|\Delta xy\|\approx 0.049\)，后续 0.035；把 \(t_0\) 污染后再做 AR-t，后续 0.053。小 DDPM 后续 0.174，**没有**超过联合回归——这是 CPU 小组件的真实结果，不编造 SOTA。

![continuous error](results/figures/continuous_error_growth.png)

### 研究问题的回答

**不适合把传播路径直接当成普通 autoregressive sequence 来生成。**

理由与 IRT 对齐：

1. 第一交互是树上的分支选择，不是局部 next-token；相对「这一条」GT，改错第一跳后后续墙准确率从 0.86 掉到 0.02。
2. 第二候选往往是另一条合法 IRT 枝（合法率仍 ~0.96，且 94% 能在场景枚举集里对上），这是换枝，不是 AR 把原序列修补回来。
3. 随机墙离开可见性树后，合法率崩溃到 0.15。
4. 连续点在离散结构给定后由几何闭合；应联合生成/回归（RadioDiff 式想法），而不是逐步 AR。

Teacher-forcing 第一跳只有 0.46，也说明多路径设定下「一条普通 AR 序列」这个监督本身就是错的。

## 6. 许可

MIT。这是研究玩具，不是 WinProp / RadioDiff 的再实现，也不能当现场规划工具用。
