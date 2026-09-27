结论（离散 n=261 条测试路径，n_bounces≥2；与 `results/transformer_comparison.json` 同一 seed=0 划分。数字来自衍射 token 重跑，不是预设口号）。
论文对照见 [`docs/paper_notes.md`](../docs/paper_notes.md)。

### 论文主张（本玩具对齐的那几条）

**WinProp IRT**（Altair 用户指南；Hoppe et al., EPMCC 1999）：寻径是在预处理好的墙面/tile 可见性关系上做 **树搜索**；交互点被约束在这些离散元件上；交互次数很少（文档称最多约三次即可）。树上每一枝是「两个元件之间的可见性关系」，预测时先展开发射端可见的第一层，再递归检查反射条件。
→ 本玩具：`R_wall_k` / `D_corner_c` token = 树节点（反射墙或绕射角点）；合法 token 邻接 = 树边；反射的连续 t 落在该墙上，绕射点就是角点（t 记 0）。普通 AR 把「树」当成「一条句子」的局部 next-token。

**RadioDiff**（Wang et al., IEEE TCCN 2024）：无采样无线电地图构建更应是 **条件生成**，而不是 RadioUNet 式纯判别 MSE。路径损耗并不已经写在输入里，必须被生成出来。
→ 本玩具 **不重实现 RadioDiff、不生成场图**。只把同一课用在 **固定离散路径结构上的连续交互点**：联合回归/小 DDPM，而不是逐步 AR 猜 t。RadioDiff 生成 pathloss map；我们生成/refinement 的是墙上的点。

**RadioUNet**：仅作占用栅格 + Tx/Rx 通道的场景编码器上下文，不估计 RM。

WinProp IRT 也跟踪垂直棱/楔的绕射。本玩具在矩形角点上加了简化的 2D Keller/UTD **存在性**判定（轮廓/前向面、自由空间、最小弯折），token 为 `D_corner_c`。这不是完整 UTD 系数、不是 3D Keller cone。下面的数字就是这次 `R_wall_k` / `D_corner_c` 词表上的测量，没有另造指标。

### 划分

seed=0，`generate_dataset` 的 240/50/70 个场景，每个 split 再加一个反射+绕射 showcase 场景。离散表、one-shot、以及下方 Transformer 一节用的都是 test 里 **n_bounces≥2** 的路径，n=261。
连续点的原实验筛选是 **n_bounces≥1**（LoS 没有交互坐标 t），同一批 test 场景，n=515，不是 261。t 误差只平均反射跳（`bounce_mask` 丢掉绕射，因为角点坐标已经由 `D_corner_c` 钉死）；xy 误差包含绕射跳，离散结构给定时该跳应为 0。与离散同一条条路径的连续误差记在 `continuous_n_bounces_ge_2`。

### 离散交互序列（不加权小 AR + one-shot）

- 第一跳（相对「这一条」GT 路径）teacher-forcing 准确率 0.261。同一 Tx/Rx 通常有多条合法反射/绕射枝，单序列监督把树压成一句话，第一跳不是唯一 next-token。
- 不加权小 AR 的 teacher-forcing 第二交互只有 0.011，free-run 0.000，整段 exact 0.000（TF exact 0.000）。1-bounce 路径的第二 token 就是 RX，不加权交叉熵把 hop2 收成 RX，所以这条原配方几乎没学到第二跳的 `R_wall_k` / `D_corner_c`。
- **Oracle 第一交互 + AR 其余**：第二交互 0.011，后续交互 0.011，整段 exact 0.011。
- **第一跳改成模型第二候选再 AR**：第二交互 0.000，后续交互 0.000。exact 0.000，但合法率 0.590（free-run 0.621），其中 0.586 落在该场景已枚举的某条 GT 枝上：第二候选多半是换枝，不是把原序列修补回来。
- **第一跳改成随机已有墙或角点再 AR**：合法率从 0.621 到 0.157，落到场景 GT 枝的比例 0.126，exact 0.000。
- One-shot（非 AR）：hop1 0.249，hop2 0.000，exact 0.000，合法率 0.636。
- Teacher-forcing 拼出来的 token 串合法率 0.605（这不是一条解码路径，只是逐位 argmax）。

不加权小 AR，逐跳 token accuracy（hop1 / hop2 / hop3 / hop4）：

| 设定 | hop1 | hop2 | hop3 | hop4 | exact | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher-force | 0.261 | 0.011 | 0.766 | 1.000 | 0.000 | 0.605 |
| free-run | 0.261 | 0.000 | 0.000 | 0.000 | 0.000 | 0.621 |
| oracle 第一交互 | 1.000 | 0.011 | 0.023 | 0.000 | 0.011 | 0.598 |
| 第二候选第一跳 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.590 |
| 随机第一跳 | 0.000 | 0.008 | 0.050 | 0.000 | 0.000 | 0.157 |
| one-shot | 0.249 | 0.000 | 0.000 | 0.000 | 0.000 | 0.636 |
| one-shot 改第一跳 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.590 |

交互加权的小 AR 和 `SeriousPathTransformer` 用同一 n、同一第一跳协议，数字在下面的 Transformer 一节（checkpoint 未重训，只复核）。

### 连续交互点（离散 R/D 序列给定）

- 筛选 n_bounces≥1，n=515。镜像法 oracle 第一跳 xy 0.00（几何闭合）。
- 联合回归 hop1 xy 0.023，后续 0.026；AR-t free-run 后续 0.030；污染 t0 后再 AR 后续 0.028。联合回归与逐步 AR-t 的后续 xy 接近。
- 小 DDPM 后续 xy 0.080，高于联合回归的 0.026（CPU 小组件、12 步采样）。它只标出「连续几何可以联合生成」这个归纳，不声称复现 RadioDiff。
逐跳 xy（hop1 / hop2 / hop3），n_bounces≥1：
- 联合 0.023 / 0.027 / 0.014；AR-t 0.029 / 0.031 / 0.016；AR-t 污染 t0 0.061 / 0.028 / 0.016；DDPM 0.058 / 0.079 / 0.031；DDPM 冻 t0 0.061 / 0.073 / 0.031；oracle 0.00 / 0.00 / 0.00。
- 同一批 n_bounces≥2 路径（n=261，与离散 / Transformer 同一条条路径）上的误差：
  联合 hop1 xy 0.021，后续 0.026；AR-t free-run 后续 0.030；污染 t0 后再 AR 后续 0.028；DDPM 后续 0.070；oracle hop1 xy 0.00。
  逐跳 xy（hop1 / hop2 / hop3）：
  联合 0.021 / 0.027 / 0.014；AR-t 0.029 / 0.031 / 0.016；AR-t 污染 t0 0.063 / 0.028 / 0.016；DDPM 0.061 / 0.071 / 0.024；DDPM 冻 t0 0.063 / 0.074 / 0.035。

### 对研究问题的回答

**不适合把传播路径直接当成普通 autoregressive sequence 来生成。**

论文理由：WinProp IRT 的对象是可见性树 + 元件上的点，不是唯一 next-token 句子；第一交互是树的第一层分支；连续点由整段离散结构的镜像几何闭合。绕射顶点没有自由 t。RadioDiff 说明的是场图应对齐条件生成；本玩具只把该课用在固定路径结构上的反射 t。

这次测量：
1. 不加权小 AR 的第一跳准确率 0.261，第二跳 0.011。第一交互是分支，而且不加权损失学不会 `R_wall_k` / `D_corner_c` 的第二跳。
2. 第一跳换成第二候选后 exact 0.000，合法率 0.590，场景 GT 枝 0.586。
3. 第一跳换成随机已有墙/角点后合法率 0.157（free-run 0.621），exact 0.000。
4. 离散结构给定后，镜像法 oracle 的第一跳 xy 是 0.00。连续点不该再当开放 AR 序列来猜。
交互加权小 AR 与 4 层 Transformer 是否改变这个结论，看下面同一 n 上的对照，不在这里另写一套数字。
模型不必优于所有 baseline；one-shot 与 oracle-first 只是「更少 AR」和「第一跳被纠正」的对照。

<!-- TRANSFORMER_BASELINE_BEGIN -->

### Transformer 序列基线（与普通 AR 同数据、同第一跳干预）

配对测量：seed=0，测试路径 n=261（`n_bounces≥2`）。划分是 `generate_dataset` 的 240/50/70 个场景再加每个 split 一个 showcase 场景，与 `experiments.run_all` 相同。Ground truth 仍是镜像法 + 已有角点绕射规则枚举出的路径。普通 AR 和 Transformer 都不是 GT 生成器。这一节与 `results/metrics.json` 的离散表是同一次 `generate_dataset(seed=0)`、同一批 n_bounces≥2 测试路径（`R_wall_k` / `D_corner_c`）。普通 AR 原配方保持 16 epoch、不加权交叉熵，checkpoint 未重训（与本次数据上按原配方重训逐权重一致）。另有一列把同一小架构配上 Transformer 的交互 token 权重和 warmup+cosine，用来分开「损失权重」和「模型容量」。

**训练（验证集 teacher-forced next-token accuracy 选 checkpoint）。**
- 普通 AR（`ARPathTransformer`）：d=64，2 层 post-norm，无位置编码，FFN 128，dropout 0.1，AdamW lr=0.002 常数，weight decay 0.0001，batch 32，跑了 16 epoch（本配对脚本指定 16）。最佳验证 TF acc 0.488。训练 loss 2.1704 → 1.5305。
- 普通 AR + 交互权重（架构仍是 `ARPathTransformer`）：R/D 损失权重 3.0，AdamW lr=0.002，warmup+cosine，按 n_bounces≥2 的验证 TF acc 选 checkpoint。最佳 0.579 在 epoch 57，共 77 epoch，停止 `val_patience_and_train_loss_plateau`。训练 loss 2.9456 → 1.2844；结束时 train hop2 0.694。
- Transformer（`SeriousPathTransformer`）：d=128，4 层 pre-norm 因果 decoder，学习位置编码，FFN 倍数 4，dropout 0.05，参数量 874721。AdamW（矩阵权重 weight decay 0.0001；bias / LayerNorm / 位置表不衰减），交互 token（R/D）损失权重 3.0 （RX 权重为 1；不加权时 hop2 会被 1-bounce 的 RX 多数类淹没）。线性 warmup 5 epoch 后 cosine 降到 min lr，梯度裁剪 1。最佳验证 TF acc（n_bounces≥2）0.581 出现在 epoch 43，共跑 91 epoch，停止原因 `val_patience_exceeded`，训练 loss 平台标记 False。训练 loss 2.8643 → 1.1670；结束时 train TF acc 0.722，该 epoch 的 val TF acc 0.538。曲线：`results/figures/transformer_train_curves.png`。

协议与 `score_sequence_model` 一致（也就是 `eval_discrete` 里 AR 的那一支）：teacher-forcing；greedy free-run；oracle 第一交互再自由生成；第一交互换成模型第二候选再自由生成；第一交互换成随机的、已存在的墙或角点 token（不等于标注第一跳）再自由生成。随机干预的 RNG 是 `np.random.default_rng(seed+7)`，两个模型各用一次相同的种子，因此非法第一跳相同。exact = 整段 token 对上这一条标注路径。valid = 镜像法/绕射重建成功。any GT branch = 预测交互序列出现在该场景已枚举路径里。

| 模型 | TF hop1 | TF hop2 | TF exact | free hop2 | free exact | free valid | oracle hop2 | oracle exact | 2nd hop2 | 2nd exact | 2nd valid | 2nd 落在场景某条 GT 枝 | rand valid | rand exact | 第一跳 top-2 含 GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 AR（原配方） | 0.261 | 0.011 | 0.000 | 0.000 | 0.000 | 0.621 | 0.011 | 0.011 | 0.000 | 0.000 | 0.590 | 0.586 | 0.157 | 0.000 | 0.429 |
| 普通 AR + 交互权重 | 0.310 | 0.406 | 0.126 | 0.218 | 0.126 | 0.632 | 0.406 | 0.356 | 0.103 | 0.000 | 0.590 | 0.483 | 0.180 | 0.000 | 0.502 |
| Transformer | 0.287 | 0.383 | 0.126 | 0.257 | 0.126 | 0.655 | 0.383 | 0.330 | 0.146 | 0.000 | 0.609 | 0.460 | 0.169 | 0.000 | 0.475 |

Teacher-forcing 与 free-run 的逐跳 token accuracy（hop1 = 第一交互，之后常含第二交互与 RX）：

| 模型 | hop1 | hop2 | hop3 | hop4 |
| --- | ---: | ---: | ---: | ---: |
| AR teacher-force | 0.261 | 0.011 | 0.766 | 1.000 |
| AR free-run | 0.261 | 0.000 | 0.000 | 0.000 |
| AR+weight teacher-force | 0.310 | 0.406 | 0.751 | 0.844 |
| AR+weight free-run | 0.310 | 0.218 | 0.521 | 0.188 |
| Transformer teacher-force | 0.287 | 0.383 | 0.701 | 1.000 |
| Transformer free-run | 0.287 | 0.257 | 0.617 | 0.188 |
| AR wrong-1st (2nd)  | 0.000 | 0.000 | 0.000 | 0.000 |
| AR+weight wrong-1st (2nd) | 0.000 | 0.103 | 0.452 | 0.109 |
| Transformer wrong-1st (2nd) | 0.000 | 0.146 | 0.525 | 0.203 |
| AR wrong-1st (random) | 0.000 | 0.008 | 0.050 | 0.000 |
| AR+weight wrong-1st (random) | 0.000 | 0.103 | 0.398 | 0.062 |
| Transformer wrong-1st (random) | 0.000 | 0.100 | 0.375 | 0.203 |

图：`results/figures/transformer_vs_ar_hop_accuracy.png`。原始数字：`results/transformer_comparison.json`。

**这一对照说明什么。** 把交互 token 的损失权重调到和 Transformer 一样之后，原来的小 AR 在 teacher-forced 第二跳上就能到 0.406，Transformer 是 0.383。多出来的层数和宽度并没有单独把第二条跳学出来：原配方 AR 的 hop2 接近 0，是因为不加权交叉熵被 1-bounce 的 RX 淹没。两边在改错第一跳之后 exact 都掉到约 0，随机离开树的第一跳也都会把几何合法率打下去。所以更强的 Transformer 没有改变结论：普通 next-token free-run 仍然不是这条可见性树的正确对象。
具体差：Transformer free-run exact − 原配方 AR free-run exact = 0.126；同权重小 AR free-run exact 0.126，其 TF hop2 0.406 → free hop2 0.218；Transformer TF hop2 0.383 → free hop2 0.257。第二候选后的后续墙准确率 原配方 AR 0.000 / Transformer 0.148；随机第一跳后的合法率 原配方 AR 0.157 （free-run 0.621）/ Transformer 0.169 （free-run 0.655）。
误差是否沿跳累积，看上表 TF hop2 与 free hop2 的差，不另定义指标。

### Transformer sequence baseline (same data, same first-hop protocol)

Paired measurement, seed=0, n=261 test paths with n_bounces≥2. Splits match `experiments.run_all` (240/50/70 scenes plus one showcase scene per split, seed 0) and the discrete table in `results/metrics.json` (tokens `R_wall_k` / `D_corner_c`). Ground truth is still the image-method enumerator plus the repo's corner-diffraction rules. Neither model generates that ground truth. Epoch budgets differ on purpose: the ordinary AR keeps this repo's 16-epoch constant-lr recipe; the Transformer is trained until validation teacher-forced accuracy stops improving and the training loss flattens, or until the epoch cap.

Ordinary AR: best validation teacher-forced accuracy 0.488 after 16 epochs at the published small-model recipe (d=64, 2 layers, constant AdamW lr=2e-3). Transformer: 874721 parameters, interaction-token loss weight 3.0, best validation teacher-forced accuracy on n_bounces≥2 0.581 at epoch 43 of 91 (val_patience_exceeded; train-loss plateau flag False). Train loss 2.8643 → 1.1670.

With the same interaction-token loss weight, the original small AR reaches teacher-forced hop-2 accuracy 0.406; the Transformer reaches 0.383. The extra depth is not what fixes hop 2. The shipped unweighted AR sits near zero there because one-bounce paths (second token = RX) dominate the loss. After a forced wrong first hop, exact match to the labeled path is about zero for both the weighted AR and the Transformer, and a random off-tree first hop drops geometric validity for both. A larger Transformer does not change the conclusion: ordinary next-token free-run is still the wrong object for this visibility tree.
Free-run exact: AR 0.000, Transformer 0.126 (delta 0.126). After the model's own second-best first hop, later-wall accuracy is AR 0.000, Transformer 0.148; exact match is AR 0.000, Transformer 0.000; geometric validity stays AR 0.590, Transformer 0.609, of which a scene-enumerated branch covers AR 0.586 and Transformer 0.460. After a random existing wall/corner first hop, validity is AR 0.157 and Transformer 0.169 (exact 0.000 / 0.000). First-hop top-2 contains the labeled token for AR 0.429 and Transformer 0.475 of paths.
Hop-wise teacher-forcing versus free-run is the table above and `results/figures/transformer_vs_ar_hop_accuracy.png`. No metric in this section was filled in by hand.

<!-- TRANSFORMER_BASELINE_END -->
