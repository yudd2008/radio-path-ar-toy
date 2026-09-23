结论（n=204 条测试路径，n_bounces≥2；数字来自本次 toy run，seed=0，不是预设口号）。  
论文对照见 [`docs/paper_notes.md`](../docs/paper_notes.md)。

### 论文主张（本玩具对齐的那几条）

**WinProp IRT**（Altair 用户指南；Hoppe et al., EPMCC 1999）：寻径是在预处理好的墙面/tile 可见性关系上做 **树搜索**；交互点被约束在这些离散元件上；交互次数很少（文档称最多约三次即可）。树上每一枝是「两个元件之间的可见性关系」，预测时先展开发射端可见的第一层，再递归检查反射条件。  
→ 本玩具：`R_wall_k` / `D_corner_c` token = 树节点（反射墙或绕射角点）；合法 token 邻接 = 树边；\(t\in(0,1)\) = 该墙上的连续点（绕射点就是角点）。普通 AR 把「树」当成「一条句子」的局部 next-token。

**RadioDiff**（Wang et al., IEEE TCCN 2024）：无采样无线电地图构建更应是 **条件生成**，而不是 RadioUNet 式纯判别 MSE。路径损耗并不已经写在输入里，必须被生成出来。  
→ 本玩具 **不重实现 RadioDiff、不生成场图**。只把同一课用在 **固定离散路径结构上的连续交互点**：联合回归/小 DDPM，而不是逐步 AR 猜 \(t_i\mid t_{<i}\)。RadioDiff 生成的是 pathloss map；我们生成/ refinement 的是墙上的点。

**RadioUNet**：仅作占用栅格 + Tx/Rx 通道的场景编码器上下文，不估计 RM。

### 离散交互序列（已有测量）
- 第一跳（相对「这一条」GT 路径）准确率只有 0.456。同一 Tx/Rx 通常有多条合法 IRT 分支，普通 AR 被训练成对准单条标注路径，第一跳本身就不是唯一 next-token。这与 IRT「第一层 = 发射端可见元件的分支选择」一致。
- Teacher-forcing 第二墙 0.858；free-run 第二墙掉到 0.451（给定自己的第一跳后续开始偏）。exact 0.230。
- **Oracle 第一墙 + AR 其余**：第二墙 0.858，后续墙 0.667，整段 exact 0.461。第一层被纠正后，其余更像沿着那条树边走。
- **把第一跳改成模型第二候选再 AR**：第二墙 0.020，后续墙 0.015，exact 0.000。
  几何合法率仍有 0.956（与 free-run 0.956 接近），且 0.941 落在该场景已枚举的某条 GT 分支上——说明第二候选往往是树上的另一枝，而不是「同一条序列的局部噪声」。这正是 IRT 树的换枝，不是语言模型式的 AR 纠错。
- **把第一跳改成随机墙再 AR**：合法率从 0.956 掉到 0.147，落到场景 GT 分支的比例 0.147，exact 0.000。离开可见性树后，后续 next-token 几乎拼不出镜面路径：镜像法要求 **整段** 墙序列与遮挡/反射定律全局一致，局部续写不够。
- One-shot 非 AR：exact 0.005，后续墙几乎对不上这条多跳 GT，但合法率 0.985（常退化成短的 1-bounce/LoS 合法枝）。

### 连续交互点（离散墙序列给定，已有测量）
- 镜像法 oracle 第一跳点误差 0.00e+00（几何闭合）。离散结构已知时，连续点由全局 unfold 决定，不该再当开放的 AR 序列来猜——这是 IRT「点落在给定 tile 上」+ 镜面闭合。
- 联合回归 hop1 xy 0.049，后续 0.035；AR-t free-run 后续 0.049；污染 t0 后再 AR 后续 0.053。联合头（RadioDiff 所说的「连续几何用生成/联合而不是逐步判别」）比逐步 AR-t 更稳。
- 本玩具里的小 DDPM 后续误差 0.174，没有超过联合回归（CPU 小组件、步数少）。它只用来表明归纳偏置，不声称复现 RadioDiff，也不编造 SOTA。

### 对研究问题的回答

**不适合把传播路径直接当成普通 autoregressive sequence 来生成。**

论文理由：WinProp IRT 的对象是可见性树 + 元件上的点，不是唯一 next-token 句子；第一交互是树的第一层分支；连续点由整段离散结构的镜像几何闭合。RadioDiff 进一步说明：连续无线电几何应对齐条件生成，而不是纯判别逐步回归——但那是场图；我们只把该课用在固定路径结构上的 \(t\)。

玩具证据（上表数字）：
1. 第一交互是分支选择，不是局部 token；改错第一跳后，相对原 GT 的后续墙准确率崩掉（0.858→0.020）；
2. 若错误第一跳仍落在树上（模型第二候选），可以走出另一条合法路径（合法率 0.956，0.941 为场景 GT 枝），但这是换枝，不是 AR 纠错；
3. 若第一跳是随机墙（离开树），几何合法率崩溃（0.956→0.147）——缺少全局镜面/镜像法一致性；
4. 连续点在离散结构给定后由镜像法决定（oracle 误差 0），联合回归/生成比逐步 AR 更合适。

模型不必优于所有 baseline；one-shot 与 oracle-first 只是「更少 AR」和「第一跳被纠正」的对照。

### 绕射（几何玩具，不是另造的 AR 指标）

WinProp IRT 也会在垂直棱/楔上绕射。本玩具在矩形角点上加了简化的 2D 存在性判定（轮廓面、自由空间、最小弯折），token 为 `D_corner_c`，与 `R_wall_k` 分开。**不是** 完整 UTD / 3D Keller cone。展示图见 `results/figures/reflect_diffract_showcase.png`。上表 AR 数字仍来自 seed=0 实验测量，没有为绕射编造新指标。

<!-- TRANSFORMER_BASELINE_BEGIN -->

### Transformer 序列基线（与普通 AR 同数据、同第一跳干预）

配对测量：seed=0，测试路径 n=261（`n_bounces≥2`）。划分是 `generate_dataset` 的 240/50/70 个场景再加每个 split 一个 showcase 场景，与 `experiments.run_all` 相同。Ground truth 仍是镜像法 + 已有角点绕射规则枚举出的路径。普通 AR 和 Transformer 都不是 GT 生成器。本环境重跑 `generate_dataset(seed=0)` 得到的 n 与仓库里先前 commit 的 n=204 表不是同一次抽样，旧表保留在上面，这里不覆盖 `metrics.json`。普通 AR 原配方保持 16 epoch、不加权交叉熵。另有一列把同一小架构配上 Transformer 的交互 token 权重和 warmup+cosine，用来分开「损失权重」和「模型容量」。

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

Paired measurement, seed=0, n=261 test paths with n_bounces≥2. Splits match `experiments.run_all` (240/50/70 scenes plus one showcase scene per split, seed 0). This draw is not the previously committed n=204 table in `results/metrics.json`; that file is left unchanged. Ground truth is still the image-method enumerator plus the repo's corner-diffraction rules. Neither model generates that ground truth. Epoch budgets differ on purpose: the ordinary AR keeps this repo's 16-epoch constant-lr recipe; the Transformer is trained until validation teacher-forced accuracy stops improving and the training loss flattens, or until the epoch cap.

Ordinary AR: best validation teacher-forced accuracy 0.488 after 16 epochs at the published small-model recipe (d=64, 2 layers, constant AdamW lr=2e-3). Transformer: 874721 parameters, interaction-token loss weight 3.0, best validation teacher-forced accuracy on n_bounces≥2 0.581 at epoch 43 of 91 (val_patience_exceeded; train-loss plateau flag False). Train loss 2.8643 → 1.1670.

With the same interaction-token loss weight, the original small AR reaches teacher-forced hop-2 accuracy 0.406; the Transformer reaches 0.383. The extra depth is not what fixes hop 2. The shipped unweighted AR sits near zero there because one-bounce paths (second token = RX) dominate the loss. After a forced wrong first hop, exact match to the labeled path is about zero for both the weighted AR and the Transformer, and a random off-tree first hop drops geometric validity for both. A larger Transformer does not change the conclusion: ordinary next-token free-run is still the wrong object for this visibility tree.
Free-run exact: AR 0.000, Transformer 0.126 (delta 0.126). After the model's own second-best first hop, later-wall accuracy is AR 0.000, Transformer 0.148; exact match is AR 0.000, Transformer 0.000; geometric validity stays AR 0.590, Transformer 0.609, of which a scene-enumerated branch covers AR 0.586 and Transformer 0.460. After a random existing wall/corner first hop, validity is AR 0.157 and Transformer 0.169 (exact 0.000 / 0.000). First-hop top-2 contains the labeled token for AR 0.429 and Transformer 0.475 of paths.
Hop-wise teacher-forcing versus free-run is the table above and `results/figures/transformer_vs_ar_hop_accuracy.png`. No metric in this section was filled in by hand.

<!-- TRANSFORMER_BASELINE_END -->
