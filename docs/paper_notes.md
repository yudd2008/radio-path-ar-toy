# 论文 → 本玩具设计对照（可审计）

本文档只做「文献里写了什么 → 我们在代码里做了什么」的对照，便于核对研究主张。  
**不重实现** RadioDiff / WinProp；数字以 `results/findings.md` 与 `results/metrics.json`（seed=0）为准，此处不另造指标。

---

## 1. 研究问题

**传播路径是否适合直接作为普通 autoregressive sequence 生成？**

「普通 AR」在这里指：把一条路径写成 token 序列，用 \(p(\text{next}\mid\text{prefix}, \text{scene}, Tx, Rx)\) 局部 next-token 生成，并默认存在一条应对齐的目标序列。

---

## 2. WinProp IRT → 离散结构

**文献**

- Altair WinProp 用户指南 *Intelligent Ray Tracing (IRT)*：  
  https://2021.help.altair.com/2021/winprop/topics/winprop/user_guide/propagation_methods/propgation_models_urban_ray_models_irt.htm
- 方法来源：R. Hoppe, G. Wölfle, F. M. Landstorfer, “Fast 3D Ray Tracing for the Planning of Microcells by Intelligent Preprocessing of the Database,” EPMCC, 1999.

**文献主张（转述，非逐字抄录全文）**

1. **墙/边先被离散成元件**：建筑物墙面分成 tile，棱边分成水平/垂直 segment。
2. **可见性关系可预处理，且与基站位置无关**：tile/segment 之间的可见性事先算好存成文件；预处理时元件用中心点代表，把路径搜索简化。
3. **预测 = 在树上搜索**：预处理结果是一棵树，**每一枝表示两个元件之间的一条可见性关系**。预测时先找基站能看见的第一层元件，再递归展开可见元件，并检查反射/绕射条件。
4. **交互点被约束在这些离散元件上**：可能的交互位置由 tile/segment 决定，而不是自由空间里的任意点。
5. **交互次数很少就停**：射线在到达接收点或达到最大交互次数时停止。文档称 **最多约三次交互** 即可得到很好的结果。
6. 第一层（发射端可见的元件）是在预测阶段算的；其余层大多来自预处理。这强化了「第一交互是特殊的根部分支选择」。

**本玩具里的对应**

| IRT 概念 | 本仓库 |
| --- | --- |
| 建筑物墙面 / tile | 轴对齐矩形的 **命名边**（稳定 `wall_id`，如 `r0_right`） |
| 树上的节点 | 离散 token：`TX`、`wall_k`、`RX` |
| 树上的边（可见性关系） | 合法的相邻 token 对，须通过镜像法 + 遮挡检验 |
| 沿树的一条根到叶路径 | 一条样本：`TX → wall_i → wall_j → RX` |
| 元件上的交互点 | 连续 \(t\in(0,1)\)（点被约束在该墙上）+ 二维坐标 |
| 最大交互次数 | `MAX_BOUNCES = 3` |
| 预处理可见性 + 预测时搜树 | `pathfind/image_method.py` 枚举短墙序列并做几何闭合（玩具版精确搜索，不是工业预处理文件） |

**为何普通 next-token AR 与此不匹配**

- IRT 的「下一步」是 **可见性树上下一条边**，并且整条枝还要满足反射条件；不是语言模型式的 \(p(\text{token}_k\mid \text{token}_{<k})\) 去拟合 **唯一** 一条标注序列。
- 同一 Tx/Rx 对应 **多条** 合法树路径（LoS、左右 1-bounce、2-bounce ping-pong 可同时存在）。普通 AR 的单序列监督把「树」压成「一条句子」。
- **第一交互对应树的第一层**（从发射端出发的可见元件）。选错第一层等于换根分支；后续 token 即使局部像「下一堵墙」，也不是原 GT 那条枝。
- **连续点服从全局镜像法**：给定整段墙序列，交互点由 unfold 一次决定（Snell / 镜像几何），不是由 \(t_{i-1}\) 局部推出 \(t_i\)。逐步 AR 连续头缺少这一全局闭合。

---

## 3. RadioDiff → 连续点（只借归纳，不复现）

**文献**

- Xiucheng Wang, Keda Tao, Nan Cheng, et al., “RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction,” *IEEE Transactions on Cognitive Communications and Networking*, 2024.  
  https://doi.org/10.1109/tccn.2024.3504489  
  arXiv: https://arxiv.org/abs/2408.08593  
  代码：https://github.com/UNIC-Lab/RadioDiff

**文献主张（摘要与引言的核心句）**

- 现有无采样 NN 方法（以 RadioUNet 为代表）把无线电地图当成 **判别式** 监督（MSE 回归），与问题的 **生成式** 属性错位：路径损耗的数值与位置并不已经写在输入环境里，网络必须把它们 **生成** 出来；连续场也很难靠有限超平面分类得到。
- 因此把 **无采样 RM 构建建模为条件生成问题**，用去噪扩散（RadioDiff）在基站位置与环境特征条件下生成 pathloss 地图。

**本玩具里的对应（必须对比清楚）**

| RadioDiff | 本玩具 |
| --- | --- |
| 生成对象：整张 **pathloss / radio map** | 生成/回归对象：固定离散墙序列上的 **连续交互点** \(t\) |
| 条件：环境占用 + 基站位置等 | 条件：场景编码 + Tx/Rx + **已经给定的离散 wall 序列** |
| 方法：完整 attention U-Net + 解耦扩散等 | **不重实现**；只用「连续几何用条件生成/联合预测，而不是纯逐步判别」这一课 |
| 输出在二维场上 | 输出在 1D 墙参数上（点必须落在 IRT 元件上） |

所以：RadioDiff 的课是 **归纳偏置**（连续无线电几何更像条件生成），不是把他们的 RM 扩散模型搬过来。我们的小 DDPM 只是这一课的玩具头；seed=0 上它 **没有** 超过联合回归，这一点保持原测量，不改写成 SOTA。

---

## 4. RadioUNet → 仅场景上下文

**文献**

- Ron Levie, Çağkan Yapar, Gitta Kutyniok, Giuseppe Caire, “RadioUNet: Fast Radio Map Estimation with Convolutional Neural Networks,” IEEE TWC / arXiv:1911.09002.  
  RadioDiff 将其作为典型的 **U-Net 图像到图像、MSE 判别式** RM 基线。

**本玩具里的对应**

- 只用占用栅格 + Tx/Rx 高斯斑作为 CNN 场景编码器输入（`env/raster.py`，RadioUNet 风格的通道）。
- **不** 估计无线电地图，**不** 当路径生成器，**不** 与 RadioUNet 比 NMSE。

---

## 5. 实验如何检验上述主张（不新增指标）

已有 seed=0 测量（见 `results/findings.md`）分别对应：

| 主张 | 用哪项已有数字 |
| --- | --- |
| 第一跳是树的分支选择，不是唯一 next-token | 第一跳 vs 单条 GT 仅 0.456；同一场景多路径图 |
| 改错第一跳 → 相对原 GT 的后续墙崩溃 | 第二候选干预：第二墙 0.858→0.020，exact 0.461→0.000 |
| 落在树上的错第一跳 = 换枝而非 AR 纠错 | 第二候选后合法率仍 0.956，且 0.941 是场景里另一条已枚举 GT 枝 |
| 离开树（随机墙）→ 几何崩溃 | 合法率 0.956→0.147 |
| 连续点由离散结构全局闭合 | 镜像法 oracle \(\|\Delta xy\|=0\)；联合回归优于污染 \(t_0\) 后的 AR-t |

---

## 6. 明确不做的事

- 不实现 WinProp 的工业预处理数据库、绕射、3D、tile 中心近似。
- 不实现 RadioDiff 的 attention U-Net、自适应 FFT、解耦扩散、动态障碍 RM。
- 不宣称本玩具模型优于所有 baseline 或达到工业精度。
