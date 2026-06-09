# KataGo 棋力评估研究文档

> 本文档记录围绕 KataGo trunk 特征提取、HumanSL 分析、可解释性与棋力评分模型的完整技术框架。  
> 基于项目讨论整理，供后续开发查阅。

---

## 目录

1. [项目目标](#一项目目标)
2. [KataGo 神经网络架构](#二katago-神经网络架构)
3. [Trunk 特征：形状、含义与可视化](#三trunk-特征形状含义与可视化)
4. [Pick 特征](#四pick-特征)
5. [HumanSL 模型](#五humansl-模型)
6. [表征理论：正确理解 Trunk 的维度](#六表征理论正确理解-trunk-的维度)
7. [特征与人类认知的对齐](#七特征与人类认知的对齐)
8. [数据结构设计原则（三层分离）](#八数据结构设计原则三层分离)
9. [评估模型架构设计](#九评估模型架构设计)
10. [评分体系设计哲学](#十评分体系设计哲学)
11. [实践路径（修订版）](#十一实践路径修订版)
12. [当前 Fork 状态（已修复）](#十二当前-fork-状态已修复)
13. [数据格式参考（KAB2 v2）](#十三数据格式参考kab2-v2)
14. [HumanSL 集成策略与开销分析](#十四humansl-集成策略与开销分析)
15. [客观评分公式：meanLogPrior](#十五客观评分公式meanlogprior)

---

## 一、项目目标

给定一局 SGF，利用 KataGo 分析结果（不依赖外部数据库），输出：

- **双方棋手的综合评分**（可映射到段位区间）
- **多维度分项评价**（开局/中盘/收官 × 准确率/风格/稳定性等）
- **参考基准**：以 KataGo HumanSL 段位模型为锚点，使评分对人类棋手有意义

灵感来源：[Animiral/go-strength-model](https://github.com/Animiral/go-strength-model) — 该项目提取 KataGo trunk per move，训练模型预测 Glicko-2 段位（基于 10s 对局数据）。

---

## 二、KataGo 神经网络架构

### 2.1 整体数据流

```
输入特征
  22 通道二值平面（落子、气、历史）
  + 全局特征向量（规则、贴目、手数等）
        ↓
  初始卷积层 (initial conv)
        ↓
┌─────────────────────────────────────┐
│            T R U N K                │  ← 本项目关注点
│  ┌─ ResidualBlock × N               │
│  ├─ GlobalPoolingResidualBlock × M  │
│  └─ Transformer Block × K (新模型)  │
│        ↓                            │
│  TrunkTip 归一化 (BN 或 RMSNorm)   │
└─────────────────────────────────────┘
        ↓                   ↓
  [Policy Head]       [Value Head]
  落子概率分布         胜率 / 得分估计
  (19×19 + 1)         whiteWinProb
                      whiteScoreMean
```

### 2.2 主要模型规格

| 模型 | 用途 | Trunk 通道数 | 参数量 |
|------|------|------------|--------|
| `kata1-b18c384nbt-*.bin.gz` | 主力分析模型 | 384 | ~18B参数量级 |
| `kata1-b28c512nbt-*.bin.gz` | 最强模型 | 512 | 更大 |
| `b18c384nbt-humanv0.bin.gz` | HumanSL 人类模拟模型 | 384 | 同规模 |

---

## 三、Trunk 特征：形状、含义与可视化

### 3.1 物理形状

```
trunk: [trunkNumChannels, nnYLen, nnXLen]
         例: [384, 19, 19]  = 138,624 个 float32

每个空间位置 (row, col) 有 384 个特征值
每个 channel 是一张 19×19 的"特征地图"
```

### 3.2 Trunk 编码的信息（推测性描述）

这些特征**不是人为设计的**，由自我对弈从零学习。但从探针实验和可视化推测，trunk 中包含：

- **局部棋形**：提子威胁、连接结构、气的计数
- **全局影响力**：势力范围、厚势方向
- **死活状态**：哪块棋活着 / 有危险 / 处于战斗中
- **博弈阶段感知**：布局 / 中盘 / 收官的内部表示
- **价值估计中间层**：走这里大约值多少目的中间计算

### 3.3 可视化方法

**方法一：单 Channel 热图**
```python
import numpy as np
import matplotlib.pyplot as plt

# trunk shape: [384, 19, 19]，单手棋的 trunk
trunk = load_trunk_for_move(...)

# 找 activation 最强的 32 个 channel
top_channels = np.argsort(trunk.mean(axis=(1, 2)))[-32:]

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ch in enumerate(top_channels):
    axes[i // 8][i % 8].imshow(trunk[ch], cmap='RdBu', vmin=-3, vmax=3)
    axes[i // 8][i % 8].set_title(f'ch {ch}')
plt.tight_layout()
```

**方法二：PCA 压缩到 RGB（位置着色）**
```python
from sklearn.decomposition import PCA

flat = trunk.reshape(384, 19 * 19).T   # [361, 384]
pca = PCA(n_components=3)
rgb = pca.fit_transform(flat).reshape(19, 19, 3)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
plt.imshow(rgb)
plt.title('PCA-3 of trunk features (spatial)')
```

**方法三：跨多手棋的 channel 方差分析**
```python
# trunks: [N_moves, 384, 19, 19]，一局棋所有手的 trunk
channel_variance = trunks.var(axis=0).mean(axis=(1, 2))  # [384]
# 方差最高的 channel 随棋局变化最显著，最可能编码了动态信息
```

---

## 四、Pick 特征

### 4.1 定义

```
pick[ch] = trunk[ch, move_row, move_col]

即：在实际落子坐标处截取 trunk 的截面向量
形状：[trunkNumChannels]，例如 [384]
```

### 4.2 直觉意义

Pick 表达 KataGo 对"棋手在这个位置落子"的完整内部感知——局部棋形与全局上下文在该点的融合。

**Animiral 的核心假设**：强棋手倾向于在 KataGo trunk 空间中"高价值激活区域"落子。训练一个在 pick 序列上预测 Glicko-2 的模型，隐含地学到了什么样的 pick 分布对应什么样的棋力。

### 4.3 Pick vs Avg-Pool Trunk

| | Pick `[384]` | Avg-Pool Trunk `[384]` |
|---|---|---|
| **含义** | 落子位置的局部感知 | 整局棋盘的全局状态摘要 |
| **信息量** | 决策质量的直接表征 | 局面复杂度 / 整体势力状态 |
| **适用场景** | 棋手选择质量分析 | 局面难度归一化 |
| **Animiral 用法** | 主要特征 | 辅助或归一化 |

---

## 五、HumanSL 模型

### 5.1 基本原理

`b18c384nbt-humanv0.bin.gz` 是用**人类棋谱监督学习**训练的独立模型（非自我对弈）。

```
输入: 当前局面 + humanSLProfile (目标段位)
输出: humanPolicy —— 该段位棋手在此局面的落子概率分布 (19×19)
```

支持的 `humanSLProfile`：
- `rank_20k` ~ `rank_9d`：按段位模拟（现代开局风格）
- `preaz_20k` ~ `preaz_9d`：AlphaZero 之前的开局风格
- `rank_{BR}_{WR}`：非对称段位（黑方 BR 段，白方 WR 段互知）
- `proyear_1800` ~ `proyear_2023`：按年份模拟职业棋手风格

### 5.2 Analysis Engine 可获取的字段

```json
{
  "humanPolicy": [0.02, 0.0, 0.15, ...],
  "policy": [0.34, 0.0, 0.28, ...],
  "rootInfo": {
    "humanWinrate": 0.52,
    "humanScoreLead": -1.3,
    "humanStScoreError": 4.2
  }
}
```

### 5.3 启用方式

```bash
# Analysis Engine 同时加载双模型
./katago analysis \
  -config configs/analysis_example.cfg \
  -model models/kata1-b18c384nbt-*.bin.gz \
  -human-model models/b18c384nbt-humanv0.bin.gz

# 查询中指定 humanSLProfile
{
  "id": "move1",
  "initialStones": [],
  "moves": [...],
  "includePolicy": true,
  "overrideSettings": {
    "humanSLProfile": "rank_3d",
    "ignorePreRootHistory": false
  }
}
```

### 5.4 在棋力评估中的应用

**方法一：HumanSL Log-Likelihood（无需训练，直接可用）**

```python
import numpy as np

def estimate_rank_by_human_sl(moves_and_human_policies, ranks):
    """
    moves_and_human_policies: list of (move_loc, {rank: humanPolicy[rank]})
    ranks: ['rank_20k', 'rank_15k', ..., 'rank_9d']
    """
    log_likelihoods = {r: 0.0 for r in ranks}
    for move_loc, policies in moves_and_human_policies:
        for rank in ranks:
            prob = policies[rank][move_loc]
            log_likelihoods[rank] += np.log(prob + 1e-9)
    return max(log_likelihoods, key=log_likelihoods.get)
```

**方法二：HumanSL 差异分析**

```python
# 对同一局面，比较不同段位的 humanPolicy 与实际落子
# "这手棋更像 5k 还是 1d 的选择？"
for rank in ['rank_5k', 'rank_1d', 'rank_4d']:
    prob = human_policy[rank][actual_move]
    print(f"{rank}: {prob:.4f}")
```

---

## 六、表征理论：正确理解 Trunk 的维度

### 6.1 384/512 是设计容量，不是可学上限

通道数是**超参数**（架构师的设计选择），代表网络的宽度 / 存储容量。不是"注意力能学到的最大维度"，而是"配备了多少存储槽"。

### 6.2 线性表征假说

传统 dummy variable 的逻辑：
```
维度 i = 特征 i（一一对应）
```

深度网络的实际情况：
```
特征 F = 激活空间中的一个方向向量 w_F

激活向量 · w_F 高  →  特征 F 在当前输入中存在
```

"读懂" trunk 不是看哪个通道亮了，而是**找激活向量在哪个方向上有显著投影**。

### 6.3 Superposition（叠加）现象

这是理解 trunk"维度多但信息更多"的关键：

```
384 维空间  →  实际编码了数千个"概念方向"

条件：
  · 每个概念是稀疏激活的（大多数局面不出现）
  · 不同概念的方向近似正交（互相干扰小）
  · 代价：每个概念有微小的"串扰噪声"

结果：
  单看任意一个 channel → 混入了几十个叠加特征 → 人类眼中是"噪声"
  但实际信息全在里面，只是被"压缩编码"了
```

### 6.4 对"噪声维度"的正确理解

低方差、难以直接解读的 channel 并不是空的或无用的，而是：

- 参与了多个特征的叠加编码
- 在特定局面类型下（稀疏地）携带高价值信息
- 对人类是噪声，对网络本身是有意义的叠加分量

对棋力估计而言，**任务相关的信息高度集中在一个低维子空间**——PCA 降到 32-64 维后性能几乎不降，这说明棋力信号只占 trunk 表达能力的一小部分。

---

## 七、特征与人类认知的对齐

### 7.1 问题的本质

```
传统 ML：人设计特征 → dummy variable → 标记清晰
Transformer：自主学习特征 + superposition → 无法直接标记
```

对齐的难点在于：不存在一个 channel 对应一个概念的映射，概念是**激活空间中的方向**，需要专门的工具来发现和命名。

### 7.2 探针分类器（Probing Classifier）

适合**有目标地验证**某个概念是否被线性编码。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# X: trunk avg-pool 向量，[N, 384]
# y: 二值标签，例如 "当前局面是否有打劫"

probe = LogisticRegression(C=0.01, max_iter=1000)
probe.fit(X_train, y_train)
auc = roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1])

# AUC > 0.90 → 该概念被线性编码在 trunk 里
# probe.coef_[0] → 该概念在 384 维空间中的"方向向量"
print(f"Concept AUC: {auc:.4f}")
```

**可构建的围棋概念探针**（标签来源：KataGo 分析结果 + 棋规）：

| 概念 | 标签构造方式 |
|------|------------|
| 打劫 | `boardHistory.ko_loc != PASS` |
| 接不归 | KataGo ownership 急剧变化 |
| 收官阶段 | 手数 > (boardSize² × 1.5) |
| 大场/急所 | `policy[loc] > 0.3` |
| 厚势方向 | 某区域 ownership 均值 > 0.7 |
| 棋形崩溃 | win-rate delta < -0.15 |

### 7.3 稀疏自编码器（SAE）

适合**发现未知概念**，是目前最接近全面解叠加的工具（Anthropic 主力使用）。

**核心架构**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GoTrunkSAE(nn.Module):
    def __init__(self, d_model=384, d_sae=4096):
        super().__init__()
        self.W_enc = nn.Linear(d_model, d_sae, bias=True)
        self.W_dec = nn.Linear(d_sae, d_model, bias=False)
        # decoder 列向量保持单位范数
        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            norms = self.W_dec.weight.norm(dim=0, keepdim=True)
            self.W_dec.weight.div_(norms.clamp(min=1e-6))

    def forward(self, x):
        hidden = F.relu(self.W_enc(x))   # 稀疏激活 [B, d_sae]
        recon = self.W_dec(hidden)        # 重建 [B, d_model]
        return recon, hidden

def sae_loss(x, recon, hidden, lam=1e-3):
    recon_loss = (recon - x).pow(2).mean()
    sparsity_loss = hidden.abs().mean()
    return recon_loss + lam * sparsity_loss
```

**训练数据**：从 batch_analysis 提取的 trunk avg-pool 向量。
- 每局棋 ~200 手 × 数千局 → 数十万样本，足够训练小型 SAE。

**分析 SAE 特征**：

```python
# 找激活率最低（最稀疏）且激活时最一致的 feature
activation_rate = (hidden > 0).float().mean(dim=0)   # [d_sae]
sparse_features = activation_rate.argsort()[:200]    # 最稀疏的 200 个

# 对每个稀疏 feature，找使其激活的棋局位置
for feat_idx in sparse_features:
    active_moves = (hidden[:, feat_idx] > threshold).nonzero()
    # 可视化这些局面 → 人工命名该特征
```

### 7.4 现有相关工作

| 工作 | 关联性 | 链接 |
|------|--------|------|
| **Othello-GPT** (Li et al., 2022) | 最直接先例：证明博弈 transformer 自发形成棋盘世界模型，线性探针可 >99% 恢复棋盘状态 | arXiv:2210.13382 |
| **SAELens** (Joseph Bloom) | 成熟的 SAE 训练库，针对 LLM 但架构通用 | github.com/jbloomAus/SAELens |
| **Captum** (Meta) | 通用神经网络可解释性工具 | captum.ai |
| **Anthropic Scaling Monosemanticity** | SAE 在 Claude 上的大规模应用，方法论参考 | anthropic.com/research |

---

## 八、数据结构设计原则（三层分离）

### 8.1 设计禁忌

| 原则 | 问题描述 | 正确做法 |
|------|---------|---------|
| **标签不入特征** | Label Ranking 若在输入特征里，模型直接"抄答案"，测试高分但零泛化 | 标签只在 loss 计算时出现 |
| **环境信息不训练** | GPU 型号、CUDA 版本与棋力无关，混入特征只引入虚假相关性 | 移到溯源字段，不参与训练 |
| **按性质分层** | 逐手特征、局级元数据、环境数据三种性质混入同一结构，序列模型每步重复处理不变量 | 三层严格分离 |
| **序列结构保留时序** | 对逐手特征做全局聚合后训练，丢失棋局弧线（开局失误→中盘追赶）| pick + scalars 作为序列时间步输入 |

### 8.2 三层数据结构

```
┌── GameHeader（每局一次，不进序列模型）
│     komi, rules, board_size, handicap
│     result, date
│     analysis_model_hash      ← 溯源用，不训练
│     katago_version            ← 溯源用，不训练
│
├── Label（训练目标，绝对不进输入）
│     black_katago_score        ← 模型主要预测目标 = meanLogPrior（连续值）
│     white_katago_score
│     black_rank_label          ← 经 ln 映射后的段位参考标签
│     white_rank_label
│
└── MoveSequence[N]（序列模型的输入，对应 scalars[0..9]）
      每步:
        [0] whiteWinProb    : float       ← Value Head（白方视角，全文统一）
        [1] whiteLossProb   : float       ← Value Head
        [2] whiteNoResult   : float       ← Value Head
        [3] whiteScoreMean/50: float      ← Value Head（归一化）
        [4] scoreStdev/10   : float       ← shorttermScoreError（局面复杂度代理）
        [5] policyPrior     : float       ← Policy Head 衍生（实际落子概率）
        [6] policyRank/361  : float       ← Policy Head 衍生（排名归一化，0=最优）
        [7] isWhite         : float       ← 1.0=白方走棋，0.0=黑方走棋
        [8] winDelta        : float       ← whiteWinProb[t+1] - whiteWinProb[t]（延迟填充）
        [9] scoreDelta/50   : float       ← whiteScoreMean[t+1] - whiteScoreMean[t]（延迟填充）
        pick[384]           : float[]     ← Trunk 在落子坐标的切片
        avg_trunk[384]      : float[]     ← Trunk 全棋盘空间均值（NCHW layout）
```

> **白方视角统一约定**：scalars[0..9] 全部保持白方视角，下游模型通过 `isWhite` 决定是否翻转。  
> **延迟填充**：`winDelta`/`scoreDelta` 在 turn t+1 的 NN 评估完成后回填到 turn t 的记录，最后一步保持 0。

### 8.3 条件变量（komi/rules）的注入位置

komi 和规则**不应放在最末层**，应在序列入口通过 FiLM 条件化注入，使 Transformer 在做注意力时已知贴目信息：

```python
# FiLM conditioning：对每个 move embedding 做 scale + shift
scale, shift = MLP(game_condition)   # game_condition = [komi, rules_embedding]
move_emb = move_emb * scale + shift  # 进 Transformer 之前完成
```

同样的落子选择在贴目 6.5 与贴目 0.5 下的战略含义完全不同，必须在 attention 阶段就能感知。

---

## 九、评估模型架构设计

### 9.1 设计哲学

独立建模每位棋手的**决策序列模式**，再通过交叉注意力感知对局互动，输出纯 KataGo 评分（不依赖任何外部评级体系）。

**重要认知**：黑白分开处理≠棋手主观视角隔离——pick 和 avg_trunk 本身已包含完整棋盘信息（含对手落子）。正确的描述是：**独立建模每位棋手的决策序列模式**，改变的是时序结构，不是信息量。

### 9.2 整体架构

```
输入层
  Black MoveSeq [Nb, 776]    White MoveSeq [Nw, 776]
  GameCondition [komi, rules, handicap]
        ↓
  FiLM Conditioning（GameCondition 注入每步 embedding）
        ↓
  ┌─ Black Self-Attention Transformer ─┐
  │  学习黑方决策序列的内部模式         │
  └─────────────────────────────────────┘
  ┌─ White Self-Attention Transformer ─┐
  │  学习白方决策序列的内部模式         │
  └─────────────────────────────────────┘
        ↓
  Causal Cross-Attention（带因果掩码）
  Black attend to past White moves，反之亦然
        ↓
  分段注意力池化
  [opening_B, midgame_B, endgame_B]
  [opening_W, midgame_W, endgame_W]
        ↓
  多任务输出头
  ├─ 黑方 KataGo 综合评分（连续，主任务）
  ├─ 白方 KataGo 综合评分（连续，主任务）
  ├─ 黑方阶段分项（开局 / 中盘 / 收官）
  ├─ 白方阶段分项
  └─ 段位标签（ln 映射后，辅助校准用）
```

### 9.3 因果掩码设计

黑白序列交错（B1→W1→B2→W2→...），Cross-Attention 必须施加因果约束，否则 Black move k 会看到 White move k 及之后的落子，构成信息泄漏：

```python
def build_causal_cross_mask(nb, nw):
    # Black move k 只能 attend 到 White move 0...(k-1)
    mask = torch.full((nb, nw), float('-inf'))
    for k in range(nb):
        if k > 0:
            mask[k, :k] = 0.0   # White move 0..k-1 可见
    return mask   # [nb, nw]
```

### 9.4 分段注意力池化

替代简单 mean pooling，避免妙招被 200 手平均稀释（2 手妙招 / 200 手 = 1% 贡献）：

```python
class SegmentedAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, seq, n_moves):
        s1, s2 = n_moves // 3, 2 * n_moves // 3
        segs = [seq[:s1], seq[s1:s2], seq[s2:]]
        out = []
        for seg in segs:
            w = torch.softmax(self.score(seg), dim=0)
            out.append((seg * w).sum(dim=0))
        return torch.cat(out)   # [3 * d_model]
```

注意力权重会自动偏向高 `score_stdev`（复杂局面）和高 `policy_rank`（意外落子）的手。

### 9.5 多任务损失

```python
loss = (
    w_score * mse(pred_score_B, true_score_B)
  + w_score * mse(pred_score_W, true_score_W)
  + w_phase * mse(pred_phase_B, true_phase_B)   # 三段分项
  + w_phase * mse(pred_phase_W, true_phase_W)
  + w_rank  * ce(pred_rank_B,   rank_label_B)   # 辅助分类，用于校准
  + w_rank  * ce(pred_rank_W,   rank_label_W)
)
# 建议初始权重比: w_score=1.0, w_phase=0.3, w_rank=0.1
```

---

## 十、评分体系设计哲学

### 10.1 核心原则：完全基于 KataGo，不依赖外部评级

本系统**不使用 Glicko-2、ELO 或任何外部段位体系**作为训练目标。所有评分信号均来自 KataGo 内部输出的统计聚合。

理由：
- 外部评级含有主观因素（服务器膨胀、sandbagging、时代差异）
- 目标是"KataGo 眼中的棋力"，而非"人类评级体系的棋力估计"
- 独立评分体系可跨服务器、跨时代公平比较

### 10.2 评分的两层结构

```
第一层：KataGo 综合评分（原始连续值）
  = meanLogPrior = mean_t[ log( policyProbs[actualMove_t] ) ]
  = 模型主要输出，数值本身有意义（负数，越接近 0 越强）
  = 可直接比较（"这局黑棋得了 -1.82"）
  = 完全客观可复现：同一 SGF + 同一模型 → 完全相同结果

第二层：段位标签（可解释的参考映射）
  = 将综合评分通过 ln-normal 映射投影到 [0, 50] 区间
  = 仅作为人类可读的参考，不是主要训练目标
  = 参数需经验标定，随数据积累逐步调整
```

> 详见 **Section 15** 对 meanLogPrior 公式的完整论证。

### 10.3 ln 映射设计

段位在人类认知中近似等间距（每段胜率差约固定），分数分布呈近似对数正态分布：

```
原始评分 s ∈ (0, +∞)
    ↓
ln(s) → 近似正态分布 N(μ, σ²)
    ↓
标准化: z = (ln(s) - μ) / σ
    ↓
线性映射到 [0, 50]: rank_label = clip(25 + z × k, 0, 50)

其中：
  μ, σ  → 经验值，由分析数据的分布决定，需初始估计后逐步校准
  k     → 控制分布铺展宽度的缩放因子（初始建议 8.0）
  25    → 中位段位（约对应业余中等水平）
  [0, 50] → 涵盖完全初学到职业顶尖的全段位区间
```

```python
import numpy as np

def score_to_rank_label(score, mu, sigma, k=8.0):
    z = (np.log(score + 1e-6) - mu) / sigma
    return float(np.clip(25.0 + z * k, 0.0, 50.0))

# 标定流程：
# 1. 收集大量已分析棋谱，计算每局综合评分
# 2. 对 ln(score) 拟合正态分布，得到初始 mu, sigma
# 3. 将分布锚点与人类认知对齐（ln 中位 ≈ 25 = 业余中等）
# 4. 随数据积累，用 MLE 持续更新 mu, sigma
```

### 10.4 用 HumanSL 做初始锚定（无外部数据时）

在获得足够自有数据之前，可借助 HumanSL 的分析结果做粗标定：

| HumanSL Profile | rank_label 目标锚点 |
|----------------|-------------------|
| rank_5k 分布中位数 | ≈ 25 |
| rank_1d 分布中位数 | ≈ 30 |
| rank_5d 分布中位数 | ≈ 35 |
| rank_9d 分布中位数 | ≈ 42 |

---

## 十一、实践路径（修订版）

### 11.1 总体流水线

```
SGF 文件
    ↓
batch_analysis（本 fork，修复后）
    ├─ MoveSequence: [win_prob, score_mean, score_stdev,
    │                 policy_prior, policy_rank, pick[384], avg_trunk[384]]
    └─ GameHeader:   [komi, rules, result, date, model_hash]
    ↓
（可选）Analysis Engine: humanPolicy[rank_k] → 用于初始标定
    ↓
评估模型（Section 9 架构）
    ↓
输出 JSON
```

### 11.2 输出格式

```json
{
  "black": {
    "katago_score": 73.2,
    "rank_label": 27.4,
    "rank_label_readable": "~3k",
    "phase_scores": {
      "opening": 71.0,
      "midgame": 68.5,
      "endgame": 82.1
    }
  },
  "white": {
    "katago_score": 81.6,
    "rank_label": 31.2,
    "rank_label_readable": "~2d",
    "phase_scores": {
      "opening": 85.0,
      "midgame": 79.3,
      "endgame": 80.8
    }
  }
}
```

### 11.3 三阶段开发路线（修订）

**阶段 1（基础设施）**
- 修复 inference mode 输出空文件的 bug（Section 12.2）
- 修正 `trunkCh` 硬编码问题（Section 12.3）
- 实现三层数据结构的序列化
- 产出：可靠的特征提取管道

**阶段 2（模型训练基线）**
- 训练 Section 9 架构的评估模型
- 用 HumanSL 做初始 ln 分布标定（Section 10.4）
- 产出：可运行的评分系统，段位标签初步对齐

**阶段 3（迭代校准 + 可解释性）**
- 积累评分数据，用 MLE 更新 μ, σ
- 训练围棋概念探针（Section 7），增加可解释维度
- 可选：训练 SAE，发现隐含棋感模式
- 产出：段位标签稳定收敛，评分体系完备

---

## 十二、当前 Fork 状态（已修复）

### 12.1 本次会话完成的修改

| 文件 | 修改内容 |
|------|---------|
| `cpp/neuralnet/nneval.h` | 新增 `int getModelTrunkNumChannels() const` 声明 |
| `cpp/neuralnet/nneval.cpp` | 实现：`return NeuralNet::getModelDesc(loadedModel).trunk.trunkNumChannels` |
| `cpp/command/batch_analysis.cpp` | **完全重写**（见 12.2） |

### 12.2 batch_analysis.cpp 修复与增强汇总

**原版已知 Bug（全部修复）：**

| Bug | 原因 | 修复方式 |
|-----|------|---------|
| Inference mode 输出空文件 | `infRecords_*` 从未 push_back | 废弃旧结构，改用 `data_B`/`data_W` 每步直接追加 |
| `trunkCh = 256` 硬编码 | 与模型实际通道数不符 | 改为 `nnEval->getModelTrunkNumChannels()` |
| Training mode trunk 颜色分配 bug | trunk 数据始终追加到 `trunkBuf_black` | 两个 buffer 均废弃，新格式不再区分模式 |
| `extractHead()` 包含占位符 | head[0,1,3,7,8,9,10] 均为硬编码常数 | 全部替换为真实 `NNOutput` 字段 |
| `-training` / `-head-only` / `-trunk-only` 模式区分 | 格式碎片化，互不兼容 | 统一为单一 KAB2 格式，始终输出完整特征 |

**新增功能：**

- `PlayerAcc` 累加器：`addMove()` + `addDelta()` + `toSummary()`，Welford 在线方差
- `PlayerSummary` 嵌入 NPZHeader（12项指标，64 bytes；含 `humanRankIdx`/`humanLogPrior`）
- `winDelta` / `scoreDelta` 延迟回填机制（turn t+1 知道 t 的 delta 后填入）
- `-sgf-dir` 实现：`FileUtils::collectFiles()` 扫描 `.sgf` 文件
- `_meta.csv` 包含双方全部 `PlayerSummary` 字段及 HumanSL 结果列
- **[已实现] HumanSL 二阶段**：`-human-model` 加载第二个 `NNEvaluator`，对每局每棋手查询 3 个候选段位，选最大 log-likelihood 档位，结果写入 `PlayerSummary.humanRankIdx`/`humanLogPrior` 及 `_meta.csv`
- **[机制已改] Pick 提取**：改为从 `nn->trunk` 直接提取 `trunk[ch * spatial + rowPos]`（NCHW），不再依赖 `includePick` 后端路径
  - ⚠️ **待验证**：实测发现 pick 数据仍约 2-4/512 非零（几乎全零），有两个竞争假设：
    - **假设 A（索引错误）**：backend 实际以 NHWC 存储 trunk，NCHW 假设导致读到错误位置；需对比 `avg_trunk` 与 `pick` 的非零通道分布来确认
    - **假设 B（语义稀疏）**：落子点评估在落子**前**进行，是空点，经 ReLU 多层后激活天然极稀疏，pick 全零是正确行为但信息量低
  - **当前状态**：**非 blocker**，KAB2 格式中 pick 字段已预留空间，scalar 和 avgTrunk 不受影响，训练管线可暂时忽略 pick 通道

### 12.3 当前可用状态

```
$ katago batch_analysis \
    -model models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
    -list games.csv \
    -output-dir out/

# 无 HumanSL
batch-analysis: 1 games  trunkCh=384  moveDim=778  headerBytes=96  output=out/

# 加 HumanSL（每局额外 ~0.4s）
$ katago batch_analysis -model ... -human-model humansl.bin.gz -list games.csv -output-dir out/
batch-analysis: 1 games  trunkCh=384  moveDim=778  headerBytes=96  [+HumanSL x3]  output=out/

输出：
  out/game_0000000000000000_B.npz   ← KAB2 格式，PlayerSummary 含 humanRankIdx
  out/game_0000000000000000_W.npz
  out/_meta.csv                     ← 含 B_humanRank、W_humanRank 列
```

---

## 十三、数据格式参考（KAB2 v2）

### 13.1 NPZHeader（96 bytes）

```cpp
// PlayerSummary：12 项聚合统计，64 bytes
struct PlayerSummary {
  float accuracy1;        // 落子与 KataGo top-1 一致率
  float accuracy3;        // 落子在 KataGo top-3 内的比率
  float meanLogPrior;     // mean(log(policyPrior)) — 核心强度信号
  float meanWinRate;      // 该玩家视角平均胜率
  float meanScoreLead;    // 该玩家视角平均得分领先（points）
  float meanComplexity;   // mean(shorttermScoreError)
  float scoreVariance;    // 得分领先方差（Welford 在线算法）
  float approxScoreDrop;  // mean(max(0, -playerWinDelta)) — 失误代理
  float meanWinDelta;     // mean 签名胜率变化（玩家视角）
  float meanScoreDelta;   // mean 签名得分变化（玩家视角，points）
  float humanRankIdx;     // HumanSL 最匹配段位索引（0=20k…28=9d），-1 表示未计算
  float humanLogPrior;    // 最佳 HumanSL 档位下的 meanLogPrior
  float reserved[4];      // 保留，共 16 × 4 = 64 bytes
};

#pragma pack(push, 1)
struct NPZHeader {           // 总计 96 bytes
  char     magic[4];        // "KAB2"
  int32_t  numMoves;        // 该玩家的落子数
  int32_t  scalarDim;       // = 10（SCALAR_DIM）
  int32_t  trunkDim;        // avg-pool trunk 通道数（动态，如 384）
  int32_t  pickDim;         // pick 通道数（= trunkDim）
  int32_t  nnXLen;          // 棋盘宽（= 19）
  int32_t  nnYLen;          // 棋盘高（= 19）
  int32_t  flags;           // bit0 = zlib 压缩
  PlayerSummary summary;    // 64 bytes
};
#pragma pack(pop)
```

### 13.2 Per-Move 数据布局

每步棋的特征块（**interleaved**，连续存储）：

```
[ scalars(10) ][ pick(trunkCh) ][ avgTrunk(trunkCh) ]

moveDim = 10 + 2 × trunkCh
         b18c384nbt: 10 + 768 = 778 floats = 3112 bytes / move
         b28c512nbt: 10 + 1024 = 1034 floats = 4136 bytes / move
```

Scalars 索引（白方视角统一）：

```
[0] whiteWinProb        [1] whiteLossProb    [2] whiteNoResultProb
[3] whiteScoreMean/50   [4] scoreError/10
[5] policyPrior         [6] policyRank/361   [7] isWhite
[8] winDelta            [9] scoreDelta/50
```

### 13.2.1 向量长度：文件内固定，文件间可变

> **关键设计决策**（勿改为可变长格式）

`moveDim = 10 + 2 × trunkCh` 在同一文件内**严格固定**——因为一个文件只来自一个模型的一次分析，`trunkCh` 不会中途变化。但若混合来自不同模型架构的文件（如 b18c384 vs b28c512），则不同文件的 `moveDim` 不同（778 vs 1034）。

**为什么不改为可变长 + 零填充？**

- 零填充在格式上可行（PyTorch `collate_fn` 一行搞定），但**语义上错误**：b18c384 的第 217 通道与 b28c512 的第 217 通道是完全不同的表征，强行对齐只会引入噪声
- `trunkCh` 不同的 trunk 向量**本质上不可比**，不能共享同一组权重

**正确的跨模型对齐方式：投影头路由**

```python
# 训练管线：按 trunkDim 分组，各架构接独立线性投影
trunk_projections = {
    384: nn.Linear(384, 128),   # b18c384nbt
    512: nn.Linear(512, 128),   # b28c512nbt
}

for npz_path in dataset:
    hdr = read_header(npz_path)          # 读取 96-byte NPZHeader
    trunk_dim = hdr["trunkDim"]
    proj = trunk_projections[trunk_dim]  # 路由到对应投影头
    emb = proj(trunk_features)           # [N, trunkDim] → [N, 128]
    # 128-dim embedding 之后统一训练，共享所有上层权重
```

| 方案 | 实现复杂度 | 语义正确性 |
|------|-----------|-----------|
| 零填充到最大 moveDim | 低 | **错误**（通道不可比） |
| 可变长格式 + offset table | 高 | 正确但无必要 |
| **文件内固定长 + 投影头路由** | 低 | **正确**（推荐） |

**结论**：KAB2 格式保持现状（文件内固定，header 记录尺寸）；跨模型对齐由训练时的**架构特定投影层**处理。

### 13.3 PlayerAcc 累加器逻辑（C++）

```cpp
struct PlayerAcc {
  // addMove()：每步评估后立即调用
  void addMove(int rank, float policyPrior,
               float playerWinRate, float playerScoreLead, float complexity);

  // addDelta()：turn t+1 评估后回填 turn t 的 delta
  void addDelta(float playerWinDelta, float playerScoreDelta);

  // toSummary()：游戏结束后生成 PlayerSummary
  PlayerSummary toSummary() const;

  // 延迟回填机制：
  //   prevBuf[prevRecordIdx * moveDim + 8] = whiteWinProb[t+1] - whiteWinProb[t]
  //   prevBuf[prevRecordIdx * moveDim + 9] = (whiteScoreMean[t+1] - prevScoreMean) / 50
};
```

### 13.4 Python 读取

```python
import zlib, struct, numpy as np

SCALAR_DIM = 10

def load_kab2(path):
    with open(path, 'rb') as f:
        # Header: magic(4) + 7×int32 + PlayerSummary(64) = 96 bytes
        magic = f.read(4)
        assert magic == b'KAB2', f"bad magic: {magic}"
        n, sd, tk, pk, nx, ny, flags = struct.unpack('<7i', f.read(28))
        summary_raw = f.read(64)   # PlayerSummary (16 floats)
        summary = np.frombuffer(summary_raw, dtype=np.float32)

        raw = f.read()
        if flags & 1:
            clen = struct.unpack('<i', raw[:4])[0]
            raw = zlib.decompress(raw[4:4+clen])

    stride = sd + tk + pk          # e.g. 10 + 384 + 384 = 778
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, stride)

    scalars   = arr[:, :sd]        # [n, 10]
    pick      = arr[:, sd:sd+tk]   # [n, 384]
    avg_trunk = arr[:, sd+tk:]     # [n, 384]

    return scalars, pick, avg_trunk, summary

# PlayerSummary 字段索引（共 16 floats，后 4 个为 reserved）
SUMMARY_FIELDS = [
    'accuracy1', 'accuracy3', 'meanLogPrior', 'meanWinRate',
    'meanScoreLead', 'meanComplexity', 'scoreVariance', 'approxScoreDrop',
    'meanWinDelta', 'meanScoreDelta',
    'humanRankIdx',    # 0-28（20k→9d），-1 表示未启用 HumanSL
    'humanLogPrior',   # 最佳档位的 mean log-likelihood
]

# Scalars 字段索引（用于下游模型）
# s[:, 0] = whiteWinProb,  s[:, 8] = winDelta,  s[:, 7] = isWhite
# 玩家视角胜率：player_win = np.where(s[:,7]==1, s[:,0], 1-s[:,0])
```

### 13.5 _meta.csv 列定义

```
file, black, white, black_elo, white_elo, total_moves, black_moves, white_moves, set,
B_acc1, B_acc3, B_logPrior, B_winRate, B_scoreLead, B_complexity, B_scoreVar, B_drop,
B_humanRank, B_humanLogPrior,
W_acc1, W_acc3, W_logPrior, W_winRate, W_scoreLead, W_complexity, W_scoreVar, W_drop,
W_humanRank, W_humanLogPrior
```

- `B_logPrior`：黑方 `meanLogPrior`，核心强度指标，可直接用于排序和分组
- `B_humanRank`：HumanSL 最匹配段位字符串（如 `"rank_3d"`），未启用 `-human-model` 时为 `"?"`
- `B_humanLogPrior`：该段位档位下的平均 log-likelihood，值越大越匹配

---

## 十四、HumanSL 集成策略与开销分析

> **状态：已实现（batch_analysis.cpp v3）**
> 通过 `-human-model` 参数启用。对每局每棋手，先用主模型 meanLogPrior 估算候选段位范围，
> 再调用 HumanSL 模型查询 3 个候选段位，最终结果写入 `PlayerSummary.humanRankIdx`。

### 14.0 实际实现概要

```
流程（每局）
  1. 主模型前向传播 → 聚合 meanLogPrior（B/W 分别）
  2. rankCandidates(logPrior)
       t = clamp((logPrior + 5.9) / 5.4, 0, 1)
       mid = clamp(round(t * 28), 1, 27)
       candidates = [mid-1, mid, mid+1]   ← 3 个相邻段位索引
  3. runHumanSLPass()：重播棋谱，每手棋对当前棋手的 3 候选分别调用 humanEval
       hbuf.includeTrunk = false   ← 仅需 policy head
       humanEval->evaluate(board, history, pla, &meta, ...)
       sumLog[k] += log(policyProbs[rowPos])
  4. 选最大 log-likelihood 候选 → 写入 PlayerSummary.humanRankIdx / humanLogPrior
  5. _meta.csv 追加 B_humanRank / W_humanRank 列
```

段位索引 ↔ 字符串映射（共 29 档）：
`0="rank_20k"  1="rank_19k"  ...  19="rank_1k"  20="rank_1d"  ...  28="rank_9d"`

### 14.1 纯 KataGo 分析即可得到每棋手本盘得分

不依赖 HumanSL，仅用主模型的逐手输出即可直接聚合出每棋手的得分：

```python
# 每手棋已有：win_delta, score_delta, policy_prior, score_stdev
# 按颜色分组聚合

black_score = aggregate(
    win_deltas   = [win_delta[t]   for t if is_black[t]],
    score_deltas = [score_delta[t] for t if is_black[t]],
    policy_qs    = [policy_q[t]    for t if is_black[t]],
)
```

此得分完全客观，不依赖任何外部评级体系。HumanSL 的唯一作用是为这个原始得分提供"段位标尺"校准。

### 14.2 HumanSL 参考向量：policy head 为主，value head 可选

| 字段 | 是否需要 | 用途 |
|------|---------|------|
| `humanPolicy[rank_k][actual_move]` | **必须** | log-likelihood 段位匹配的核心信号 |
| `humanWinrate[rank_k]` | 可选 | 与 KataGo 客观胜率对比 → 认知偏差维度 |
| `humanScoreLead[rank_k]` | 不需要 | 对段位估计无增量贡献 |
| `humanStScoreError[rank_k]` | 不需要 | 同上 |

段位估计的核心计算：

```python
def rank_log_likelihood(moves, human_policies, rank):
    return sum(
        np.log(human_policies[rank][move.loc] + 1e-9)
        for move in moves
    )

best_rank = max(candidate_ranks,
                key=lambda r: rank_log_likelihood(moves, human_policies, r))
```

### 14.3 开销分析（基于 batch_analysis 无搜索模式）

当前 `batch_analysis` 每手棋调用一次 `nnEval->evaluate()`，无 MCTS 搜索。加入 HumanSL 需第二个 `nnEval` 实例：

```cpp
auto humanEval = Setup::initializeNNEvaluator(
    humanModelFile, humanModelFile, "", cfg, logger, seedRand, 64,
    nnXLen, nnYLen, defaultMaxBatch, true, false, Setup::SETUP_FOR_ANALYSIS
);
// 每手棋额外调用 M 次（每档位一次）
for(auto& rankProfile : candidateRanks) {
    MiscNNInputParams humanParams;
    humanParams.humanSLProfile = rankProfile;
    humanEval->evaluate(board, history, evalPla, humanParams, humanBuf, false, false);
}
```

| 方案 | 每局耗时估算 | 备注 |
|------|------------|------|
| 纯主模型（无搜索）| ~0.5s | 当前模式 |
| + HumanSL × 3 档位 | ~0.9s | 预估范围后定向查询 |
| + HumanSL × 5 档位 | ~1.5s | 全段位覆盖 |
| + HumanSL × 15 档位 | ~3.5s | 穷举所有档位 |
| 主模型 + MCTS 100 visits | ~20s | Analysis Engine 完整模式 |

HumanSL 模型（b18c384nbt）单次 eval 约为主力模型的 40% 耗时。3 档位方案开销约增加 80%，绝对时间仍可接受。

### 14.4 预估段位范围，定向查询（推荐方案）

**两阶段流程**：

```python
# 阶段一：主模型分析完成后，计算粗略得分
mean_score_loss = np.mean([abs(d) for d in score_deltas])
mean_policy_q   = np.mean(policy_priors)

# 粗标定映射（经验值，随数据积累校准）
def estimate_rank_range(mean_score_loss):
    if   mean_score_loss > 3.0:  return ['rank_20k', 'rank_15k', 'rank_10k']
    elif mean_score_loss > 1.5:  return ['rank_10k', 'rank_5k',  'rank_2k' ]
    elif mean_score_loss > 0.6:  return ['rank_2k',  'rank_1d',  'rank_3d' ]
    elif mean_score_loss > 0.2:  return ['rank_3d',  'rank_5d',  'rank_7d' ]
    else:                        return ['rank_6d',  'rank_8d',  'rank_9d' ]

# 阶段二：只对 3 个候选档位跑 HumanSL
candidate_ranks = estimate_rank_range(mean_score_loss)
# 200 手 × 3 档位 = 600 次额外 eval，约 +0.4s
```

**二分搜索方案**（更精确，约 5-7 档位）：

```
初始区间 [rank_20k, rank_9d]
  → 测试区间中点 → 取最高 log-likelihood 侧收缩
  → 重复 2-3 次
  → 总计约 5-7 档位，精度 ±1-2 段
```

### 14.5 推荐集成架构

```
步骤 1：主模型前向传播（batch_analysis 现有逻辑）
  输出: pick, avg_trunk, win_prob, score_mean, score_stdev, policy_prior
  耗时: ~0.5s/局

步骤 2：快速段位预估（纯计算，无 NN）
  输入: 步骤 1 的聚合得分
  输出: 3 个候选 HumanSL 档位
  耗时: <1ms

步骤 3：HumanSL 定向查询（3 档位 × 200 手，无搜索）
  输出: humanPolicy[rank][actual_move] per move
  耗时: ~0.4s/局

步骤 4：log-likelihood 计算 → 最终段位参考标签

总计: ~1s/局（vs 纯主模型 ~0.5s，开销约 +100%）
```

---

## 十五、客观评分公式：meanLogPrior

### 15.1 公式定义

```
gameScore(player) = mean_t [ log( policyProbs[actualMove_t] ) ]
                  = PlayerSummary.meanLogPrior
```

其中 `policyProbs[actualMove_t]` 是 KataGo 主模型在当前局面下，对棋手实际落子位置给出的策略概率。

### 15.2 为何选择这个公式

| 性质 | 说明 |
|------|------|
| **完全确定性** | 同一 SGF + 同一模型 → 逐字节相同的输出，无随机性 |
| **无自由参数** | 不需要权重、阈值或任何人工校准常数 |
| **理论根基** | 等价于棋手策略与 KataGo 策略的负 KL 散度：`-KL(human‖KataGo)` |
| **单调强度相关** | 强者更倾向选择 KataGo 高概率的落点，分数更高（更接近 0） |
| **量程直觉** | 随机落子 ≈ log(1/362) ≈ -5.9；顶尖选手 ≈ -0.5 ~ -1.0 |

### 15.3 量程参考（经验估计，待数据校准）

```
meanLogPrior 范围   对应棋力层次（粗估）
─────────────────   ──────────────────────
  -0.5 ~ -1.0       职业顶尖 / KataGo 自弈级别
  -1.0 ~ -1.8       高段职业 / 业余顶尖（7d+）
  -1.8 ~ -2.5       业余中高水平（3d ~ 6d）
  -2.5 ~ -3.5       业余中等（1k ~ 2d）
  -3.5 ~ -4.5       业余初中级（5k ~ 10k）
  -4.5 以下          初学者
```

> 上表为粗估，需用有标注数据集（含已知段位棋谱）校准。

### 15.4 ln-normal 投影到 [0, 50]

```python
from scipy.stats import norm
import numpy as np

def score_to_rank_label(mean_log_prior, mu, sigma, k=8.0):
    """
    mean_log_prior: PlayerSummary.meanLogPrior（负数）
    mu, sigma: 由校准数据集的 meanLogPrior 分布拟合（初始可用 HumanSL 锚定）
    k: 分布铺展因子（初始建议 8.0）
    返回: 0~50 的段位标签（25 = 业余中等）
    """
    z = (mean_log_prior - mu) / sigma
    return float(np.clip(25.0 + z * k, 0.0, 50.0))

# 校准流程（无监督，仅用 KataGo 分析结果）：
# 1. 收集大量棋谱的 meanLogPrior
# 2. 拟合正态分布 → 得到初始 mu, sigma
# 3. 用 HumanSL 段位锚点（见 Section 10.4）对 mu 做零点校准
# 4. 随数据积累用 MLE 持续更新
```

### 15.5 与 HumanSL 的对称关系

主模型得分和 HumanSL 得分用**完全相同的公式**，只换模型来源：

```python
# 主模型得分（已在 batch_analysis 中计算）
main_score = mean(log(mainModel.policyProbs[actualMove]))

# HumanSL 段位匹配得分（二阶段查询）
def humansl_score(moves, human_policies, rank_k):
    return mean(log(human_policies[rank_k][actualMove] + 1e-9)
                for actualMove in moves)

# 找最匹配的段位
best_rank = max(candidate_ranks, key=lambda r: humansl_score(moves, human_policies, r))
```

`main_score`（= meanLogPrior）作为粗估，驱动 Section 14.4 的"预估范围→定向查询"流程，两者共同输出最终段位参考标签。

### 15.6 已在 batch_analysis 中实现

- `PlayerAcc::addMove()` 中：`sumLogPrior += std::log(std::max(policyPrior, 1e-10f))`
- `PlayerAcc::toSummary()` 中：`s.meanLogPrior = sumLogPrior / n`
- 写入 `NPZHeader.summary.meanLogPrior`（二进制头部，不解压即可读取）
- 写入 `_meta.csv` 的 `B_logPrior` / `W_logPrior` 列

---

## 参考链接

- KataGo 主库：https://github.com/lightvector/KataGo
- KataGo Analysis Engine 文档：`docs/Analysis_Engine.md`（本项目内）
- HumanSL 模型下载：https://github.com/lightvector/KataGo/releases/tag/v1.15.0
- Animiral/go-strength-model：https://github.com/Animiral/go-strength-model
- Othello-GPT 论文（arXiv:2210.13382）：博弈 transformer 世界模型的最直接先例
- SAELens：https://github.com/jbloomAus/SAELens
- Anthropic Scaling Monosemanticity：https://anthropic.com/research/scaling-monosemanticity

---

*文档版本：2026-06-09 v3 | 本次更新：batch_analysis 完全重写（KAB2格式、SCALAR_DIM=10、PlayerSummary、延迟delta回填）、新增 Section 15（meanLogPrior 客观评分公式）、Section 8/10/12/13 同步修订*
