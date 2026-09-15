# CRAA — Causal–Reachability Aligned Attack 计划(最小化可行性判断)

- Written: 2026-09-14
- Parent decision: P0-C route=D(MCCA 暂停,content-loss 梯度无法推内容)
- Supersedes: MRA("Measure → Project → PGD" 已因一阶投影等价性被废弃)
- Status: **计划 + 可行性判断阶段,尚未跑 GPU**

---

## 1. 一句话研究问题

> 内部方向的**因果性**不足以决定它能否成为图像攻击目标;还必须考虑它相对像素输入的**局部可控性**。

$$
\text{有效攻击目标} = \underbrace{C_l}_{\text{因果性}} \times \underbrace{R_l}_{\text{像素可控性}}
$$

新意不是"投影一下",而是**选择因果性 × 可控性同时最高的层/方向**去攻击。这是区别于"找最因果方向"(已有人做)和"用 SVD 投影改进攻击"(已有人做)的第三件事。

---

## 2. 关键数学(已通过 CPU 测试验证)

设 $J = \partial h_l / \partial x \in \mathbb R^{d\times p}$,$u$ 为目标方向。

### 2.1 为什么放弃 MRA 投影(硬伤)

$$J^\top P_{\operatorname{col}(J)} u = J^\top u \quad\Rightarrow\quad \nabla_x \mathcal L_{u_{\mathrm{proj}}} = \nabla_x \mathcal L_u$$

投影是恒等操作——反向传播本来就会自动丢弃 $u$ 中落在 $\operatorname{null}(J^\top)$ 的分量。

### 2.2 真正的像素可控性(对偶范数)

$$\max_{\|\delta\|_\infty \le \varepsilon} u^\top J\delta = \varepsilon\,\|J^\top u\|_1$$

$$R_\infty(u) = \frac{\|J_l(x,q)^\top u_l\|_1}{\|u_l\|_2} \qquad R_2(u) = \frac{\|J_l(x,q)^\top u_l\|_2}{\|u_l\|_2}$$

实现:$J^\top u$ 用一次 `torch.autograd.grad(u^⊤h, x)` 得到,**不显式构造 $J$**(避免 $3584 \times 338688$ 的内存)。

建议同时报告三个量:

| 量 | 含义 |
|---|---|
| $\|U_k^\top u\|_2 / \|u\|_2$ | 目标落在 top-k 子空间的比例 |
| $\|\Sigma_k U_k^\top u\|_2 / \|u\|_2$ | 考虑增益后的谱可达性 |
| $\|J^\top u\|_1 / \|u\|_2$ | ℓ∞ 像素预算下的一阶可控性 |

---

## 3. 可行性判断

| 项 | 状态 | 依据 |
|---|---|---|
| $R_\infty$ 计算 | ✅ 已通 | `src/otw/open_phase.py::pgd_open` 已在做 $u^\top h$ 对 δ 的反向传播 |
| $C_l$ 测量 | ✅ 已通 | `src/p0_qwen/model.py::generate_with_patch`(full/sub) |
| 目标方向 | ✅ 冻结 | `outputs/p0_qwen/full/native/p0s_u.json`,L24,`s_mode=+⟨h,u⟩` |
| 图像重建 | ✅ 已通 | `src/n1/preflight.py::_reconstruct_x01` |
| 标签 | ✅ 已通 | `otw.gate.decode_axes` / `n0.axes.four_axes` |
| 算力 | ✅ 可控 | Stage 0 分钟级;R-1 小时级;单卡 4080 SUPER |

**未决风险(只能靠 Stage 0 实测):**
1. 4-bit 下 hidden 梯度的数值精度(open_phase 已用 backtrack 缓解,但 $R_\infty$ 绝对值可能带噪声)。
2. $R_\infty$ 是否真的"低且能区分方向"——这是经验问题。
3. 单图 last-token hidden 的 $J$ 代表性是否足够。

---

## 4. 关键优化(相对原 PoC 方案的改动)

1. **把 R0 的 $R_\infty$ 测量提前为 Stage 0(廉价预筛)。** 反馈的门槛 2("$U_{refusal}$ 的 $R_\infty$ 显著低于随机方向")可以独立于 R-1 先测,而且只需要 forward+backward、分钟级,**不需要跑完整 PGD 轨迹**。如果 Stage 0 就失败(可控性不低、不区分方向),核心假设不成立,直接 NO-GO,省掉 R-1/R0 的全部 GPU。

2. **R-1 的轨迹复用 CR-0 已有 delta 作为共同起点**,不从零随机初始化。CR-0 已有 192 个 delta(40 步 refusal-margin PGD 产物),同一 δ0 出发对比不同方向更干净,与 CR-0 的 pairing 哲学一致。

3. **Stage 0 先 smoke(8 条 = 2 query × 2 carrier × 2 restart)**,确认 $R_\infty$ 测量稳定、分布可区分,再扩大到全量。

4. **符号审计前移到 Stage 0**:$R_\infty(u) = R_\infty(-u)$(对符号不变)是免费 sanity check;同时检查 $+U$ vs $-U$ 的 $C_l$ 符号是否相反(一个推 REF、一个推 ANSWER),把"可控性(非负,只测大小)"与"因果性(带符号,测方向)"在实现层面彻底分开。

---

## 5. 分阶段计划

### Stage 0 — 可控性预筛(廉价,分钟级,先跑)

- 样本:smoke 8 条 → 全量(CR-0 中 12 query × 2 carrier × 若干 restart)。
- 对每条:算 $R_\infty(U_{refusal})$、$R_2$、谱可达性。
- 对照:多个 Gaussian random 方向、多个范数匹配随机正交方向的 $R_\infty$ 分布。
- 输出:分布对比、$U_{refusal}$ 是否显著低于 random、top-k 占比曲线。
- **Go/No-Go**:$U_{refusal}$ 的 $R_\infty$ 显著低于随机方向分布(否则 NO-GO)。

### Stage R-1 — 复现 random > semantic(地基)

- 设置:8–12 query × 2 carrier × 4 restart = 64 条配对轨迹。
- 相同 δ0、相同步数、相同 ε=16/255、相同 pixel-step、相同层(L24)与 token 位置。
- 比较:$U_{refusal}$ / $-U_{refusal}$ / 多个 Gaussian random / 多个范数匹配随机正交 / 现有 Open loss。
- 每步记录:$s_{mode}, \Delta h, \|J^\top u\|_1, u^\top \Delta h, \|\Delta h\|_2$ + 四轴标签。
- 必须控制:符号、raw vs sign gradient 一致、各目标 pixel-gradient norm、normalization 一致、random 是否因尺度占便宜、层/位置与 patch 实验一致。
- **Go/No-Go**:random > semantic 在几十条轨迹后仍成立。

### Stage R0 — 因果性 × 可控性错配图

- 对每个 layer/query/image 算三量:$C_l$(残差 patch 的 $\Delta s_{mode}$)、$R_l$(可控性)、$A_l$(实际 pixel attack 的 $\Delta s_{mode}$)。
- 画 $C_l$–$R_l$–$A_l$ 三变量散点图。
- 核心预测:只有 $C_l R_l$ 都高的层/方向才攻击有效。

---

## 6. Go/No-Go 总门槛(八项至少满足三项才进完整攻击)

1. random > semantic 在 ≥几十条配对轨迹后仍成立;
2. $U_{refusal}$ 的 $R_\infty$ 显著低于随机方向分布;
3. $R_\infty$ 能预测实际 pixel attack 的 $\Delta s_{mode}$(Spearman ρ 显著 + 效应量);
4. $C_l R_l$ 比单独的 $C_l$ 更能预测攻击成功。

攻击论文的进一步门槛(CRAA 必须超过 $U_{refusal}$、random direction、普通 output-loss PGD,相同 ε / backward 次数,且提高实际 ANSWER / core_RHC,并跨 ≥2 个模型)。

---

## 7. 冻结项(不可改)

- 模型:Qwen2-VL-7B-Instruct,native template,4-bit,seed 2026,greedy。
- 目标方向:$U_{refusal}$(p0s_u.json,L24),不 refit。
- ε=16/255,α=1/255(沿用 CR-0)。
- 不读 h91–h130;不重开 L24 rank 搜索;不用 sampling 造 SAFE。
- 不启动 P1(仍锁定)。
