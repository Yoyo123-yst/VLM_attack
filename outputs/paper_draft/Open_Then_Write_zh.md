# 先开后写：解耦拒绝压制与有害内容的视觉越狱
# Open-then-Write: A Two-Phase Visual Jailbreak for Vision–Language Models

> 本稿按**攻击方法论文**写。机制只出现在「为什么这样设计」；贡献句必须是算法、预算分配与 core_RHC 提升。  
> 标 **（P0）/（N0）/（FC）** 为已有证据，用作方法动机；标 **（方法主实验，待跑）** 的才是投稿要交的攻击表。

---

## 创新点（方法，不是分析）

现有视觉越狱（FigStep、HIMRD、首词 PGD、Seeing No Evil）用**一个损失**同时打「别拒绝」和「说出有害内容」。ASR 在模型开口讲安全建议时就已经饱和，优化停在 RELATED_SAFE。我们提出 **先开后写（Open-then-Write, OtW）**：把攻击拆成两个相位、两笔预算——

1. **开相（Mode phase）：** 只用视觉扰动把模型推过拒绝阈值（沿冻结拒绝方向 / 压制安全前缀注意力），直到判定为 ANSWER。  
2. **写相（Write phase）：** 在已经开口的前提下，把剩余扰动预算用于把回答从相关安全推到可执行有害顺从（core_RHC），不允许停在 RELATED_SAFE。

PGD 仍是优化器，**方法是分相目标与分相预算**，不是又发明一种扰动。主指标是 **core_RHC**，不是 ASR。

---

## 摘要

视觉语言模型的越狱攻击通常把「非拒绝」当作成功，并用单一对抗损失同时优化拒绝压制与有害续写。我们发现这一设计存在结构性缺陷：一旦模型开口，即使内容只是相关安全建议，攻击损失也会饱和，从而在 RELATED_SAFE 处形成局部最优。因果证据表明，无目标视觉扰动能够稳定改变的是响应模式（回答/拒绝），而不是回答内部的安全内容。

据此我们提出 **先开后写（OtW）**。开相将 \(\ell_\infty\) 预算的一部分用于沿冻结拒绝残差方向（并可叠加安全前缀注意力压制）迫使模型回答；写相在开相成功之后，用剩余预算优化有害内容目标，并显式惩罚相关安全续写。开相在查询间共享（可做成通用模式扰动），写相按查询自适应，因为内容不能随模式一起迁移。

在同一总预算下，OtW 应以 **核心有害顺从率（core_RHC）** 超过联合 PGD、仅开相、以及 FigStep / HIMRD 类单阶段攻击；ASR 仅作参考，用以说明联合优化如何虚高。方法的可迁移部分是开相，不是有害剧本。

---

## 1. 引言

视觉越狱已经有一套成熟配方：字形写入（FigStep）、图像藏恶意（HADES）、跨模态拆风险加诱导肯定（HIMRD）、打乱文本（SI-Attack）、首词拒绝边距 PGD、注意力压制安全前缀（Seeing No Evil）、指定答句的跨问题注意力劫持。它们大多把 ASR 当目标函数或当唯一报表。

**问题不在于缺少另一种 PGD，而在于单阶段攻击的目标函数与真实有害成功不对齐。** ASR 在「开口劝你别犯罪」时已经满分。联合优化因此不必把内容推到可执行步骤。这与我们在同一模型上的测量一致：残差里可因果推动的是拒绝轴，不是有害/相关安全轴（P0）；同预算重启主要切换开口与拒绝，只有个别问题能同时出现两种回答（FC）。Seeing No Evil 也指出，有害目标与安全检索之间存在梯度冲突。单阶段视觉越狱同时拉这两股力，容易停在「已经开口、内容仍安全」。

本文把上述观察写成攻击算法，而不是停在分析。

**先开后写（OtW）** 将一次 \(\epsilon\)-预算的视觉攻击拆成：

- 开相 \(\delta_{\mathrm{m}}\)：只优化模式，直到 ANSWER 门控通过，或用尽开相预算 \(\epsilon_{\mathrm{m}}\)。  
- 写相 \(\delta_{\mathrm{w}}\)：在 \(x+\delta_{\mathrm{m}}\) 已开口的条件下优化内容，约束 \(\|\delta_{\mathrm{m}}+\delta_{\mathrm{w}}\|_\infty\leq\epsilon\)，损失显式抑制 RELATED_SAFE、奖励 core_RHC 代理。

与 HIMRD 的差别：HIMRD 用诱导提示抬肯定率（仍是模式），风险拆分同时塞内容，仍是单次 ASR。与 Seeing No Evil 的差别：他们用注意力减轻梯度冲突，仍报 ASR；我们从目标函数上禁止在相关安全处停下来。与 Attention Hijacking 的差别：他们指定完整目标答句并跨问题锁定；OtW 的写相是越狱式的有害顺从，不是固定剧本，且开相可跨问题复用。

### 贡献（三项方法创新；放弃共享安全瓶颈）

**C1. 状态约束的两阶段攻击，而不是两次换损失。** 同一 \(\ell_\infty\) 预算下，开相只最大化回答模式分数 \(s_{\mathrm{mode}}\)；写相最小化内容损失，并**硬约束** \(s_{\mathrm{mode}}\geq\tau\)（保持 ANSWER）。写相不得靠重新拒答来降低有害输出，也不得把已打开的口关掉。成功始终是 core_RHC，不是非拒答。这与 \(\mathcal L_{\mathrm{mode}}+\lambda\mathcal L_{\mathrm{content}}\) 的联合 PGD、以及「前 20 步 A、后 20 步 B」的固定课表都不同。

**C2. 用因果响应模式方向做动态相位门控。** 在独立数据上拟合 \(u_{\mathrm{ref}}=\mathrm{normalize}(\mu_{\mathrm{REF}}-\mu_{\mathrm{ANSWER}})\)，其中 ANSWER 含 RELATED_SAFE 与 RHC。\(s_{\mathrm{mode}}=-\langle h_{L24},u_{\mathrm{ref}}\rangle\) 过阈且周期性 greedy 为 ANSWER 时，才从 Open 切到 Write。\(u_{\mathrm{ref}}\) 只负责跨过拒答门，**不是**安全内容方向，**不再**用它直接控制 RHC。P0 已表明晚层残差控制的是模式；旧 \(U_{\mathrm{safety}}\) 不成立，本文不重开该叙事。

**C3. 全回答无目标、查询条件的 Write。** 不指定完整有害答句（避免退化成锁定一篇预写作文），但允许查询条件的行为原型：有害短续写集合 \(P_H(q)\)、相关安全集合 \(P_S(q)\)，外加相关性项 \(\mathcal L_{\mathrm{rel}}\)。最终是否 core_RHC 由独立四轴判定，不由优化目标自证。

---

## 2. 威胁模型

攻击者在推理时修改图像，不改权重、不要求目标有害全文（写相可用短顺从前缀或有害内容代理，而不是整段作案手册作为唯一目标）。官方对话模板。扰动 \(\|\delta\|_\infty\leq\epsilon\)（默认 \(16/255\)）。成功定义为：有害问题上输出 **RHC \(\wedge\) ANSWER \(\wedge\) GROUNDED \(\wedge\) FLUENT**。开口但 RELATED_SAFE 算开相成功、写相失败，**不算攻击成功**。

---

## 3. 方法：先开后写

### 3.1 为什么要分相（设计动机，不是本文贡献）

记 \(h\) 为 last-prompt 残差。P0：沿 \(u_{\mathrm{ref}}\) 能改 ANSWER/REFUSE；沿 \(u_{\mathrm{safety}}\) 不能稳定改 RHC。FC：同 \(\epsilon\) 重启大量停在拒绝或 RELATED_SAFE。联合损失

\[
\mathcal{L}_{\mathrm{joint}}=\mathcal{L}_{\mathrm{refuse}}+\lambda\mathcal{L}_{\mathrm{harm}}
\]

在 \(\mathcal{L}_{\mathrm{refuse}}\) 已小（模型开口）时，即使 \(\mathcal{L}_{\mathrm{harm}}\) 仍大（内容是安全建议），ASR 已饱和。OtW 把门控从 ASR 换成 ANSWER，并把 \(\mathcal{L}_{\mathrm{harm}}\) 留到开口之后，避免写相梯度被拒绝项淹没。

### 3.2 开相（Mode phase）

输入干净图 \(x_0\)、有害问题 \(q\)。优化 \(\delta_{\mathrm{m}}\)：

\[
\max_{\|\delta_{\mathrm{m}}\|_\infty\leq\epsilon_{\mathrm{m}}}\;
\langle h(x_0+\delta_{\mathrm{m}},q),\; -u_{\mathrm{ref}}\rangle
\;+\;
\gamma\,\mathcal{L}_{\mathrm{attn}}
\]

其中 \(u_{\mathrm{ref}}\) 为冻结 L24 拒绝方向（不重拟合、不改 rank）。\(\mathcal{L}_{\mathrm{attn}}\) 可选：压制系统/安全前缀 token 的注意力（Seeing No Evil 的把手，这里是开相的辅助项，不是本文唯一卖点）。

**门控：** 贪心解码后响应模式为 ANSWER 则进入写相；否则在 \(\epsilon_{\mathrm{m}}\) 内继续。开相**禁止**把 RELATED_SAFE 当成功。

**通用开相（推荐作为方法的第二块）：** 在一组训练问题 \(\{q_i\}\) 上优化共享 \(\delta_{\mathrm{m}}^{\star}\)，目标为平均开口率，而不是平均 ASR。测试时直接加 \(\delta_{\mathrm{m}}^{\star}\)。

### 3.3 写相（Write phase）

在 \(\tilde x=x_0+\delta_{\mathrm{m}}\) 已开口的条件下优化 \(\delta_{\mathrm{w}}\)：

\[
\min_{\|\delta_{\mathrm{m}}+\delta_{\mathrm{w}}\|_\infty\leq\epsilon}
\;
\mathcal{L}_{\mathrm{harm}}(\tilde x+\delta_{\mathrm{w}},q)
\;-\;
\alpha\,\mathcal{L}_{\mathrm{safelec}}(\tilde x+\delta_{\mathrm{w}},q)
\]

\(\mathcal{L}_{\mathrm{harm}}\)：有害顺从代理（例如操作性续写的首词/短前缀边距，或与「步骤/材料/话术」类内容相关的 token 边距——实现时用固定、与查询主题相关的短前缀，而不是整篇有害范文）。  
\(\mathcal{L}_{\mathrm{safelec}}\)：相关安全讲座的代理（惩罚「以下是合法替代 / 我必须提醒你这违法但……」类续写）。  
写相在拒绝方向上使用**止步或弱约束**，避免把已经打开的口重新关上（解决 Seeing No Evil 所说的梯度冲突）。

### 3.4 预算分配

\[
\epsilon_{\mathrm{m}}+\epsilon_{\mathrm{w}}=\epsilon,\quad
\epsilon_{\mathrm{m}}=\rho\epsilon,\ \rho\in\{0.3,0.5,0.7\}
\]

消融 \(\rho\)：\(\rho=1\) 退化为仅开相（高 ASR、低 core_RHC）；\(\rho=0\) 退化为仅写相（拒绝仍在，内容损失难优化）；中间 \(\rho\) 应给出最高 core_RHC。这一消融是方法论文的主图，不是机制附录。

### 3.5 算法（投稿伪代码）

```
OtW(x0, q, ε, ρ, u_ref):
    ε_m, ε_w ← ρ*ε, (1-ρ)*ε
    δ_m ← 0
    for t in ModeSteps:
        δ_m ← PGD_step(δ_m, L_mode(x0+δ_m, q, u_ref), ε_m)
        if greedy_mode(x0+δ_m, q) == ANSWER: break
    δ_w ← 0
    if greedy_mode(x0+δ_m, q) != ANSWER:
        return x0+δ_m          # 开相失败，记为攻击失败
    for t in WriteSteps:
        δ_w ← PGD_step(δ_w, L_write(x0+δ_m+δ_w, q), 约束到总 ε)
    return clip(x0+δ_m+δ_w)
```

训练共享开相时，外层对 \(q_i\) 平均 \(L_{\mathrm{mode}}\)，得到 \(\delta_{\mathrm{m}}^{\star}\)，测试只跑写相。

---

## 4. 实验必须交出的攻击表（不是分析表）

| 实验 | 要赢的数字 | 基线 |
|---|---|---|
| 同 \(\epsilon\) 下 core_RHC | OtW > 联合 PGD、仅开相、仅写相 | 三路消融是方法成立的最低条 |
| ASR vs core_RHC | 联合 PGD / HIMRD 式：ASR 高、core_RHC 低；OtW：core_RHC 升、RELATED_SAFE 降 | 证明不是刷 ASR |
| 预算比 \(\rho\) | 中间 \(\rho\) 最优 | 分相不是空话 |
| 共享开相迁移 | \(\delta_{\mathrm{m}}^{\star}\) 在未见问题上抬开口率；写相仍按问题 | 对 UltraBreak/C-PGC：我们迁移的是模式位 |
| 有目标对照 | 指定全文的注意力劫持可锁内容；OtW 不指定全文，core_RHC 按问题变化 | 划清与 Attention Hijacking 的方法边界 |
| 良性误开 | 开相不应把无害问题全部打开 | 特异性 |

问题分层（h01 类易 RHC、h11 类易 RELATED_SAFE、h41 类死拒）用来解释写相必要性：h11 上仅开相失败、OtW 应拉开 core_RHC。这是攻击结果的解释，不是论文主旨。

---

## 5. 与顶会攻击论文怎么错开

| 论文 | 他们的方法 | OtW 多出来的方法件 |
|---|---|---|
| FigStep / HADES | 把有害语义塞进图 | 先保证开口，再写内容；不靠字形当唯一手段 |
| HIMRD | 拆风险 + 诱导肯定 | 诱导肯定只相当于开相；我们加写相与 RELATED_SAFE 惩罚 |
| 首词 PGD / UltraBreak | 单阶段视觉梯度 | 分相 + 开口门控，避免 ASR 局部最优 |
| Seeing No Evil | 注意力压制安全前缀 | 把该把手放进开相，写相专门打内容 |
| Attention Hijacking | 指定答句、跨 query 锁内容 | 不指定全文；开相共享、写相按查询 |
| IAG / ImgTrojan | 后门 / 毒图 | 本稿是测试时攻击；后门可作为后续把开相做成触发器 |

---

## 6. 局限（攻击论文也要写）

写相的有害代理若做成完整目标文本，会退化成有目标攻击，必须在实现里限制为短前缀/类别代理。开相依赖已冻结 \(u_{\mathrm{ref}}\)，换模型要重估方向，但不重开旧 rank 搜索当贡献。单模型 7B 先把 core_RHC 表做实，再谈迁移。

---

## 7. 投稿定位句

单阶段视觉越狱把拒绝和内容捆在同一损失里，于是停在开口的安全回答上。先开后写把预算拆成关拒绝与写有害两相，用开口门控切换，以核心有害顺从为成功，而不是以 ASR 为成功。
