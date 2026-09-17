# V-CachePoll 阶段汇报：问题、方法与结果

- 日期：2026-09-16
- 用途：向导师汇报初步方法是否可行，作为申请计算/存储扩容的依据
- 当前判定：**方法论证阶段可行（CONTINUE）**；尚非论文主表
- 代码与报告：`V-CachePoll/`，阶段原文见 `out/P0_REPORT.md`–`out/P3_REPORT.md`

---

## 0. 一页结论

多图视觉语言模型为了效率，会按各图「重要性」把有限视觉 token 槽位分给不同图片，但通常**没有来源隔离**。我们提出威胁模型 **Visual Token Squatting / 视觉缓存污染**：攻击者只修改低可信附图 B，抬高无任务价值的 token 重要性，驱逐未修改主图 A 中的证据。错误应只在共享压缩打开时出现。

攻击方法 **V-CachePoll** 由两层构成：图像级配额投毒 + token 级定向驱逐集合。有限尺度搜索只用于触发真实的离散 token 交换，不作为独立创新。

在本机 **Qwen2-VL-7B-Instruct + COCO 替身** 上，P0–P3 已跑通：

- 只改 B 即可改变 A 的配额和 survivor set；全 token 答案保持正确。
- 紧预算下出现**压缩专属失败**：32 条中 7 条（21.9%）；隔离配额 / 恢复 A 的 clean survivor 可挽回这 7 条。
- 同预算下高于 Random、Task-PGD、CAA、CAGE、Rank；其中 Task-PGD 为 0/32，Rank 为 5/32。后来又加 17 条，合计 12/49 vs Rank 9/49。

因此：**机制存在、攻击可复现、失败模式与威胁模型一致。** 但数据是 COCO 替身、模型不是 AVTP 原文的 Qwen3-VL，相对最近基线 Rank 的优势偏薄。下一阶段需要 TextVQA / ChartQA 与新架构，当前磁盘不足，需扩容后再做论文级实验。

---

## 1. 问题

### 1.1 AVTP 是什么

**AVTP（Adaptive Visual Token Pruning，自适应视觉 token 剪枝）** 是一篇效率论文，不是安全论文。完整题目是 *Multi-Image Visual Token Pruning in Large Visual Language Models*（arXiv:2608.26806）。它要解决的是：多图 VLM 的视觉前缀太长，推理又慢又占显存。

相对 FastV、VisionZip 这类方法，AVTP 强调三件事：

1. **不用 attention 打分，因而能和 FlashAttention 一起用。** FastV 依赖 LLM 浅层 attention 对 visual token 排序；存 attention 矩阵在多图长序列上很容易 OOM，也和 FlashAttention 冲突。AVTP 改用 visual token 在相邻剪枝层之间的 hidden-state 变化当重要性：变化大的 token 被认为更重要。原文写 \(I_{v}=1-\mathrm{Sim}(h^{l}_{v}, h^{\mathrm{prior}}_{v})\)。
2. **按模型选剪枝层。** LLaVA 系视觉信息偏浅层，Qwen3-VL / InternVL 偏中后层。Qwen3-VL-8B 上他们用的层是 **1 / 14 / 19**。
3. **多图时不要每张图剪一样多。** 先对每张图的 token 重要性取平均 \(\bar I_i\)，再按相对重要性分配保留率（keep ratio）：

\[
r_i = r_{\mathrm{base}} + \alpha(\bar I_i - \bar I_{\mathrm{avg}})
\]

\(r_{\mathrm{base}}\) 是全局基础保留率（主实验常用 50%），\(\alpha\) 控制「重要图多留、不重要图少留」的幅度。然后在每张图内部再做 Top-K。

他们在 Qwen3-VL-8B 上报告：保留约 50% 视觉 token，推理约 2×，多图基准仍有原精度的 96.1%；InternVL3.5-8B、LLaVA-OV-7B 上也做了同样设定。AVTP **默认所有图片都诚实**，把 \(\bar I_i\) 当成任务相关性的代理，没有讨论这套报价能不能被不可信附图操纵。我们把它当成**受害系统**，不是当成攻击方法去改进。

本仓库是 **AVTP 式复现**：配额公式、Qwen 剪枝层 1/14/19、训练免费剪枝这三点对齐原文；模型暂用本地 Qwen2-VL-7B。重要性在实现上用层间 hidden-state 变化幅度，与原文的 \(1-\mathrm{Sim}\) 同属 variation 打分，尚未与官方代码逐项对齐。

### 1.2 现象：AVTP 如何把多图预算做成「此消彼长」

多图输入时，视觉 token 数量随图片张数近似线性增长，系统必须把视觉前缀压到固定预算 \(K\)。AVTP 的做法不是全局混在一起排 Top-K，而是**先按图分配额，再图内 Top-K**。

对上面的 \(r_i\) 在未裁剪时求和：

\[
\sum_i r_i = N r_{\mathrm{base}} + \alpha\sum_i(\bar I_i-\bar I_{\mathrm{avg}}) = N r_{\mathrm{base}}
\]

因为 \(\sum_i(\bar I_i-\bar I_{\mathrm{avg}})=0\)。因此**一张图的配额上升，必然由其他图的配额下降来补偿**。这是公式的代数推论，不是比喻。我们实现里对 \(r_i\) 做上下裁剪后还会再归一化，把总和拉回 \(N r_{\mathrm{base}}\)；CPU 测试 `test_raising_b_cuts_a` 验证了提高 B 的平均重要性会切开 A 的配额。

重要性 \(\bar I_i\) 来自该图自己的 hidden-state 变化，等于让**图片提供者参与资源报价**。效率论文把它写成「更相关的图多留 token」；对抗环境下，「更相关」只是这张图自己报出来的分数。

### 1.3 1.1–1.2 哪些有论文、哪些是类比、哪些是我们的实验

向老师讲这两段时，不要混成「已有论文已经证明了视觉缓存污染」。应分成三层：

**（1）效率论文里的事实（可引用 AVTP / FastV）**

- 多图视觉前缀过长、需要剪到预算 \(K\)：FastV、VisionZip、AVTP 等效率工作的共同前提。
- 按图平均重要性分配 \(r_i\)、重要图多留、不重要图少留：AVTP §3.3.3 的设定。
- \(\sum r_i = N r_{\mathrm{base}}\)：由该公式直接推出。
- 分数来自图片自身的 hidden-state 变化、不依赖 query attention：AVTP §3.3.1。

**（2）系统安全里的类比（有出处，但不是 VLM 论文的现成结论）**

「这与共享缓存中的 cache pollution 同构」是**分析框架**，不是 AVTP 或 CAA 已经证明过的命题。共享缓存污染、可构造 eviction set、用随机化/分区降低可驱逐性，是体系结构与系统安全里的成熟机制（LLC Prime+Probe、cache occupancy、ScatterCache 一类工作）。我们借用的是结构：共享容量、可预测选择、跨安全域。  
**不能说成：**「已有论文证明了多图 VLM 就是 cache pollution。」那是本文要检验的威胁，不是别人的结论。

**（3）我们自己的实验（这才是安全缺口能落地的部分）**

| 表述 | 依据 | 强度 |
|---|---|---|
| 不同来源进同一资源池 | 双图 pack 后按 owner 分配额 / Top-K | 实现事实 |
| 报价由输入自己产生 | 只改 B 就能改 \(\bar I_B\) 和 \(r_B\) | P0 / P1 |
| 一图升配额另一图降 | 公式 + 守恒测试；P1 均 \(\lvert\Delta r_A\rvert=0.056\) | 强 |
| 可驱逐对方 token | P0 随机噪声 A-out 37%；P1 优化后 A-out 100%、约 7.3 个 token | 强 |
| 不必改 A 的像素 | 协议只扰动 B | 强 |
| 不是普通跨图语义污染 | P0/P1 全 token 准确率 100% | 强 |
| 占槽导致**任务**失败 | P2 才有 7/32 压缩专属失败 | 初步；COCO 替身偏弱 |

「网页配图 / 用户附件 / 多轮早期图」是威胁模型假设，合理，但还不是对某个上线产品的部署测量。当前实验是构造的「主图 A + 附图 B」协议。

### 1.4 安全缺口

效率文献默认所有图片都诚实。对抗环境下这不成立：攻击者可以控制其中一张低可信附图（网页配图、用户上传附件、多轮对话里的早期图片），而不必修改可信主图（文档、证件、题干图）。

缺口可以表述为（其中「同构」按 1.3 理解为分析框架，不是已有 VLM 结论）：

> 不同来源的视觉 token 被放进同一个有限资源池，映射可预测，且报价由输入自己产生。这与共享缓存中的 cache pollution 同构：攻击者装入大量「重要性高、任务价值低」的 cache line，驱逐受害者真正需要的数据。

映射关系：

| 共享缓存 | 多图视觉压缩 |
|---|---|
| cache line | visual token |
| cache 容量 / set | Top-K 保留集或每图配额 |
| attacker line | B 中攻击者控制的 token |
| victim line | A 中承载关键证据的 token |
| eviction set | 能越过阈值的一组 B tokens |
| cache partitioning | 每来源最低配额或隔离 Top-K |

### 1.5 与已有攻击的区别（必须切开，否则没有新问题）

| 工作 | 他们在做什么 | 我们不是这个 |
|---|---|---|
| AVTP | 自适应分配 keep ratio，提高效率 | 他们不研究报价能否被操纵 |
| CAA | 攻击**同一张**图，让本图关键 token 掉出 Top-K | 我们要求 A 像素不被修改 |
| CAGE | 让受扰 token 在未知预算下进入 survivor set | survivor 存活 ≠ 驱逐另一来源的证据 |
| LAMP | 少量污染图经 attention 影响 clean tokens | 无压缩也应错；我们必须证明无压缩 / 隔离后恢复 |
| DMN / MLAI | 多图承载攻击语义或越狱 | 我们不需要 B 含恶意文字，也不优化目标回答 |

因此论文问题不是「再提高越狱 ASR」，而是：

> 一张不可修改的可信图 A，会不会因为附图 B 的分数被人为抬高，而在共享压缩后丢失任务证据？

### 1.6 三条可检验命题

**Claim 1** 共享自适应压缩缺少来源隔离。  
固定 A，只改 B，应看到 \(r_B\uparrow\)、\(r_A\downarrow\)，以及 A 的 survivor 被替换。

**Claim 2** 可以构造跨图 visual eviction set。  
V-CachePoll 应比 Random、普通任务 PGD、CAA-B-only 更有效地驱逐 A 的关键 token，同时全 token 推理仍正确。

**Claim 3** 错误由资源驱逐引起，不是一般跨图语义干扰。  
无压缩、每图独立配额、把 A 的 clean survivor 塞回这三项干预，应能显著恢复任务。若不恢复，则更像 LAMP，不能叫缓存污染。

成功样本的操作定义（全程冻结）：

\[
\mathrm{Acc}_{full}(A,B') \approx \mathrm{Acc}_{full}(A,B),\qquad
\mathrm{Acc}_{comp}(A,B') < \mathrm{Acc}_{comp}(A,B)
\]

即 **compression-only failure / 压缩专属失败**。

---

## 2. 方法

### 2.1 威胁模型

- 文本问题 \(q\) 固定，且只询问第一张图 A。
- 攻击者不能改 A 的像素，只能改 B：\(B'=\mathrm{clip}(B+\delta)\)，\(\|\delta\|_\infty\le \epsilon\)。
- 主表 \(\epsilon=16/255\)，步长 \(1/255\)，40 步。
- 压缩器在 A+B 的联合视觉前缀上分配槽位；未选中的 visual token 从序列中删除后再调用官方 `generate`。
- 不指定目标回答，不使用越狱前缀。

### 2.2 受害压缩器

实现 AVTP 式两级选择（AVTP 本身见 §1.1）：

1. 用 LLM 层 1 / 14 / 19 的 hidden-state variation 得到每个 visual token 的重要性；
2. 按 AVTP 公式计算每图配额 \(r_i\)，再在图内 Top-K。

对照压缩器：全局跨图 Top-K（无每图配额）、每图独立 \(r_i=r_{\mathrm{base}}\)（隔离防御）、无压缩。

预算两档：\(r_{\mathrm{base}}=0.5\)（对齐 AVTP 论文）用于 P0/P1 机制；\(r_{\mathrm{base}}=0.2\)（对齐 CAA，槽更紧）用于 P2/P3 任务失败。

### 2.3 V-CachePoll：双层攻击

**（1）图像级配额投毒。** 让 B 获得不合理的图片级预算：

\[
L_{\mathrm{quota}} = r_A - r_B
\]

最小化后 \(r_B\) 上升、\(r_A\) 下降。同时用价值分离项约束「B 并没有真正变得更相关」：

\[
L_{\mathrm{value}} = D\big(f_{\mathrm{full}}(A,B',q),\, f_{\mathrm{full}}(A,B,q)\big)
\]

全 token 表示 / 输出应尽量不变，从而排除 LAMP 式语义污染。

**（2）token 级驱逐集合。** 仅抢配额可能只挤掉 A 的背景 token。设 \(U\) 为 A 中干净压缩下、删掉后会改答案的关键 token，\(V\) 为 B 中靠近阈值的候选。交换间隙

\[
g_{u,v} = s_u^A - s_v^B
\]

\(g_{u,v}<0\) 表示 B 的 \(v\) 排到 A 的 \(u\) 前面。目标是构造最小驱逐集合 \(E_B\)，而不是翻转 B 的全部低分 token：

\[
L_{\mathrm{evict}} = \sum_{u\in U}\operatorname{softmin}_{v\in V}\operatorname{softplus}(g_{u,v}+\kappa)
\]

另有阈值项 \(L_{\mathrm{crit}}\)：把 \(U\) 压到 A 图内 keep cutoff 以下。P3 显示，相对 Rank 多出来的任务失败主要来自这一项，而不是搜索。

**（3）精确压缩器上的交换反馈（实现细节，非贡献）。** Top-K 是离散事件，平滑 loss 下降不一定改变 survivor。每若干步用精确压缩器评估候选尺度，只接受真实产生 B-in / A-out 的更新。实验中每 5 步搜索一次；每步都搜（Armijo）并不更好。

总体损失：

\[
L = \lambda_q L_{\mathrm{quota}} + \lambda_e L_{\mathrm{evict}} + \lambda_c L_{\mathrm{crit}} + \lambda_v L_{\mathrm{value}} + \lambda_p L_{\mathrm{perc}}
\]

职责划分：\(L_{\mathrm{quota}}\) 抢图级预算，\(L_{\mathrm{evict}}/L_{\mathrm{crit}}\) 把预算对准 A 的关键证据，\(L_{\mathrm{value}}\) 隔离一般语义干扰，\(L_{\mathrm{perc}}\)（TV）限制扰动可见性。

### 2.4 关键 token \(U\) 如何得到

不手标。对干净 AVTP 的 A survivor 做 drop / leave-one-out：删掉后答案改变的进入 \(U\)。COCO 上发现丢掉**低分** A token 比丢掉高分 token 更容易改答案（drop-bot 7 vs drop-top 1），因此配额驱逐（先挤掉排名尾部）与任务失败是对齐的。

### 2.5 同协议基线（证明不是换个 PGD 名字）

全部只扰动 B、相同 \(\epsilon\) 与步数：

| 基线 | 目的 |
|---|---|
| Random（40 次随机重启，用精确压缩器 J 打分） | 排除自然排名波动 |
| Task-PGD（降低 gold 的 CE） | 证明普通输出攻击打不出该失败模式 |
| CAA-B-only | 证明跨图驱逐目标超过把单图方法搬到 B |
| CAGE-B-only | 证明 survivor disruption ≠ victim eviction |
| Rank-PGD（quota+evict，无交换搜索） | 判断搜索是否必要 |
| Armijo（Rank + 每步搜索） | 排除收益来自普通步长调整 |

V-CachePoll 本体即 P2 的完整攻击（quota + evict + crit + 每 5 步交换搜索）。

### 2.6 明确不做

新越狱 loss；把有限尺度搜索写成独立创新；NLVR2 / Counting 等对称多图当主 A/B；在 LLaVA-1.5 上继续加步数硬打任务失败。

---

## 3. 实验设置（当前，替身级）

| 项 | 实际 | 相对论文原方案 |
|---|---|---|
| 模型 | 本地 Qwen2-VL-7B-Instruct | AVTP 原文为 Qwen3-VL-8B，替身已记录 |
| 数据 | coco300 VQA 作 A，其余 COCO val2017 作 B | 计划主数据为 TextVQA / ChartQA，磁盘不足未上盘 |
| 问题 | 只问第一张图，忽略第二张 | 符合 A/B 分离 |
| 压缩 | AVTP 式共享配额 + 图内 Top-K | 自实现，层 1/14/19 |
| 筛选 | clean-correct：A-only 有内容，且 full-token 与 AVTP 均与 A-only 一致 | 与方案一致 |
| 环境 | 单卡 RTX 4080 SUPER 32GB，HF offline | — |

P0 筛 40 对，保留 35 条。P2 在 \(r_{\mathrm{base}}=0.2\) 下再筛，保留 32 条。之后又加 c040–c059 共 20 对，P2 筛后 17 条，用于小幅加量，不覆盖原 32 条结果。

---

## 4. 结果

### 4.1 P0：机制探针（随机噪声，35 条）

仅对 B 加 \(\epsilon=8/255\) 随机噪声，不训练攻击。

| 指标 | 值 |
|---|---|
| 筛选保留 / 剔除 | 35 / 5 |
| 均 \(\lvert\Delta r_A\rvert\) | 0.0035（随机几乎推不动配额） |
| AVTP 出现 A-out | 37.1% |
| 同时 B-in / A-out | 37.1%（全局 Top-K 为 51.4%） |
| 全 token 仍对 | 100% |
| 压缩专属失败 | 0% |
| 有 A-out 时，隔离配额 / restore-A 恢复 | 1.0 / 1.0 |

**读法：** AVTP **不是**每图独立 Top-K。B 的改变可以动未修改 A 的 survivor，且不是跨图语义污染。但随机噪声不够当攻击，任务尚未破坏。判定 CONTINUE。

### 4.2 P1：配额投毒（\(r_{\mathrm{base}}\approx 0.5\)，35 条）

优化 B，损失为 quota + evict + value + TV，40 步，每 5 步交换搜索。

| 指标 | 值 |
|---|---|
| 均 \(\lvert\Delta r_A\rvert\) | 0.056（约为 P0 随机的 16 倍） |
| AVTP A-out | 100%，均约 7.3 个 token |
| 全 token 仍对 | 100% |
| 压缩专属失败 | **0%** |

**读法：** 槽位能稳定抢走，Claim 1 在优化攻击下成立。COCO VQA 丢掉约 7 个 A token 后仍能答对，需要更紧预算和真正改答案的 \(U\)。这正是方案里「探针成功、还不是论文表」的状态。

### 4.3 P2：紧预算 + 关键 token（主攻击，32 条）

\(r_{\mathrm{base}}=0.2\)。先 leave-one-out / drop 得到 \(U\)，再跑完整 V-CachePoll。

| 指标 | 值 |
|---|---|
| 完成 / 出错 | 32 / 0 |
| 均 A-out / U-out | 7.41 / 5.59 |
| U 驱逐率 | 0.826 |
| 全 token 仍对 | 0.969（31/32） |
| **压缩专属失败** | **0.219（7/32）** |
| 隔离配额仍对 | 0.969 |
| restore-A 仍对 | 1.000 |
| restore 挽回那 7 个失败 | **7/7** |

**读法：** 第一次出现方案所定义的成功模式。4/7 个压缩专属失败使用的是 drop-bot \(U\)，与「配额先挤掉低分 A token」一致。Claim 3 在替身上成立：错误跟着共享槽位走。判定 CONTINUE。V-CachePoll 主结果即此次 run，P3 不再重跑它。

### 4.4 P3：同协议基线（32 条，主对照表）

同一 32 对、同一 \(\epsilon\) 与 40 步。`vcache` = P2。

| 方法 | 全 token 仍对 | 压缩专属失败 | 均 A-out | 均 U-out | U 驱逐率 |
|---|---|---|---|---|---|
| Random | 1.000 | 2/32 (0.062) | 0.91 | 0.91 | 0.138 |
| Task-PGD | 0.969 | **0/32 (0.000)** | 0.47 | 0.47 | 0.069 |
| CAA-B-only | 1.000 | 2/32 (0.062) | 3.09 | 2.91 | 0.430 |
| CAGE-B-only | 0.969 | 4/32 (0.125) | 3.69 | 3.47 | 0.518 |
| Rank-PGD | 0.969 | 5/32 (0.156) | 6.69 | 5.66 | **0.833** |
| Armijo | 0.969 | 5/32 (0.156) | 5.94 | 4.97 | 0.733 |
| **V-CachePoll** | 0.969 | **7/32 (0.219)** | 7.41 | 5.59 | 0.826 |

要点：

1. 普通输出攻击**打不出**这个失败模式（Task-PGD 0/32）。
2. CAA 能挪排名，定向驱逐不够。
3. CAGE 证明「攻击 token 活下来 ≠ 把 A 的关键证据挤掉」（U-evict 0.52，低于本方法）。
4. Rank 的 U 驱逐率已与 V-CachePoll 持平。多 2 个任务失败来自 \(L_{\mathrm{crit}}\)，**不是**步长搜索。
5. 每步交换搜索（Armijo）没有更好。有限尺度模块可以写进实现，不能写进贡献列表。

### 4.5 小幅加量（+17 条，合计 49）

协议不变，仅新增 c040–c059，筛后 17 条；新批次只跑 V-CachePoll 与 Rank。

| 集合 | n | V-CachePoll | Rank-PGD | 全 token |
|---|---|---|---|---|
| 原 32 | 32 | 7/32 (21.9%) | 5/32 (15.6%) | 0.969 |
| 新 17 | 17 | 5/17 (29.4%) | 4/17 (23.5%) | 1.000 |
| 合计 | 49 | **12/49 (24.5%)** | **9/49 (18.4%)** | 0.980 |

方向与 32 条一致，优势仍约 6 个百分点。新 17 条中失败样本 restore-A 仍全部挽回。说明不是偶然抽到 32 条，但也说明 **COCO 上再加样本不太可能把方法优势拉成论文级碾压**。

### 4.6 附加迁移（非主表）

**FastV**（8 条，非攻击）：同样总 K 下，last-token attention Top-K 几乎全留给被问的图 A（约 48 A / 4 B）。FastV 与 AVTP 答案均正确。Query-attention 天生偏任务图，**共享配额 AVTP 才是更自然的污染面**。

**LLaVA-1.5-7B**：双图可 pack。6/8 在 \(r_{\mathrm{base}}=0.2\) 下 AVTP clean-correct。quota+evict 40 步：均 A-out 13，U-evict 0.48，全 token 6/6，**压缩专属失败 0/6**。机制能迁（能抢槽），任务失败迁不过 576-token CLIP 网格。下一步迁移应是 InternVL / LLaVA-OneVision，不要在 1.5 上加 loss。

### 4.7 三条命题的当前状态

| Claim | 替身证据 | 状态 |
|---|---|---|
| 1 共享压缩缺来源隔离 | P0/P1：固定 A 只改 B，\(r_A\) 下降、A survivor 被换，full-token 仍对 | **成立** |
| 2 可构造跨图 eviction set | P3：明显优于 Random/Task/CAA/CAGE；相对 Rank 主要赢在任务失败而非 raw U-evict | **弱成立** |
| 3 错误来自资源驱逐 | P2：无压缩 / 隔离配额 / restore-A 恢复那 7 个失败 | **成立** |

---

## 5. 局限（汇报时主动说明）

1. **数据是 COCO VQA 替身**，不是方案中的 TextVQA / ChartQA。COCO 物体问答在丢失约 7 个 token 后仍常能答对，会压低压缩专属 ASR。
2. **模型是 Qwen2-VL-7B**，不是 AVTP 使用的 Qwen3-VL-8B；压缩器为自实现的 AVTP 式配额，尚未对接官方代码逐项核对守恒。
3. **相对 Rank 的优势偏薄**（7 vs 5，合计 12 vs 9）。若主数据上仍打平，论文应改成「新威胁 + 因果分析 + 隔离防御」，不宜硬卖方法碾压。
4. **样本只有几十条**，只用于机制与同协议基线，不能做显著性结论。
5. LAMP 实现、DivPrune、MuirBench、InternVL / LLaVA-OV 尚未做。
6. 本机 `/root/autodl-tmp` 约 50GB，剩余约 5.2GB，无法同时放置 TextVQA 图像与新的 8B 模型。

---

## 6. 下一步与资源需求

协议、loss、筛选标准保持不变。扩容后按下列顺序，每阶段有 GO/STOP：

1. **TextVQA 约 30–40 条探针**（仍先用 Qwen2-VL）：A 为 OCR 图，B 为无关附图。只对比 Rank 与 Task-PGD。  
   - GO：压缩专属失败明显高于 COCO，且明显超过 Rank；full-token 仍高；restore-A 仍能挽回。  
   - STOP：只能抢槽、破不了任务，或无压缩也错。
2. GO 之后做 TextVQA 百级主表 + 全基线 + Claim 3 五格。
3. ChartQA「文档 + 附件」，再迁一个新架构（Qwen3-VL 或 InternVL）。
4. 补 LAMP 对照与隔离配额防御。LLaVA-1.5 只留附录。

**申请扩容的原因：** 不是方法还没写完，而是替身实验已经给出 CONTINUE，主协议数据和图模型目前放不进去。建议数据盘 ≥100GB。

---

## 7. 建议给老师的口头判断

方法作为**新安全问题**是成立的，作为**可复现的小样本攻击**也已经跑通；作为 **CCF-A 方法论文** 还差主数据、第二架构，以及相对 Rank 是否拉开。现在最合理的决策是：认可初步可行性，批准扩容，用 TextVQA 小探针决定是否把「配额投毒 + 驱逐集合」继续当作核心方法贡献。
