# V-CachePoll 进展纪要

更新日期：2026-09-17  
当前判定：**CONTINUE**（P0–P3 替身过关；过夜 P4 全量 32 条 ASR 25.0%，相对 P2 的 21.9% 未崩。P5–P7 仅 smoke。P10/P11 缺权重/图。）

本文汇总截至今日上午的方向、实验阶段、数字和下一步。阶段报告原文仍以 `out/P0_REPORT.md`、`out/P1_REPORT.md`、`out/P2_REPORT.md`、`out/P3_REPORT.md` 为准。

---

## 1. 主创新（已冻结）

题目暂定：**Visual Token Squatting: Cache-Pollution Attacks on Multi-Image Pruning**  
攻击方法名：**V-CachePoll**

核心问题不是再加一种视觉越狱 loss，也不是把有限尺度搜索写成论文：

> 多图视觉 token 压缩器把不同来源的图片放进同一个有限资源池，却没有来源隔离。攻击者只需控制低可信附图 B，制造“重要性高、任务价值低”的 token，像缓存污染一样占据视觉槽位，驱逐未修改主图 A 中的必要证据。错误只在共享压缩打开时出现。

三条必须守住的贡献：

1. **新设置**：攻击来源 B 与受害来源 A 分离；A 的像素不被修改。
2. **新机制**：可操纵的重要性分数引发跨图资源驱逐。
3. **新方法**：图像级配额投毒 + token 级驱逐集合。有限尺度搜索只是离散 token 交换的内部算法，不承担新颖性。

明确不做：新越狱 loss；把 FiRC / 有限尺度搜索当独立创新；以 NLVR2 / Counting 这类对称多图当主 A/B。

---

## 2. 实验设置（当前）

| 项 | 现状 |
|---|---|
| 数据 | COCO val2017 + coco300 VQA **替身**。A 来自 coco300，B 为未用作 A 的另一张 val2017 图。问题只问第一张图。 |
| 主协议数据 | TextVQA / ChartQA **尚未上盘**，按计划等磁盘后再换。COCO 只作探针。 |
| 主模型 | 本地 `Qwen2-VL-7B-Instruct`（AVTP 原文是 Qwen3-VL；替身已记录） |
| 压缩器 | AVTP 式共享配额 \(r_i=r_{base}+\alpha(\bar I_i-\bar I_{avg})\)，再图内 Top-K。重要性来自 LLM 层 1/14/19 的 hidden-state variation。 |
| 攻击协议（P2 起） | \(r_{base}=0.2\)，\(\epsilon=16/255\)，步长 \(1/255\)，40 步 signed PGD，只扰动 B |
| 成功定义 | **压缩专属失败**：全 token 仍对，共享 AVTP 后答错。对照：独立配额、restore-A survivor。 |

环境：conda `vattack`，HF offline，单卡 RTX 4080 SUPER 32GB。

---

## 3. 阶段总览

```text
P0 机制探针     CONTINUE  噪声即可改变 A 的配额/survivor
P1 配额投毒     CONTINUE  优化 B 稳定抢槽，但任务还不破
P2 紧预算+U    CONTINUE  7/32 压缩专属失败；restore-A / 隔离配额可恢复
P3 同协议基线   CONTINUE  V-CachePoll 高于 Random/Task/CAA/CAGE/Rank/Armijo
FastV / LLaVA  附加       FastV 不是好污染面；LLaVA-1.5 机制能迁、任务失败迁不过
```

---

## 4. P0：共享压缩缺来源隔离

35 条 clean-correct。仅对 B 加 \(\epsilon=8/255\) 随机噪声。

| 指标 | 值 |
|---|---|
| 筛选保留 / 剔除 | 35 / 5 |
| 均 \(\lvert\Delta r_A\rvert\) | 0.0035（随机几乎推不动配额） |
| AVTP 出现 A-out | 37.1% |
| 全 token 仍对 | 100% |
| 压缩专属失败 | 0% |
| 有 A-out 时，隔离配额 / restore-A 恢复 | 1.0 / 1.0 |

结论：AVTP **不是**每图独立 Top-K。B 的改变可以动未修改 A 的 survivor。随机噪声不够当攻击。

---

## 5. P1：配额投毒 + 驱逐集合

35 条，\(r_{base}=0.5\) 量级（P1 配置），\(\epsilon=16/255\)，40 步，quota + evict + value + TV，每 5 步精确压缩器交换搜索。

| 指标 | 值 |
|---|---|
| 完成 / 出错 | 35 / 0 |
| 均 \(\lvert\Delta r_A\rvert\) | 0.056（相对 P0 随机明显更大） |
| AVTP A-out 比例 | 100%，均约 7.3 个 token |
| 全 token 仍对 | 100% |
| 压缩专属失败 | **0%** |

结论：槽位能抢，COCO VQA 在丢掉约 7 个 A token 后仍能答对。需要更紧预算和真正改答案的 U。

---

## 6. P2：\(r_{base}=0.2\) + 关键 token

筛选后 **32** 条在 \(r_{base}=0.2\) 下仍 clean-correct。Leave-one-out 发现：丢掉低分 A token 比丢掉高分 token 更容易改答案（drop-bot 7 vs drop-top 1）。U 按此重标后攻击。

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

这是 Claim 3 在替身上的证据：错误跟随**共享槽位驱逐**，不是 LAMP 式跨图语义污染。4/7 个压缩专属失败用的是 drop-bot U。

V-CachePoll 主结果就是这次 run：`out/p2_attack.json`。P3 不再重跑它。

---

## 7. P3：同一协议下的基线（主表）

32 条、\(r_{base}=0.2\)、\(\epsilon=16/255\)、40 步、相同前向/反向预算。  
V-CachePoll = P2 攻击。过关条件：压缩专属失败和 U 驱逐率明显高于基线，且全 token 大体正确。

| 方法 | n | 全 token 仍对 | 压缩专属失败 | 均 A-out | 均 U-out | U 驱逐率 |
|---|---|---|---|---|---|---|
| Random（40 次 J 评分重启） | 32 | 1.000 | 2/32 (0.062) | 0.91 | 0.91 | 0.138 |
| Task-PGD（降低 gold 的 CE） | 32 | 0.969 | **0/32 (0.000)** | 0.47 | 0.47 | 0.069 |
| CAA-B-only | 32 | 1.000 | 2/32 (0.062) | 3.09 | 2.91 | 0.430 |
| CAGE-B-only（B 进入 survivor） | 32 | 0.969 | 4/32 (0.125) | 3.69 | 3.47 | 0.518 |
| Rank-PGD（quota+evict，无搜索） | 32 | 0.969 | 5/32 (0.156) | 6.69 | 5.66 | 0.833 |
| Armijo（Rank + 每步交换搜索） | 32 | 0.969 | 5/32 (0.156) | 5.94 | 4.97 | 0.733 |
| **V-CachePoll** | 32 | 0.969 | **7/32 (0.219)** | 7.41 | 5.59 | 0.826 |

读法：

- 普通输出攻击 **打不出** 这个失败模式（Task-PGD 0/32）。
- CAA 能挪排名，定向驱逐不够。
- CAGE 证明：**攻击 token 活下来 ≠ 把 A 的关键证据挤掉**（4/32、U-evict 0.52，仍低于 V-CachePoll）。
- Rank 的 U 驱逐率已经和 V-CachePoll 持平（0.833 vs 0.826）。多出来的 2 个压缩专属失败来自阈值定向项 \(L_{crit}\)，不是步长搜索。
- 每步都做交换搜索（Armijo）**没有更好**。有限尺度搜索不要写成论文创新。

P3 判定：**CONTINUE**。

---

## 8. 附加迁移（非主表）

### FastV（8 条，非攻击）

同样总 K 下，last-token attention Top-K 几乎全留给 A（典型约 48 A / 4 B），因为 prompt 要求看第一张图。FastV 与 AVTP 答案均正确。Query-attention 天生偏被问的图；**共享配额 AVTP 才是更自然的缓存污染面**。不要把 FastV 当第一攻击对象。

### LLaVA-1.5-7B

本地权重完整，双图可以 pack（每图 576 个 CLIP token）。\(r_{base}=0.2\) 下 8 条里 6 条 AVTP clean-correct。

对这 6 条做 quota+evict PGD：

| 步数 | 均 A-out | U 驱逐率 | 全 token | 压缩专属失败 |
|---|---|---|---|---|
| 20 | 9.7 | 0.36 | 6/6 | **0/6** |
| 40 | 13.0 | 0.48 | 6/6 | **0/6** |

机制能迁（B 能抢 A 的槽），任务失败迁不过 576-token CLIP 网格。下一步迁移应是 **InternVL / LLaVA-OneVision**，不要在 1.5 上加 loss 硬打。

---

## 9. 对论文叙事的含义

三条命题的替身证据：

| Claim | 状态 |
|---|---|
| 1 共享自适应压缩缺少来源隔离 | 成立（P0/P1：固定 A，只改 B，\(r_A\) 下降、A survivor 被换） |
| 2 可构造跨图 visual eviction set | 弱成立（P3：结构攻击明显优于 Random/Task/CAA/CAGE；相对 Rank 的优势主要在任务失败而非 raw U-evict） |
| 3 错误由资源驱逐而非一般跨图干扰引起 | 成立（P2：无压缩 / 隔离配额 / restore-A 均可恢复那 7 个失败） |

方法上应对外说：**配额投毒 + 定向驱逐集合**。搜索模块可以写进实现，不要当卖点。

---

## 10. 过夜 P4–P9（2026-09-17 05:00–05:21，不覆盖 `out/p2_*`）

| 阶段 | n | Acc（无剪枝） | 压缩专属失败 | restore | δ_A=0 |
|---|---:|---:|---:|---:|---:|
| P4 freeze 全量 | 32 | 100% | **25.0%**（P2 为 21.9%） | 87.5%（7/8） | 100% |
| P5 smoke M_aux=1 | 2 | 100% | 0% | — | 100% |
| P6 smoke | 1 | 100% | 0% | — | 100% |
| P7 smoke family | 2 | 100% | 0% | — | 100% |
| P9 LAMP-like | 8 | 100% | 25.0% | 100% | 100% |

P4 8 条试点曾是 2/8 ASR，全量 8/32 未塌。P10 缺新模型权重；P11 TextVQA/ChartQA 缺图；P12 `L_amp` 按文档跳过。

**还没做：**

- P5 `n_aux` 2/4 完整攻击；P6 budget sweep；P7 N>2
- TextVQA 图、ChartQA、InternVL / LLaVA-OV / Qwen3-VL 权重

**不要做：**

- 再加一种越狱 loss
- 把有限尺度搜索写成独立创新
- 对称多图（NLVR2 / Counting）当主 A/B
- 在 LLaVA-1.5 上继续堆步数指望任务失败

---

## 11. 建议的下一步（按优先级）

1. **P5 multi-aux**：`n_aux` 2/4，超出 smoke。
2. **P6 budget sweep** 与 **P7 N>2**。
3. **补数据 / 换模型**：TextVQA 图与 InternVL / LLaVA-OV 仍等磁盘；LLaVA-1.5 只保留附录。

---

## 12. 代码与结果路径

项目根：`/root/autodl-tmp/multimodal_attack_project/V-CachePoll`

| 路径 | 内容 |
|---|---|
| `src/vcachepoll/` | 压缩器、Qwen/LLaVA wrapper、loss、P0–P3 攻击 |
| `configs/p{0,1,2,3}.yaml`、`configs/p{4,5,6,7,9,10,11}.yaml` | 各阶段配置 |
| `scripts/run_p{0,1,2,3}.py` | 主入口；另有 `run_cage.py`、`run_armijo.py`、`run_llava.py`、`run_fastv.py` |
| `scripts/run_extend.py`、`scripts/night_extend.py` | P4–P11 入口与过夜链 |
| `out/p4/` … `out/p9/` | 过夜扩展结果（json/md；扰动 `.pt` 不入库） |
| `tests/test_p0_cpu.py`、`test_p1_cpu.py`、`test_extend_cpu.py` | CPU 单测（P0/P1 回归 + P4–P12 张量逻辑） |
| `scripts/run_extend.py` | P4–P11 入口；GPU 占用时跳过攻击 |
| `out/extend/STATUS.md` | P4–P11 实时状态 |
| `out/P{0,1,2,3}_REPORT.md` | 各阶段判定报告 |
| `out/p2_attack.json` | V-CachePoll 主结果（32 条） |
| `out/p3_{random,task,caa,cage,rank,armijo}.json` | P3 基线 |
| `out/p2_screen.json` | 32 条 pool + `u_idx` |
| `out/fastv_probe.json` | FastV 8 条 |
| `out/llava_probe.json`、`out/llava_attack.json` | LLaVA 探针与 40 步攻击 |
| `PROTOCOL.md` | 冻结的运行协议 |

复现 P3 报告：

```bash
conda activate /root/autodl-tmp/conda/envs/vattack
export HF_HOME=/root/autodl-tmp/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
cd /root/autodl-tmp/multimodal_attack_project/V-CachePoll
python scripts/run_p3.py --stage report
```
