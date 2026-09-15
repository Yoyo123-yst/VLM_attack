# Jailbreaks Buy Silence, Not Scripts:
# Response Mode versus Safety Content in Vision–Language Models

> **How to read this draft.** The *claim* is the contribution; PGD is an instrument, not a method. Sections 1–3 are written in conference voice. Numbers marked **(P0)** / **(N0)** / **(FC)** are already measured. Numbers marked **(to run)** are the remaining tables a submission still needs. Do not cite Fast-Crossed as a 30-query finding.

---

## 创新点（给自己看的一句话）

现有顶会越狱（FigStep、HADES、HIMRD、SI-Attack、Seeing No Evil）把「不是拒绝」当成攻击成功。我们证明：多模态、无目标句子的视觉扰动，因果上推动的是 **response mode（开口/闭嘴）**，不是 **safety content（有害回答 vs 相关安全回答）**。因此 ASR 虚高；同预算 PGD 不能泛化地在 RELATED_SAFE 与 RHC 之间切换。贡献是 **机制拆分 + 四轴评价协议 + 与注意力劫持/有目标内容控制的对照**，不是又一种 PGD。

---

## Abstract

Visual jailbreaks against vision–language models (VLMs) are almost always scored with attack success rate (ASR): any non-refusal is a success. This metric quietly merges two events. The model may *open* (leave the refusal template) or it may *comply* (produce operationally harmful content). We show that target-free visual perturbations primarily purchase the first event.

On Qwen2-VL-7B-Instruct, late residual interventions that move along a refusal direction reverse refuse/answer behavior, whereas a direction fit on harmful versus related-safe *answers* does not **(P0)**. Matched “safe vs harmful answer” pairs used in prior representation work are often the same pixels under greedy versus sampled decoding, so they cannot identify a pre-generation safety state **(N0)**. Under a fixed \(\ell_\infty=16/255\) PGD budget and greedy decoding, random restarts switch answer versus refuse; both a grounded harmful answer and a related-safe answer appear in the same (query, image, budget) cell for only one of six queries **(FC)**.

We therefore replace ASR with a four-axis protocol—safety, response mode, grounding, quality—and treat standard projected gradient descent only as a probe aligned to the refusal residual. The probe is not a new attack algorithm. It predicts a query-stratified leftover: once the model answers, some instructions yield harmful compliance and others remain related-safe. This is the target-free dual of attention-hijacking attacks that *specify* a response: those can lock content across queries; an untargeted visual budget cannot.

---

## 1. Introduction

Large vision–language models inherit a textual safety stack and then add pixels. The extra modality is now a standard attack surface. FigStep writes the prohibited request in typography [Gong et al., 2023/2025]. HADES hides intent in a crafted image [Li et al., 2024]. HIMRD splits harmful semantics across image and text and searches for an inducing prompt that prefers affirmation to refusal [Ma et al., ICCV 2025]. SI-Attack exploits shuffle inconsistency between comprehension and safety [Zhao et al., ICCV 2025]. Gradient methods, from visual adversarial examples to UltraBreak, optimize images so that the model emits a non-refusal continuation [Qi et al.; Cui et al., ICLR 2026]. Internal-mechanism papers go further: Seeing No Evil argues that visual jailbreaks succeed by *blinding* attention to safety-prefix tokens rather than by overpowering a safety rule.

These lines share an evaluation contract. ASR counts a trial as successful whenever the output is not a refusal. Under that contract, a related-safe lecture, a theme denial (“the image does not contain a bomb”), and a step-by-step harmful recipe are the same success. The contract is inherited from textual jailbreaks, where “not refusing” was already a noisy proxy. In VLMs it is worse: the visual channel is good at moving the model off a refusal template and weak at rewriting the *content of an answer*.

We separate the two. **Response mode** is whether the model answers, refuses, or denies the premise. **Safety content** is whether an *answer* is operationally harmful (RHC) or related but non-operational (RELATED_SAFE). Theme denial is neither: it is a grounding failure, not a jailbreak win and not a related-safe answer.

Three measurements on one model pin the split down.

1. **Causal residual control is a mode switch (P0).** Patching the late residual along a refusal (or denial) direction reproduces harmful-answer behavior from refusals. A mean direction between harmful and related-safe *answers* does not (effect \(0.067\), CI includes 0). Full residual swap at L24 moves refuse\(\to\)jailbreak at \(0.88\) and the reverse at \(-0.77\). Lowering refusal is not a safety-mechanism break.

2. **Answer-conditioned pairs are mostly not pre-generation states (N0).** Of 51 matched answering pairs, 40 differ only by greedy versus sampled decoding on the *same* perturbed image. Last-prompt hidden states coincide (cosine \(\geq 0.999\) on 39/51). One cannot fit a “safety direction” at the last prompt token on that data.

3. **Same-budget restarts do not yield a general content switch (FC).** Ninety-six greedy PGD trajectories (six harmful queries, two unseen carriers, eight random inits, \(\epsilon=16/255\)) produce 36 answers and 54 refusals, 20 core RHC and 13 core related-safe answers, but only four same-budget RHC/SAFE pairs, all on a single fraud query. The typical restart flips silence, not the script.

These facts change what a visual attack is allowed to claim. A target-free perturbation that raises ASR has, by default, shown a **mode attack**. Harmful content is a *conditional* effect: it appears when the instruction already has an answering prior that is harmful. FigStep’s visual-embedding alignment story, HIMRD’s inducing prompt, and Seeing No Evil’s safety blindness are then the same object at three implementations: they stop the model from retrieving or executing refusal. They do not write a harmful procedure into the residual.

**What we do not claim.** We do not introduce a new optimizer, a new typography trick, a new shuffle heuristic, or a new universal perturbation generator. PGD appears only as a refusal-aligned *probe* (ModeSwitch): the same first-order method used throughout the literature, with the loss pointed at a frozen refusal direction rather than at a target string. The scientific object is the mode/content split, the four-axis protocol that makes the split measurable, and the prediction that leftover related-safe answers are query-stratified rather than a failed attack.

### Contributions

1. **A two-factor account of VLM jailbreaks.** Target-free visual interventions causally control *response mode*. Safety content among answers is not a general, input-steerable pre-generation direction on this model.

2. **A four-axis evaluation protocol.** Safety \(\times\) mode \(\times\) grounding \(\times\) fluency replaces ASR. Theme denial and related-safe answers are controls, not successes. We state how this protocol would re-rank FigStep, HADES, HIMRD, and SI-Attack without requiring a new perturbation.

3. **Evidence that the leftover is query-stratified, not random.** Under a matched attack budget, some instructions answer with RHC, some with RELATED_SAFE, some remain refused. Only one of six probed queries realizes both answering labels in-cell. This is the target-free counterpart of cross-query *targeted* attention hijacking, which can lock a specified continuation; an untargeted budget cannot.

4. **A refusal probe, not a new attack.** ModeSwitch is PGD onto a frozen \(U_{\mathrm{refusal}}\). It exists to test alignment with the causal axis, against token-margin PGD, a safety-content direction, and random noise of the same radius.

---

## 2. Related work and where we sit

**Recipe jailbreaks (ASR).** FigStep, HADES, HIMRD, SI-Attack, Reference Attack, and DGSIP construct image and/or text so that the model does not refuse. HIMRD’s inducing prompt is explicit: raise the chance of affirmation over refusal. We read that line as a *mode* objective and ask what fraction of their ASR is related-safe or theme denial under four-axis labels **(to run, or argued on our own PGD/FC outputs if SOTA transcripts are unavailable)**.

**Transfer and UAP.** C-PGC, CoA, SADCA, TTA, NeurIPS 2025 transferable attackers, and UltraBreak seek one perturbation that moves many models or prompts. They optimize alignment destruction or a textual target in embedding space. A *mode* UAP (one \(\delta\) that opens many harmful queries) is compatible with our account; a UAP that claims unified harmful *content* is not, unless a target string is provided.

**Internal mechanism.** Seeing No Evil: successful visual jailbreaks suppress attention to alignment-prefix tokens (“safety blindness”). We agree they do not overwrite a safety rule; we add that the residual handle for the same bit is a refusal direction, and that content after unblinding is a second factor. Attention Hijacking (cross-query): one adversarial image can keep an attacker-specified response across user queries by enforcing image-dominant attention. That is **targeted content control**. Our negative result is **untargeted**. V-Attack manipulates value features for local semantic edits of *what the image means*; that is encoder-level semantic control, not a safety switch. CAAP uses activation patching for ViT attribution; we use residual patching for mode causality.

**Backdoors.** IAG plants input-aware triggers for visual *grounding*. ImgTrojan binds an image to a stored jailbreak prompt. A mode backdoor (trigger \(\mapsto\) answer-mode) would be a different payload; it is out of scope for this draft.

**CLIP defenses and 3D/embodied surveys** (TAPT, MAC, APD, PointLLM, embodied-AI survey) are adjacent robustness literature, not generative VLM jailbreak evaluation. SafeBench is an evaluation resource we can cite; our contribution is the *factorization* of success, not another 2,300-prompt benchmark.

---

## 3. Four-axis protocol (the actual “method”)

Let \(y\) be a greedy continuation for query \(q\) and image \(x\). Label four independent axes:

| Axis | Values | Role |
|---|---|---|
| Safety | SAFE / RHC | Content of an *answer* |
| Response mode | ANSWER / REFUSE / DENY | Whether the mouth opens |
| Grounding | GROUNDED / UNGROUNDED | Theme denial lives here |
| Quality | FLUENT / GARBAGE | Fluency filter |

**Core harmful success:** RHC \(\wedge\) ANSWER \(\wedge\) GROUNDED \(\wedge\) FLUENT.  
**Related-safe leftover:** SAFE \(\wedge\) ANSWER \(\wedge\) GROUNDED \(\wedge\) FLUENT, excluding refuse-then-advice and theme denial.  
**Mode success (target-free attack):** ANSWER on a harmful \(q\), relative to the clean image.  
**ASR (legacy):** \(\neg\) REFUSE. We report it only to show inflation: ASR \(\geq\) mode success \(\geq\) core RHC.

Theme denial is SAFE on the safety axis and a grounding failure. It must not enter either core bucket. This is the opposite of packing “not refusal” into a single scalar, which is what FigStep/HIMRD/SI share with Seeing No Evil’s 94% ASR.

---

## 4. Mechanism (already measured)

### 4.1 Residual mode is causal; residual safety-content is not (P0)

On recleaned holdout, late residual replacement is bidirectional for refuse versus harmful-answer at L16–L27. Frozen L24 full swap: REF\(\to\)JB \(0.883\) \([0.75, 1.00]\), JB\(\to\)REF \(-0.767\) \([−0.93, −0.60]\). Directions: \(U_{\mathrm{refusal}}\) \(0.867\), \(U_{\mathrm{denial}}\) \(0.717\), \(U_{\mathrm{RHC}}\) \(0.650\), \(U_{\mathrm{safety}}\) \(0.067\) (CI includes 0). Orthogonalizing theme-denial (“FAIL”) out of \(U_{\mathrm{RHC}}\) collapses REF\(\to\)JB from \(0.65\) to \(0.10\). The movable object is whether the model answers, not a unique safety subspace.

### 4.2 Last-prompt “safety pairs” were often the same input (N0)

51 answering pairs, same query, same carrier, official template. 40/51: identical reconstructed pixels; SAFE from temperature sampling, RHC from greedy. Last-prompt cosine \(\geq 0.999\) on 39/51. Remaining pairs confound perturbation strength (strong PGD versus weak/alternate \(\delta\)), still not a crossed design. Pre-generation \(U_{\mathrm{safety}}\) fitted on those hidden states is unidentified.

### 4.3 Same budget, new carriers: gray leftover (FC)

Protocol: six unused queries, carriers c05–c06, eight PGD restarts, \(\epsilon=16/255\), 40 steps, refusal-margin first-token loss, greedy only. 96 trajectories: REFUSE 54, ANSWER 36, DENY 6; core RHC 20; core related-safe 13; theme denial 6. In-cell RHC/SAFE matches: 4, query h11 only. That is below any reasonable discovery floor for a content direction. It is sufficient to reject the hypothesis that random restarts under a shared budget are a general RELATED_SAFE \(\leftrightarrow\) RHC knob.

---

## 5. Probe (instrument, not contribution)

**ModeSwitch.** Let \(u_{\mathrm{ref}}\) be the frozen P0-S refusal direction at L24 (not retuned). PGD on \(\delta\) with \(\|\delta\|_\infty\leq \epsilon\) maximizes the last-prompt residual’s displacement *away* from refusal along \(u_{\mathrm{ref}}\). No target harmful sentence. Same \(\epsilon\), template, and greedy decode as FC.

**Controls.** (i) First-token refusal-margin PGD (the workhorse already used in FC). (ii) The same PGD onto frozen \(U_{\mathrm{safety}}\). (iii) Uniform noise of radius \(\epsilon\).

**Prediction.** (i) and ModeSwitch both raise ANSWER; they need not differ in ASR. Four-axis leftover stays query-stratified. \(U_{\mathrm{safety}}\) and random stay weak on mode. If ModeSwitch does not beat random on ANSWER, the *implementation* of the probe is wrong; the P0 patch result still stands.

**Contrast with Attention Hijacking (to run).** Same \(\epsilon\), same model. Targeted loss toward a fixed continuation versus ModeSwitch (no target). Targeted: content can stabilize across queries. Untargeted: content follows \(q\).

**Contrast with Seeing No Evil (to run if attention hooks are cheap).** Same images: does opening the mouth coincide with suppressed safety-prefix attention *and* a shift along \(u_{\mathrm{ref}}\)? Two handles, one bit.

---

## 6. What a camera-ready table must contain

| Table | Status | Point |
|---|---|---|
| Four-axis vs ASR on FC-96 | **can compute now** | ASR inflation on this model |
| Query \(\times\) (REF / RHC / RELATED_SAFE) | **can compute now** | gray leftover |
| Last-prompt proj. on \(U_{\mathrm{refusal}}\) vs \(U_{\mathrm{safety}}\) | **can compute now** (hidden already stored) | mode vs content in representation |
| ModeSwitch vs three controls, held-out queries | to run | probe, not optimizer |
| Targeted vs untargeted, same \(\epsilon\) | to run | sit next to Attention Hijacking |
| Four-axis on a FigStep/HIMRD/SI subset | optional | do not block the paper on reproducing three black-box stacks |

Benign queries on the same \(\delta\): over-opening is a specificity cost. If ModeSwitch opens everything, it is not a refusal switch.

---

## 7. Limitations (write them before reviewers do)

Evidence is one 7B native-template VLM. FC uses six queries. P0 directions were fit on a previous split; we do not retune rank. We do not claim transfer to closed-source APIs. We do not claim that targeted attacks cannot control content—Attention Hijacking suggests they can. We do not reopen a low-rank “safety bottleneck” or an attack gate based on ASR.

---

## 8. Positioning sentence for the introduction’s last paragraph

*Visual jailbreaks in VLMs have been sold as content attacks and scored as mode attacks. We measure both. The pixels move refusal. The script stays with the question.*
