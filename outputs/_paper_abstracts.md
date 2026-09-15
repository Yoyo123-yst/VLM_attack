# Paper abstracts harvest



## A safety evaluation framework for multimodal large language models.pdf

```
Springer Nature 2021 LATEX template
SafeBench: A Safety Evaluation Framework for Multimodal
Large Language Models
Zonghao Ying1, Aishan Liu1,4*, Siyuan Liang2, Lei Huang1, Jinyang Guo1, Wenbo
Zhou5, Xianglong Liu1,3,4* and Dacheng Tao6*
1Beihang University, China.
2National University of Singapore, Singapore.
3Zhongguancun Laboratory, China.
4Hefei Comprehensive National Science Center, China.
5University of Science and Technology of China, China.
6Nanyang Technological University, Singapore.
*Corresponding author(s). E-mail(s): liuaishan@buaa.edu.cn; xlliu@buaa.edu.cn;
dacheng.tao@gmail.com;
Contributing authors: yingzonghao@buaa.edu.cn; pandaliang521@gmail.com;
huanglei@buaa.edu.cn; guojinyang@buaa.edu.cn; welbeckz@ustc.edu.cn;
Abstract
Multimodal Large Language Models (MLLMs) are showing strong safety concerns (e.g., generating harmful outputs for users), which motivates the development of safety evaluation benchmarks. However, we observe that
existing safety benchmarks for MLLMs show limitations in query quality and evaluation reliability limiting the
detection of model safety implications as MLLMs continue to evolve. In this paper, we propose SafeBench, a
comprehensive framework designed for conducting safety evaluations of MLLMs. Our framework consists of a
comprehensive harmful query dataset and an automated evaluation protocol that aims to address the above limitations, respectively. We first design an automatic safety dataset generation pipeline, where we employ a set of
LLM judges to recognize and categorize the risk scenarios that are most harmful and diverse for MLLMs; based
on the taxonomy, we further ask these judges to generate high-quality harmful queries accordingly resulting in 23
risk scenarios with 2,300 multi-modal harmful query pairs. During safety evaluation, we draw inspiration from
the jury system in judicial proceedings and pioneer the jury deliberation evaluation protocol that adopts collaborative LLMs to evaluate whether target models exh

>>> ABSTRACT:
Multimodal Large Language Models (MLLMs) are showing strong safety concerns (e.g., generating harmful outputs for users), which motivates the development of safety evaluation benchmarks. However, we observe that
existing safety benchmarks for MLLMs show limitations in query quality and evaluation reliability limiting the
detection of model safety implications as MLLMs continue to evolve. In this paper, we propose SafeBench, a
comprehensive framework designed for conducting safety evaluations of MLLMs. Our framework consists of a
comprehensive harmful query dataset and an automated evaluation protocol that aims to address the above limitations, respectively. We first design an automatic safety dataset generation pipeline, where we employ a set of
LLM judges to recognize and categorize the risk scenarios that are most harmful and diverse for MLLMs; based
on the taxonomy, we further ask these judges to generate high-quality harmful queries accordingly resulting in 23
risk scenarios with 2,300 multi-modal harmful query pairs. During safety evaluation, we draw inspiration from
the jury system in judicial proceedings and pioneer the jury deliberation evaluation protocol that adopts collaborative LLMs to evaluate whether target models exhibit specific harmful behaviors, providing a reliable and unbiased
assessment of content security risks. In addition, our benchmark can also be extended to the audio modality showing high scalability and potential. Based on our framework, we conducted large-scale experiments on 15 widelyused open-source MLLMs and 6 commercial MLLMs ( e.g., GPT-4o, Gemini), where we revealed widespread
safety issues in existing MLLMs and instantiated several insights on MLLM safety performance such as image
quality and parameter size. Our benchmark offers (1) a comprehensive dataset and evaluation pipeline for MLLM
safety evaluation; (2) an up-to-date leaderboard on MLLM safety; and (3) a nuanced understanding of the safety
issues presented by these models. Our benchmark and code are available at https://safebench-mm.github.io/.
```



## Adversarial_Prompt_Distillation_For_Vision-Language_Models.pdf

```
ADVERSARIAL PROMPT DISTILLATION FOR VISION-LANGUAGE MODELS
Luo Lin1 Xin Wang1 Bojia Zi2 Shihao Zhao3 Xingjun Ma1†
1Fudan University 2Chinese University of Hong Kong 3University of Hong Kong
ABSTRACT
Large pre-trained Vision-Language Models (VLMs) such
as Contrastive Language-Image Pre-training (CLIP) have
been shown to be susceptible to adversarial attacks, raising
concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One
promising approach for robustifying pre-trained VLMs is
Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However,
existing APT methods are mostly single-modal methods that
design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy.
In this work, we propose Adversarial Prompt Distillation
(APD), a bimodal and online distillation framework that
learns optimized prompts for both visual and textual modalities through distillation with a cleanly pre-trained CLIP
teacher model. Extensive experiments on multiple benchmark datasets demonstrate that our APD method surpasses
state-of-the-art APT approaches in both adversarial robustness and clean accuracy. Moreover, the effectiveness of APD
validates the feasibility of using a non-robust teacher to enhance the generalization and robustness of fine-tuned VLMs.
Index Terms— Adversarial Defense, Adversarial Prompt
Distillation, Vision-Language Models
1. INTRODUCTION
Large pre-trained Vision-Language Models (VLMs) [1, 2]
have demonstrated remarkable success in aligning visual and
textual representations through a joint embedding space, enabling superior performance on cross-modal tasks. For example, CLIP [1] employed contrastive learning to align matched
image-text pairs while separating mismatched ones. Despite
their effectiveness, these models remain vulnerable to adversarial attacks [3, 4] , where carefully crafted adversarial pertu

>>> ABSTRACT:
Large pre-trained Vision-Language Models (VLMs) such
as Contrastive Language-Image Pre-training (CLIP) have
been shown to be susceptible to adversarial attacks, raising
concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One
promising approach for robustifying pre-trained VLMs is
Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However,
existing APT methods are mostly single-modal methods that
design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy.
In this work, we propose Adversarial Prompt Distillation
(APD), a bimodal and online distillation framework that
learns optimized prompts for both visual and textual modalities through distillation with a cleanly pre-trained CLIP
teacher model. Extensive experiments on multiple benchmark datasets demonstrate that our APD method surpasses
state-of-the-art APT approaches in both adversarial robustness and clean accuracy. Moreover, the effectiveness of APD
validates the feasibility of using a non-robust teacher to enhance the generalization and robustness of fine-tuned VLMs.
```



## CoGA_A_Collaborative_Gray-Box_Adversarial_Attack_for_Multimodal_Language_Models(1).pdf

```
870 IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY , VOL. 21, 2026
CoGA: A Collaborative Gray-Box Adversarial
Attack for Multimodal Language Models
Tong Wu
 , Feng Lin
 , Senior Member , IEEE, Gaojian Wang
 , Tiantian Liu
 ,
Zhibo Wang
 , Senior Member , IEEE, Weizhi Meng
 , Senior Member , IEEE, Ajian Liu
 ,
and Kui Ren
 , Fellow, IEEE
Abstract—Multimodal language models (LMs) have shown
signiﬁcant potential for applications across various domains but
remain vulnerable to adversarial attacks. Current research in
white-box or black-box settings generally struggles with unrealistic attack assumptions and limited e ﬃcacy of targeted attacks.
This paper introduces CoGA, a novel gray-box collaborative
adversarial attack method for multimodal LMs. Under our graybox settings, attackers have access only to the victim model’s
input encoders. With the guidance of di ﬀerent modalities, we
perturb the embedding representations from encoders to disrupt
the semantic alignment across modalities, ultimately causing
inaccurate outputs on various downstream tasks. Speciﬁcally, we
integrate text embeddings into the loss calculations of the image
attack and utilize image embeddings to guide the ranking of
vulnerable words and the selection of ﬁnal samples. Extensive
experiments demonstrate that our method achieves superior
attack performance across diverse models and tasks, suggesting
the shared vulnerability of multimodal LMs in confronting
adversarial challenges. Our work provides new insights into the
security of multimodal LMs, facilitating the deployment of more
robust and secure models in practical applications.
Index Terms —Adversarial attack, multimodal, gray-box
attack, semantic alignment, language model.
I. I NTRODUCTION
T
HE recent success of multimodal language models (LMs)
has attracted broad attention from both academia and
industry [1], [2], [3]. These advanced models process and
align heterogeneous data (e.g., text, images, and audio) for
cross-modal understan

>>> ABSTRACT:
Multimodal language models (LMs) have shown
signiﬁcant potential for applications across various domains but
remain vulnerable to adversarial attacks. Current research in
white-box or black-box settings generally struggles with unrealistic attack assumptions and limited e ﬃcacy of targeted attacks.
This paper introduces CoGA, a novel gray-box collaborative
adversarial attack method for multimodal LMs. Under our graybox settings, attackers have access only to the victim model’s
input encoders. With the guidance of di ﬀerent modalities, we
perturb the embedding representations from encoders to disrupt
the semantic alignment across modalities, ultimately causing
inaccurate outputs on various downstream tasks. Speciﬁcally, we
integrate text embeddings into the loss calculations of the image
attack and utilize image embeddings to guide the ranking of
vulnerable words and the selection of ﬁnal samples. Extensive
experiments demonstrate that our method achieves superior
attack performance across diverse models and tasks, suggesting
the shared vulnerability of multimodal LMs in confronting
adversarial challenges. Our work provides new insights into the
security of multimodal LMs, facilitating the deployment of more
robust and secure models in practical applications.
```



## Fang_One_Perturbation_is_Enough_On_Generating_Universal_Adversarial_Perturbations_against_ICCV_2025_paper(1).pdf

```
One Perturbation is Enough: On Generating Universal Adversarial
Perturbations against Vision-Language Pre-training Models
Hao Fang1,* Jiawei Kong1,2,∗ Wenbo Yu1 Bin Chen2,3† Jiawei Li4
Hao Wu5 Shu-Tao Xia1,3 Ke Xu1
1Tsinghua University 2 Harbin Institute of Technology, Shenzhen 3Peng Cheng Laboratory
4Huawei Technology 5Shenzhen ShenNong Information Technology Co., Ltd.
ffhibnese@gmail.com, kongjiawei@stu.hit.edu.cn, wenbo.research@gmail.com, chenbin2021@hit.edu.cn,
li-jw15@tsinghua.org.cn, whpc79@163.com, xiast@sz.tsinghua.edu.cn, xuke@tsinghua.edu.cn;
Abstract
Vision-Language Pre-training (VLP) models have exhibited
unprecedented capability in many applications by taking
full advantage of the learned multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these attacks are generally instance-specific and require generating perturbations for each input sample. In
this paper, we reveal that VLP models are also susceptible
to the instance-agnostic universal adversarial perturbation
(UAP). Specifically, we design a novel Contrastive-training
Perturbation Generator with Cross-modal conditions (CPGC). In light that the pivotal multimodal alignment in VLP
models is achieved via contrastive learning, we devise to
turn this powerful weapon against VLP models themselves.
I.e., we employ a malicious version of contrastive learning
to train the proposed generator using our carefully crafted
positive and negative image-text pairs. Once training is
complete, the generator is able to produce universal perturbations that can essentially destroy the established alignment relationship in VLP models. Besides, C-PGC fully
utilizes the characteristics of Vision-and-Language (V+L)
scenarios by incorporating both unimodal and cross-modal
information as effective guidance. Extensive experiments
show that C-PGC successfully forces adversarial samples
to move away from their original area in the VLP 

>>> ABSTRACT:
Vision-Language Pre-training (VLP) models have exhibited
unprecedented capability in many applications by taking
full advantage of the learned multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these attacks are generally instance-specific and require generating perturbations for each input sample. In
this paper, we reveal that VLP models are also susceptible
to the instance-agnostic universal adversarial perturbation
(UAP). Specifically, we design a novel Contrastive-training
Perturbation Generator with Cross-modal conditions (CPGC). In light that the pivotal multimodal alignment in VLP
models is achieved via contrastive learning, we devise to
turn this powerful weapon against VLP models themselves.
I.e., we employ a malicious version of contrastive learning
to train the proposed generator using our carefully crafted
positive and negative image-text pairs. Once training is
complete, the generator is able to produce universal perturbations that can essentially destroy the established alignment relationship in VLP models. Besides, C-PGC fully
utilizes the characteristics of Vision-and-Language (V+L)
scenarios by incorporating both unimodal and cross-modal
information as effective guidance. Extensive experiments
show that C-PGC successfully forces adversarial samples
to move away from their original area in the VLP model’s
feature space, thus fundamentally enhancing attack performance across various victim models and V+L tasks.
```



## FigStep Jailbreaking Large Vision-language Models via Typographic Visual Prompts.pdf

```
FigStep: Jailbreaking Large Vision-Language Models via Typographic Visual
Prompts
Yichen Gong1*, Delong Ran 2*, Jinyuan Liu 3, Conglei Wang4,
Tianshuo Cong3†, Anyu Wang3,5,6†, Sisi Duan 3,5,6,7, Xiaoyun Wang3,5,6,7,8
1Department of Computer Science and Technology, Tsinghua University,
2Institute for Network Sciences and Cyberspace, Tsinghua University,
3Institute for Advanced Study, BNRist, Tsinghua University,
4Carnegie Mellon University,
5Zhongguancun Laboratory,
6National Financial Cryptography Research Center,
7Shandong Institute of Blockchain,
8School of Cyber Science and Technology, Shandong University
{gongyc18, rdl22, liujinyuan24}@mails.tsinghua.edu.cn, congleiw@andrew.cmu.edu,
{congtianshuo, anyuwang, duansisi, xiaoyunwang}@tsinghua.edu.cn
Abstract
Large Vision-Language Models (LVLMs) signify a groundbreaking paradigm shift within the Artificial Intelligence (AI)
community, extending beyond the capabilities of Large Language Models (LLMs) by assimilating additional modalities
(e.g., images). Despite this advancement, the safety of LVLMs
remains adequately underexplored, with a potential overreliance on the safety assurances purported by their underlying
LLMs. In this paper, we propose FigStep, a straightforward
yet effective black-box jailbreak algorithm against LVLMs. Instead of feeding textual harmful instructions directly, FigStep
converts the prohibited content into images through typography to bypass the safety alignment. The experimental results
indicate that FigStep can achieve an average attack success
rate of 82.50% on six promising open-source LVLMs. Not
merely to demonstrate the efficacy of FigStep, we conduct
comprehensive ablation studies and analyze the distribution
of the semantic embeddings to uncover that the reason behind
the success of FigStep is the deficiency of safety alignment
for visual embeddings. Moreover, we compare FigStep with
five text-only jailbreaks and four image-based jailbreaks to
demonstrate the superiority of FigStep, i

>>> ABSTRACT:
Large Vision-Language Models (LVLMs) signify a groundbreaking paradigm shift within the Artificial Intelligence (AI)
community, extending beyond the capabilities of Large Language Models (LLMs) by assimilating additional modalities
(e.g., images). Despite this advancement, the safety of LVLMs
remains adequately underexplored, with a potential overreliance on the safety assurances purported by their underlying
LLMs. In this paper, we propose FigStep, a straightforward
yet effective black-box jailbreak algorithm against LVLMs. Instead of feeding textual harmful instructions directly, FigStep
converts the prohibited content into images through typography to bypass the safety alignment. The experimental results
indicate that FigStep can achieve an average attack success
rate of 82.50% on six promising open-source LVLMs. Not
merely to demonstrate the efficacy of FigStep, we conduct
comprehensive ablation studies and analyze the distribution
of the semantic embeddings to uncover that the reason behind
the success of FigStep is the deficiency of safety alignment
for visual embeddings. Moreover, we compare FigStep with
five text-only jailbreaks and four image-based jailbreaks to
demonstrate the superiority of FigStep, i.e., negligible attack
costs and better attack performance. Above all, our work reveals that current LVLMs are vulnerable to jailbreak attacks,
which highlights the necessity of novel cross-modality safety
alignment techniques.
Code, Datasets — https://github.com/ThuCCSLab/FigStep
Extended version — https://arxiv.org/abs/2311.05608
```



## IAG Input-aware Backdoor Attack on VLM-based Visual Grounding.pdf

```
IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
Junxian Li1*, Beining Xu1*, Simin Chen3, Jiatong Li4, Jingdi Lei5, Haodong Zhao1†, Di Zhang2†
1Shanghai Jiao Tong University,2Fudan University, 3Columbia University,
4Hong Kong Polytechnic University, 5Nanyang Technological University
{lijunxian0531, zhaohaodong}@sjtu.edu.cn, di.zhang@ustc.edu
W ARNING: This paper contains unsafe model responses.
Abstract
Recent advances in vision-language models (VLMs) have
significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack
on VLM-based visual grounding. Unlike prior attacks that
rely on static triggers or fixed targets, we propose IAG,
a method that dynamically generates input-aware, textguided triggers conditioned on any specified target object description to execute the attack. This is achieved
through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that
balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth.
Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+,
RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate
that IAG achieves thebestASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses,
and exhibiting transferability across datasets and models. These findings underscore critical security risks in
grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
Code is available at https://git

>>> ABSTRACT:
Recent advances in vision-language models (VLMs) have
significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack
on VLM-based visual grounding. Unlike prior attacks that
rely on static triggers or fixed targets, we propose IAG,
a method that dynamically generates input-aware, textguided triggers conditioned on any specified target object description to execute the attack. This is achieved
through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that
balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth.
Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+,
RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate
that IAG achieves thebestASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses,
and exhibiting transferability across datasets and models. These findings underscore critical security risks in
grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
Code is available at https://github.com/lijunxian111/IAG.
```



## Images are Achilles Heel of Alignment Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models.pdf

```
Images are Achilles’ Heel of Alignment:
Exploiting Visual Vulnerabilities for Jailbreaking
Multimodal Large Language Models
Yifan Li1,3,⋆, Hangyu Guo1,3,⋆, Kun Zhou2,3,⋆,
Wayne Xin Zhao1,3,†, and Ji-Rong Wen1,2,3
1 Gaoling School of Artificial Intelligence, Renmin University of China
2 School of Information, Renmin University of China
3 Beijing Key Laboratory of Big Data Management and Analysis Methods
{liyifan0925,hyguo0220,batmanfly}@gmail.com
Abstract. In this paper, we study the harmlessness alignment problem
of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative
MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method
named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images.
Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of
90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and
data are available at https://github.com/RUCAIBox/HADES.
Warning: this paper contains example data that may be offensive.
Keywords: Multimodal Large Language Models· Harmlessness Alignment · Adversarial Attack
1 Introduction
Recently,byleveragingthepowerfulcapacityoflargelanguagemodels(LLMs)[33],
a variety of multimodal large language models (MLLMs) [32] have emerged,
which can process both textual and visual information similarly as that LLMs
process textual input. MLLMs have not only shown superior performance in
various visual-language tasks but also possess the capability to engage in imagerelated dialogues with human users [15,35]. However, MLLMs also confront similar harmlessness challenges that afflict their backbone LLMs.
Despite undergoing harmlessness alignment like reinforcement learning from
humanfeedback(RLHF)[19],LLMsremainvulnerabletoblack-b

>>> ABSTRACT:
. In this paper, we study the harmlessness alignment problem
of multimodal large language models (MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative
MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method
named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images.
Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate (ASR) of
90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and
data are available at https://github.com/RUCAIBox/HADES.
Warning: this paper contains example data that may be offensive.
```



## Kim_When_CLIP_Sees_More_It_Fights_Back_Harder_Multi-View_Guided_CVPR_2026_paper.pdf

```
When CLIP Sees More, It Fights Back Harder: Multi-View Guided Adaptive
Counterattacks for Test-Time Adversarial Robustness
Sunoh Kim
Dankook University
Yongin, South Korea
suno8386@dankook.ac.kr
Daeho Um*
University of Seoul
Seoul, South Korea
daehoum@uos.ac.kr
Abstract
Vision-language models such as CLIP have achieved remarkable zero-shot recognition capabilities, yet their robustness against adversarial perturbations remains limited.
Test-time counterattack (TTC) was recently proposed to improve CLIP’s robustness by perturbing an input image to
steer it away from a corrupted state during inference. However, TTC remains fragile under strong attacks because its
counterattack relies on a directly corrupted original view
and employs a noise-driven hard-gating scheme that cannot adapt to varying corruption severity. To address these
limitations, we introduce Multi-view guided Adaptive Counterattack (MAC), which performs counterattacks for multiview with corruption-aware soft weighting. Specifically,
MAC first constructs augmented views of an input image to
obtain diverse embeddings. It then performs counterattacks
to refine corrupted embeddings of views. Next, MAC adaptively scales the counterattack intensity for each view based
on its estimated corruption degree. Finally, the adaptively
counterattacked views are aggregated to yield a robust final prediction. Extensive experiments across 20 datasets
and diverse attack scenarios demonstrate that MAC substantially improves robustness while preserving high inference speed and memory efficiency with its tuning-free design. Our code is available athttps://github.com/
sunoh-kim/MAC.
1. Introduction
Vision-language models (VLMs) learn joint image-text
representations that enable strong zero-shot generalization
across diverse tasks and domains, attracting growing attention from both academia and industry [24, 27, 35, 46, 47,
50, 82, 85]. Among them, CLIP [46] stands out as a representative model that achieves remarkable zero-s

>>> ABSTRACT:
Vision-language models such as CLIP have achieved remarkable zero-shot recognition capabilities, yet their robustness against adversarial perturbations remains limited.
Test-time counterattack (TTC) was recently proposed to improve CLIP’s robustness by perturbing an input image to
steer it away from a corrupted state during inference. However, TTC remains fragile under strong attacks because its
counterattack relies on a directly corrupted original view
and employs a noise-driven hard-gating scheme that cannot adapt to varying corruption severity. To address these
limitations, we introduce Multi-view guided Adaptive Counterattack (MAC), which performs counterattacks for multiview with corruption-aware soft weighting. Specifically,
MAC first constructs augmented views of an input image to
obtain diverse embeddings. It then performs counterattacks
to refine corrupted embeddings of views. Next, MAC adaptively scales the counterattack intensity for each view based
on its estimated corruption degree. Finally, the adaptively
counterattacked views are aggregated to yield a robust final prediction. Extensive experiments across 20 datasets
and diverse attack scenarios demonstrate that MAC substantially improves robustness while preserving high inference speed and memory efficiency with its tuning-free design. Our code is available athttps://github.com/
sunoh-kim/MAC.
```



## Li_IAG_Input-aware_Backdoor_Attack_on_VLM-based_Visual_Grounding_CVPR_2026_paper.pdf

```
IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
Junxian Li1*, Beining Xu1*, Simin Chen3, Jiatong Li4, Jingdi Lei5, Haodong Zhao1†, Di Zhang2†
1Shanghai Jiao Tong University,2Fudan University, 3Columbia University,
4Hong Kong Polytechnic University, 5Nanyang Technological University
{lijunxian0531, zhaohaodong}@sjtu.edu.cn, di.zhang@ustc.edu
W ARNING: This paper contains unsafe model responses.
Abstract
Recent advances in vision-language models (VLMs) have
significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack
on VLM-based visual grounding. Unlike prior attacks that
rely on static triggers or fixed targets, we propose IAG,
a method that dynamically generates input-aware, textguided triggers conditioned on any specified target object description to execute the attack. This is achieved
through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that
balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth.
Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+,
RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate
that IAG achieves thebestASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses,
and exhibiting transferability across datasets and models. These findings underscore critical security risks in
grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
Code is available at https://git

>>> ABSTRACT:
Recent advances in vision-language models (VLMs) have
significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack
on VLM-based visual grounding. Unlike prior attacks that
rely on static triggers or fixed targets, we propose IAG,
a method that dynamically generates input-aware, textguided triggers conditioned on any specified target object description to execute the attack. This is achieved
through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that
balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth.
Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+,
RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate
that IAG achieves thebestASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses,
and exhibiting transferability across datasets and models. These findings underscore critical security risks in
grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
Code is available at https://github.com/lijunxian111/IAG.
```



## Ma_Heuristic-Induced_Multimodal_Risk_Distribution_Jailbreak_Attack_for_Multimodal_Large_Language_ICCV_2025_paper.pdf

```
Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for
Multimodal Large Language Models
Teng Ma1,2,3, Xiaojun Jia 4,†, Ranjie Duan 5, Xinfeng Li 4, Yihao Huang 4,
Xiaoshuang Jia6, Zhixuan Chu 2,7, Wenqi Ren 1,8,9.†
1Shenzhen Campus of Sun Yat-Sen University 2The State Key Laboratory of Blockchain and Data Security, Zhejiang University
3 BraneMatrix AI 4Nanyang Technological University 5Alibaba Group 6Renmin University of China
7Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security
8 Guangdong Key Laboratory of Information Security Technology 9MoE Key Laboratory of Information Technology
Abstract
With the rapid advancement of multimodal large language
models (MLLMs), concerns regarding their security have
increasingly captured the attention of both academia and
industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique
challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper , we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box
and consists of two elements: multimodal risk distribution
strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs.
The heuristic-induced search strategy identiﬁes two types
of prompts: the understanding-enhancing prompt, which
helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of afﬁrmative outputs over refusals, enabling a successful jailbreak
attack. HIMRD achieves an average attack success rate
(ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLL

>>> ABSTRACT:
With the rapid advancement of multimodal large language
models (MLLMs), concerns regarding their security have
increasingly captured the attention of both academia and
industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique
challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper , we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box
and consists of two elements: multimodal risk distribution
strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs.
The heuristic-induced search strategy identiﬁes two types
of prompts: the understanding-enhancing prompt, which
helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of afﬁrmative outputs over refusals, enabling a successful jailbreak
attack. HIMRD achieves an average attack success rate
(ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLLMs.
HIMRD reveals cross-modal security vulnerabilities in current MLLMs and underscores the imperative for developing
defensive strategies to mitigate such emerging risks. Code
is available at https://github.com/MaTengSYSU/HIMRDjailbreak.
```



## NeurIPS-2025-towards-building-modelprompt-transferable-attackers-against-large-vision-language-models-Paper-Conference.pdf

```
Towards Building Model/Prompt-Transferable
Attackers against Large Vision-Language Models
Xiaowen Cai1∗, Daizong Liu 2∗†, Xiaoye Qu 1, Xiang Fang 3, Jianfeng Dong 4,
Keke Tang5, Pan Zhou 1‡, Lichao Sun 6, Wei Hu 7‡
1Huazhong University of Science and Technology 2Wuhan University
3Nanyang Technological University 4Zhejiang Gongshang University 5Guangzhou University
6Lehigh University 7Peking University
{xwcai,xiaoye,panzhou}@hust.edu.cn, daizongliu@whu.edu.cn,xfang9508@gmail.com
djf@zjgsu.edu.cn,tangbohutbh@gmail.com,lis221@lehigh.edu,forhuwei@pku.edu.cn
Abstract
Although Large Vision-Language Models (LVLMs) exhibit impressive multimodal
capabilities, their vulnerability to adversarial examples has raised serious security
concerns. Existing LVLM attackers simply optimize adversarial images that easily
overfit a certain model/prompt, making them ineffective once they are transferred
to attack a different model/prompt. Motivated by this research gap, this paper aims
to develop a more powerful attack that is transferable to black-box LVLM models
of different structures and task-aware prompts of different semantics. Specifically, we introduce a new perspective of information theory to investigate LVLMs’
transferable characteristics by exploring the relative dependence between outputs
of the LVLM model and input adversarial samples. Our empirical observations
suggest that enlarging/decreasing the mutual information between outputs and the
disentangled adversarial/benign patterns of input images helps to generate more
agnostic perturbations for misleading LVLMs’ perception with better transferability.
In particular, we formulate the complicated calculation of information gain as an
estimation problem and incorporate such informative constraints into the adversarial learning process. Extensive experiments on various LVLM models/prompts
demonstrate our significant transfer-attack performance.
1 Introduction
Large Vision-Language Models (LVLMs) have garnered significant atten

>>> ABSTRACT:
Although Large Vision-Language Models (LVLMs) exhibit impressive multimodal
capabilities, their vulnerability to adversarial examples has raised serious security
concerns. Existing LVLM attackers simply optimize adversarial images that easily
overfit a certain model/prompt, making them ineffective once they are transferred
to attack a different model/prompt. Motivated by this research gap, this paper aims
to develop a more powerful attack that is transferable to black-box LVLM models
of different structures and task-aware prompts of different semantics. Specifically, we introduce a new perspective of information theory to investigate LVLMs’
transferable characteristics by exploring the relative dependence between outputs
of the LVLM model and input adversarial samples. Our empirical observations
suggest that enlarging/decreasing the mutual information between outputs and the
disentangled adversarial/benign patterns of input images helps to generate more
agnostic perturbations for misleading LVLMs’ perception with better transferability.
In particular, we formulate the complicated calculation of information gain as an
estimation problem and incorporate such informative constraints into the adversarial learning process. Extensive experiments on various LVLM models/prompts
demonstrate our significant transfer-attack performance.
```



## Nie_V-Attack_Targeting_Disentangled_Value_Features_for_Controllable_Adversarial_Attacks_on_CVPR_2026_paper.pdf

```
V-Attack: Targeting Disentangled Value Features for Controllable Adversarial
Attacks on LVLMs
Sen Nie1,2,Jie Zhang 1,2*,Jianxin Yan 3,Shiguang Shan 1,2,Xilin Chen 1,2
1State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences
2University of Chinese Academy of Sciences 3Zhejiang University
sen.nie@vipl.ict.ac.cn,{zhangjie, sgshan, xlchen}@ict.ac.cn, yanjianxin@zju.edu.cn
8980
 
ORI
Change “horse” 
to “donkey”
Change “dog” 
to “tiger”
Please describe this image in one sentence.
 Please identify dog's neighbor by bio-traits.
Controllable 
Adversarial Attack
Baselines
V-Attack
horse→donkey
dog→tiger
GPT-o3 (Thought for 7 seconds): Based on the characteristics 
of the creature next to the horse ... build typical of a tiger. 
GPT-o3 (Thought for 15 seconds) Based on its biology ... 
standing beside the dog is most consistent with a donkey. 
GPT-o3 (Thought for 3 seconds) A brown horse and a yellow 
dog are standing together in a grassy field ...
 Please identify horse's neighbor by bio-traits.
Figure 1. By leveraging rich, disentangled value features instead of commonly used entangled patch features, V-Attack enables precise
local semantic manipulations that expose the true vulnerabilities of LVLMs, overcoming the imprecise strategies of existing baselines.
Abstract
Adversarial attacks have evolved from simply disrupting
predictions on conventional task-specific models to the more
complex goal of manipulating image semantics on Large
Vision-Language Models (LVLMs). However, existing methods struggle with controllability and fail to precisely manipulate the semantics of specific concepts in the image.
We attribute this limitation to semantic entanglement in
the patch-token representations on which adversarial attacks typically operate: global context aggregated by selfattention in the vision encoder dominates patch features,
making them unreliable handles for precise local semantic manipulation. Our systematic investigation reveals a

>>> ABSTRACT:
Adversarial attacks have evolved from simply disrupting
predictions on conventional task-specific models to the more
complex goal of manipulating image semantics on Large
Vision-Language Models (LVLMs). However, existing methods struggle with controllability and fail to precisely manipulate the semantics of specific concepts in the image.
We attribute this limitation to semantic entanglement in
the patch-token representations on which adversarial attacks typically operate: global context aggregated by selfattention in the vision encoder dominates patch features,
making them unreliable handles for precise local semantic manipulation. Our systematic investigation reveals a
key insight: value features (V) computed within the transformer attention block serve as much more precise handles for manipulation. We show that V suppresses globalcontext channels, allowing it to retain high-entropy, disentangled local semantic information. Building on this discovery, we proposeV-Attack, a novel method designed for
precise local semantic attacks. V-Attack targets the value
features and introduces two core components: (1) a Self*Corresponding author.
Value Enhancement module to refine V’s intrinsic semantic
richness, and (2) a Text-Guided Value Manipulation module that leverages text prompts to locate the source concept and optimize it toward a target concept. By bypassing the entangled patch features, V-Attack achieves highly
effective semantic control. Extensive experiments across
diverse LVLMs, including LLaVA, InternVL, DeepseekVL,
and GPT-4o, show that V-Attack improves the attack success rate by an average of 36% over state-of-the-art methods, exposing critical vulnerabilities in modern visuallanguage understanding. Our code and data are available
at: https://github.com/Summu77/V-Attack.
```



## PointLLM  Empowering Large Language Models to Understand Point Clouds.pdf

```
PointLLM: Empowering Large Language Models
to Understand Point Clouds
Runsen Xu1,2, Xiaolong Wang3, Tai Wang2†, Yilun Chen2, Jiangmiao Pang2†,
and Dahua Lin1,2,4
1 The Chinese University of Hong Kong
2 Shanghai AI Laboratory
3 Zhejiang University
4 Centre for Perceptual and Interactive Intelligence
Abstract. The unprecedented advancements in Large Language Models (LLMs) have shown a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This
paper introduces PointLLM, a preliminary effort to fill this gap, empowering LLMs to understand point clouds and offering a new avenue beyond 2D data. PointLLM understands colored object point clouds with
human instructions and generates contextually appropriate responses,
illustrating its grasp of point clouds and common sense. Specifically,
it leverages a point cloud encoder with a powerful LLM to effectively
fuse geometric, appearance, and linguistic information. To overcome the
scarcity of point-text instruction following data, we developed an automated data generation pipeline, collecting a large-scale dataset of more
than 730K samples with 660K different objects, which facilitates the
adoption of the two-stage training strategy prevalent in MLLM development. Additionally, we address the absence of appropriate benchmarks
and the limitations of current evaluation metrics by proposing two novel
benchmarks: Generative 3D Object Classification and 3D Object Captioning, which are supported by new, comprehensive evaluation metrics derived from human and GPT analyses. Through exploring various training strategies, we develop PointLLM, significantly surpassing
2D and 3D baselines, with a notable achievement in human-evaluated
object captioning tasks where it surpasses human annotators in over
50% of the samples. Codes, datasets, and benchmarks are available at
https://github.com/OpenRobotLab/PointLLM.
Keywords: Multi-Modal LLM · 3D Understanding · Point Cloud
1 Introduction
Recent

>>> ABSTRACT:
. The unprecedented advancements in Large Language Models (LLMs) have shown a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This
paper introduces PointLLM, a preliminary effort to fill this gap, empowering LLMs to understand point clouds and offering a new avenue beyond 2D data. PointLLM understands colored object point clouds with
human instructions and generates contextually appropriate responses,
illustrating its grasp of point clouds and common sense. Specifically,
it leverages a point cloud encoder with a powerful LLM to effectively
fuse geometric, appearance, and linguistic information. To overcome the
scarcity of point-text instruction following data, we developed an automated data generation pipeline, collecting a large-scale dataset of more
than 730K samples with 660K different objects, which facilitates the
adoption of the two-stage training strategy prevalent in MLLM development. Additionally, we address the absence of appropriate benchmarks
and the limitations of current evaluation metrics by proposing two novel
benchmarks: Generative 3D Object Classification and 3D Object Captioning, which are supported by new, comprehensive evaluation metrics derived from human and GPT analyses. Through exploring various training strategies, we develop PointLLM, significantly surpassing
2D and 3D baselines, with a notable achievement in human-evaluated
object captioning tasks where it surpasses human annotators in over
50% of the samples. Codes, datasets, and benchmarks are available at
https://github.com/OpenRobotLab/PointLLM.
```



## Towards Robust and Secure Embodied AI A Survey on.pdf

```
Towards Robust and Secure Embodied AI: A Survey on
Vulnerabilities and Attacks
WENPENG XING, Zhejiang University, Hangzhou, China and Binjiang Institute of Zhejiang University,
Hangzhou, China
MINGHAO LI, College of Big Data and Software Engineering, Chongqing University, Chongqing, China
MOHAN LI, Cyberspace Institute of Advanced Technology, Guangzhou University, Guangzhou, China
MENG HAN, Zhejiang University, Hangzhou, China, Binjiang Institute of Zhejiang University, Hangzhou,
China, and GenTel.io, hangzhou, China
Embodied AI systems, integrating Large Vision-Language Models (LVLMs) and Large Language Models (LLMs)
withphysicalactuatorsandsensors,faceuniquerobustnessandsecuritychallengesstemmingfromthecomplex
interplay between perception, cognition, and actuation in real-world environments. This survey provides a
systematic analysis of these vulnerabilities and associated attack surfaces. We propose a tripartite vulnerability
taxonomy comprising foundational, integration, and contextual risks. Foundational vulnerabilities arise from
inherent limitations in current AI architectures and training paradigms; Integration vulnerabilities emerge
from the composition of cyber-physical components; And contextual vulnerabilities stem from dynamic
physical environments and deployment conditions. Correspondingly, we present a comprehensive attack
taxonomy that encompasses foundational attacks on LLMs/LVLMs (including logits-based, optimization-based,
prompt-based, and cross-modality attacks), integration-level cybersecurity threats (such as man-in-the-middle,
firmware, side-channel, and supply chain attacks), and contextual attacks (primarily sensor spoofing across
multiple modalities). We further examine representative failure modes of the cognitive core, review existing
evaluation methodologies and benchmarks, and synthesize a multi-layer defense framework that integrates
perceptual redundancy, runtime monitoring, and hardware-enforced safety mechanisms. This work offers a
```



## Wang_TAPT_Test-Time_Adversarial_Prompt_Tuning_for_Robust_Inference_in_Vision-Language_CVPR_2025_paper.pdf

```
TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in
Vision-Language Models
Xin Wang1, Kai Chen1, Jiaming Zhang2, Jingjing Chen1, Xingjun Ma1*
1Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University
2Hong Kong University of Science and Technology
Abstract
Large pre-trained Vision-Language Models (VLMs) such
as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of
CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant
safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time
Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks.
TAPT is a test-time defense method that learns defensive
bimodal (textual and visual) prompts to robustify the inference process of CLIP . Specifically, it is an unsupervised
method that optimizes the defensive prompts for each test
sample by minimizing a multi-view entropy and aligning
adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that
it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while
largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning
methods across various backbones, achieving an average
robustness improvement of at least 36.6%. Code is available at https://github.com/xinwong/TAPT.
1. Introduction
Vision-Language Models (VLMs) pre-trained on largescale datasets of image-text pairs have emerged as powerful
backbones for numerous applications, including computer
vision [17, 34, 57], medical image analysis [15, 51], and
robotics [3, 18, 39]. Despite these advancements, studies
indicate th

>>> ABSTRACT:
Large pre-trained Vision-Language Models (VLMs) such
as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of
CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant
safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time
Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks.
TAPT is a test-time defense method that learns defensive
bimodal (textual and visual) prompts to robustify the inference process of CLIP . Specifically, it is an unsupervised
method that optimizes the defensive prompts for each test
sample by minimizing a multi-view entropy and aligning
adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that
it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while
largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning
methods across various backbones, achieving an average
robustness improvement of at least 36.6%. Code is available at https://github.com/xinwong/TAPT.
```



## Xie_Chain_of_Attack_On_the_Robustness_of_Vision-Language_Models_Against_CVPR_2025_paper.pdf

```
Chain of Attack: On the Robustness of Vision-Language Models Against
Transfer-Based Adversarial Attacks
Peng Xie*, Yequan Bie*, Jianda Mao, Yangqiu Song, Yang Wang, Hao ChenB, Kani ChenB
The Hong Kong University of Science and Technology, Hong Kong, China
{pxieaf, ybie}@connect.ust.hk, {jhc, makchen}@ust.hk
Abstract
Pre-trained vision-language models (VLMs) have showcased remarkable performance in image and natural language understanding, such as image captioning and response generation. As the practical applications of VLMs
become increasingly widespread, their potential safety and
robustness issues raise concerns that adversaries may
evade the system and cause these models to generate toxic
content through malicious attacks. Therefore, evaluating
the robustness of open-source VLMs against adversarial attacks has garnered growing attention, with transfer-based
attacks as a representative black-box attacking strategy.
However, most existing transfer-based attacks neglect the
importance of the semantic correlations between vision and
text modalities, leading to sub-optimal adversarial example
generation and attack performance. To address this issue,
we present Chain of Attack (CoA) 1, which iteratively enhances the generation of adversarial examples based on the
multi-modal semantic update using a series of intermediate attacking steps, achieving superior adversarial transferability and efficiency. A unified attack success rate computing method is further proposed for automatic evasion evaluation. Extensive experiments conducted under the most
realistic and high-stakes scenario, demonstrate that our attacking strategy is able to effectively mislead models to generate targeted responses using only black-box attacks without any knowledge of the victim models. The comprehensive
robustness evaluation in our paper provides insight into the
vulnerabilities of VLMs and offers a reference for the safety
considerations of future model developments.
1. Introduction
Vision-lang

>>> ABSTRACT:
Pre-trained vision-language models (VLMs) have showcased remarkable performance in image and natural language understanding, such as image captioning and response generation. As the practical applications of VLMs
become increasingly widespread, their potential safety and
robustness issues raise concerns that adversaries may
evade the system and cause these models to generate toxic
content through malicious attacks. Therefore, evaluating
the robustness of open-source VLMs against adversarial attacks has garnered growing attention, with transfer-based
attacks as a representative black-box attacking strategy.
However, most existing transfer-based attacks neglect the
importance of the semantic correlations between vision and
text modalities, leading to sub-optimal adversarial example
generation and attack performance. To address this issue,
we present Chain of Attack (CoA) 1, which iteratively enhances the generation of adversarial examples based on the
multi-modal semantic update using a series of intermediate attacking steps, achieving superior adversarial transferability and efficiency. A unified attack success rate computing method is further proposed for automatic evasion evaluation. Extensive experiments conducted under the most
realistic and high-stakes scenario, demonstrate that our attacking strategy is able to effectively mislead models to generate targeted responses using only black-box attacks without any knowledge of the victim models. The comprehensive
robustness evaluation in our paper provides insight into the
vulnerabilities of VLMs and offers a reference for the safety
considerations of future model developments.
```



## Zhao_Jailbreaking_Multimodal_Large_Language_Models_via_Shuffle_Inconsistency_ICCV_2025_paper.pdf

```
Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency
Shiji Zhao, Ranjie Duan *, Fengxiang Wang *, Chi Chen, Caixin Kang,
Shouwei Ruan, Jialing Tao *, YueFeng Chen *, Hui Xue *, Xingxing Wei †
Institute of Artificial Intelligence, State Key Laboratory of Virtual Reality Technology and Systems,
Beihang University, Beijing, China
{zhaoshiji123, xxwei}@buaa.edu.cn
Abstract
Multimodal Large Language Models (MLLMs) have
achieved impressive performance and have been put into
practical use in commercial applications, but they still have
potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety
mechanisms and discover MLLMs’ potential risks. Existing
MLLMs’ jailbreak methods often bypass the model’s safety
mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving
some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research,
we empirically find that there exists a Shuffle Inconsistency
between MLLMs’ comprehension ability and safety ability
for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand
the shuffled harmful text-image instructions well. However,
they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image
jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization
method to select the most harmful shuffled inputs based
on the feedback of the toxic judge model. A series of experiments show that SI-Attack can effectively improve the
attack’s performance on three benchmarks for both opensource and closed-source MLLMs. Warning: This paper
contains examples of harmful texts and images, and reader
discretion is recom

>>> ABSTRACT:
Multimodal Large Language Models (MLLMs) have
achieved impressive performance and have been put into
practical use in commercial applications, but they still have
potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety
mechanisms and discover MLLMs’ potential risks. Existing
MLLMs’ jailbreak methods often bypass the model’s safety
mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving
some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research,
we empirically find that there exists a Shuffle Inconsistency
between MLLMs’ comprehension ability and safety ability
for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand
the shuffled harmful text-image instructions well. However,
they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image
jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization
method to select the most harmful shuffled inputs based
on the feedback of the toxic judge model. A series of experiments show that SI-Attack can effectively improve the
attack’s performance on three benchmarks for both opensource and closed-source MLLMs. Warning: This paper
contains examples of harmful texts and images, and reader
discretion is recommended.
```



## 【内部机制】Reference Attack A New Cross-Modal Jailbreaking Attack against.pdf

```
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (V olume 1: Long Papers), pages 17860–17881
July 2-7, 2026 ©2026 Association for Computational Linguistics
Reference Attack: A New Cross-Modal Jailbreaking Attack against
Multimodal Large Language Models
Yulong Wang and Yifei Fu and Jiayi Gao
Beijing University of Posts and Telecommunications, Beijing, China
{wyl, yifei_fu, jiayi.gao}@bupt.edu.cn
Abstract
Red team testing, an effective proactive method
for evaluating the security of multimodal large
language models (MLLMs), requires an expanding toolkit alongside the development of
MLLM safeguards. We propose the Reference
Attack, a powerful tool for red team testing
against MLLMs. The Reference Attack is a
reference-guided cross-modal jailbreak method
that enhances existing prompt-to-image injection attacks by exploiting MLLMs’ semantic reconstruction capabilities. Our method embeds
malicious prompts in non-text modalities (e.g.,
images, spreadsheets) and constructs recursive
symbolic references in text, enabling MLLMs
to gradually recover and generate harmful content through layered reference resolution. The
attack introduces a new vector that circumvents
conventional content moderation by exploiting
MLLMs’ lack of security checks during crossmodal reference resolution. We evaluate the
Reference Attack on leading MLLMs, including ChatGPT, Gemini, Claude, and the widely
used open-source LLaMA model, and achieved
an attack success rate of over 93% across all
tested models. Compared to state-of-the-art
attacks, Reference Attack achieves higher success rates than all baselines under identical evaluation, with a maximum gain of 70.8%. Our
study reveals a critical gap in MLLM security
and highlights the need for strict security auditing of cross-modal interactions in future content moderation.
1 Introduction
Large Language Model (LLM)-powered systems,
such as ChatGPT, have become essential tools for
accessing information and knowledg

>>> ABSTRACT:
Red team testing, an effective proactive method
for evaluating the security of multimodal large
language models (MLLMs), requires an expanding toolkit alongside the development of
MLLM safeguards. We propose the Reference
Attack, a powerful tool for red team testing
against MLLMs. The Reference Attack is a
reference-guided cross-modal jailbreak method
that enhances existing prompt-to-image injection attacks by exploiting MLLMs’ semantic reconstruction capabilities. Our method embeds
malicious prompts in non-text modalities (e.g.,
images, spreadsheets) and constructs recursive
symbolic references in text, enabling MLLMs
to gradually recover and generate harmful content through layered reference resolution. The
attack introduces a new vector that circumvents
conventional content moderation by exploiting
MLLMs’ lack of security checks during crossmodal reference resolution. We evaluate the
Reference Attack on leading MLLMs, including ChatGPT, Gemini, Claude, and the widely
used open-source LLaMA model, and achieved
an attack success rate of over 93% across all
tested models. Compared to state-of-the-art
attacks, Reference Attack achieves higher success rates than all baselines under identical evaluation, with a maximum gain of 70.8%. Our
study reveals a critical gap in MLLM security
and highlights the need for strict security auditing of cross-modal interactions in future content moderation.
```



## 【内部机制】Seeing No Evil Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking.pdf

```
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (V olume 1: Long Papers), pages 18304–18323
July 2-7, 2026 ©2026 Association for Computational Linguistics
Seeing No Evil: Blinding Large Vision-Language Models to Safety
Instructions via Adversarial Attention Hijacking
Jingru Li♠ Wei Ren♠* Tianqing Zhu♣
♠ China University of Geosciences, Wuhan
♣City University of Macau
✉:{jingruli810, weirencs}@cug.edu.cn,tqzhu@cityu.edu.mo
Abstract
Large Vision-Language Models (LVLMs) rely
on attention-based retrieval of safety instructions to maintain alignment during generation.
Existing attacks typically optimize image perturbations to maximize harmful output likelihood, but suffer from slow convergence due
to gradient conflict between adversarial objectives and the model’s safety-retrieval mechanism. We proposeAttention-Guided Visual
Jailbreaking, which circumvents rather than
overpowers safety alignment by directly manipulating attention patterns. Our method introduces two simple auxiliary objectives: (1)
suppressing attention to alignment-relevant prefix tokens and (2) anchoring generation on adversarial image features. This simple yet effective push-pull formulation reduces gradient
conflict by 45% and achieves94.4% attack
success rateon Qwen-VL (vs. 68.8% baseline)
with 40% fewer iterations. At tighter perturbation budgets (ϵ=8/255), we maintain 59.0%
ASR compared to 45.7% for standard methods.
Mechanistic analysis reveals a failure mode we
term safety blindness: successful attacks suppress system-prompt attention by 80%, causing models to generate harmful content not by
overriding safety rules, but by failing to retrieve
them.
Code:/githubgithub.com/Landsayy/AttentionJailbreak
1 Introduction
Large Vision-Language Models (LVLMs) are increasingly deployed in safety-critical applications,
including AI assistants, content moderation, and education (OpenAI et al., 2024; Liu et al., 2024a; Bai
et al., 2023; Zhu et al., 2024). Their safety al

>>> ABSTRACT:
Large Vision-Language Models (LVLMs) rely
on attention-based retrieval of safety instructions to maintain alignment during generation.
Existing attacks typically optimize image perturbations to maximize harmful output likelihood, but suffer from slow convergence due
to gradient conflict between adversarial objectives and the model’s safety-retrieval mechanism. We proposeAttention-Guided Visual
Jailbreaking, which circumvents rather than
overpowers safety alignment by directly manipulating attention patterns. Our method introduces two simple auxiliary objectives: (1)
suppressing attention to alignment-relevant prefix tokens and (2) anchoring generation on adversarial image features. This simple yet effective push-pull formulation reduces gradient
conflict by 45% and achieves94.4% attack
success rateon Qwen-VL (vs. 68.8% baseline)
with 40% fewer iterations. At tighter perturbation budgets (ϵ=8/255), we maintain 59.0%
ASR compared to 45.7% for standard methods.
Mechanistic analysis reveals a failure mode we
term safety blindness: successful attacks suppress system-prompt attention by 80%, causing models to generate harmful content not by
overriding safety rules, but by failing to retrieve
them.
Code:/githubgithub.com/Landsayy/AttentionJailbreak
```



## 【内部机制】TOWARD UNIVERSAL AND TRANSFERABLE.pdf

```
Published as a conference paper at ICLR 2026
TOWARD UNIVERSAL AND TRANSFERABLE
JAILBREAK ATTACKS ON VISION -L ANGUAGE
MODELS
Kaiyuan Cui1 Yige Li2 Yutao Wu3 Xingjun Ma4 Sarah Erfani1
Christopher Leckie1 Hanxun Huang1
1School of Computing and Information Systems, The University of Melbourne, Australia
2School of Computing and Information Systems, Singapore Management University, Singapore
3School of Information Technology, Deakin University, Australia
4Institute of Trustworthy Embodied AI, Fudan University, China
{kaiyuan.cui}@student.unimelb.edu.au;{yigeli}@smu.edu.sg;
{oscar.w}@deakin.edu.au;{xingjunma}@fudan.edu.cn;
{sarah.erfani,caleckie,hanxun}@unimelb.edu.au.
ABSTRACT
Vision–language models (VLMs) extend large language models (LLMs) with vision encoders, enabling text generation conditioned on both images and text.
However, this multimodal integration expands the attack surface by exposing the
model to image-based jailbreaks crafted to induce harmful responses. Existing
gradient-based jailbreak methods transfer poorly, as adversarial patterns overfit to
a single white-box surrogate and fail to generalise to black-box models. In this
work, we propose Universal and transferable jailbreak (UltraBreak), a framework that constrains adversarial patterns through transformations and regularisation in the vision space, while relaxing textual targets through semantic-based
objectives. By defining its loss in the textual embedding space of the target LLM,
UltraBreak discovers universal adversarial patterns that generalise across diverse
jailbreak objectives. This combination of vision-level regularisation and semantically guided textual supervision mitigates surrogate overfitting and enables strong
transferability across both models and attack targets. Extensive experiments show
that UltraBreak consistently outperforms prior jailbreak methods. Further analysis reveals why earlier approaches fail to transfer, highlighting that smoothing
the loss landscape via semantic obje

>>> ABSTRACT:
Vision–language models (VLMs) extend large language models (LLMs) with vision encoders, enabling text generation conditioned on both images and text.
However, this multimodal integration expands the attack surface by exposing the
model to image-based jailbreaks crafted to induce harmful responses. Existing
gradient-based jailbreak methods transfer poorly, as adversarial patterns overfit to
a single white-box surrogate and fail to generalise to black-box models. In this
work, we propose Universal and transferable jailbreak (UltraBreak), a framework that constrains adversarial patterns through transformations and regularisation in the vision space, while relaxing textual targets through semantic-based
objectives. By defining its loss in the textual embedding space of the target LLM,
UltraBreak discovers universal adversarial patterns that generalise across diverse
jailbreak objectives. This combination of vision-level regularisation and semantically guided textual supervision mitigates surrogate overfitting and enables strong
transferability across both models and attack targets. Extensive experiments show
that UltraBreak consistently outperforms prior jailbreak methods. Further analysis reveals why earlier approaches fail to transfer, highlighting that smoothing
the loss landscape via semantic objectives is crucial for enabling universal and
transferable jailbreaks. The code is publicly available in our GitHub repository.
1 I NTRODUCTION
Recent advances in large Vision–Language Models (VLMs) (Google et al., 2023; OpenAI, 2024;
Bai et al., 2025) have enabled deployment in safety-critical domains such as healthcare (He et al.,
2025) and autonomous driving (Zhou et al., 2024), where failures may cause severe consequences.
VLMs typically integrate a vision encoder with a large language model (LLM). Despite advances
in safety alignment (Bai et al., 2022; Ouyang et al., 2022), LLMs remain vulnerable to jailbreak
attacks, where adversaries craft textual inputs that bypass safety mechanisms and elicit harmful
responses (Zou et al., 2023; Carlini et al., 2023; Ma et al., 2025). Incorporating additional modalities
further expands the attack surface: unlike discrete text, images are continuous and high-dimensional,
offering a much broader space for adversarial manipulation.
Jailbreak attacks against VLMs fall into two main categories: manually designed jailbreaks (Ma
et al., 2024; Gong et al., 2025) and gradient-based jailbreaks (Wang et al., 2024c; Qi et al., 2024;
Niu et al., 2024). Manually designed approaches conceal malicious intent within elaborate scenarios, tricking the model into harmful responses in ways reminiscent of human deception. However,
each harmful target requires a target-specific image input, limiting their cross-target transferability.
Gradient-based attacks optimise inputs using model gradients, allowing adversaries to drive jailbroken behaviour across multiple harmful objectives. Yet these methods tend to overfit to a surrogate
1
Publ
```



## 【每日推荐】Jailbreaking Vision-Language Models via Dissonance-Guided Su!x.pdf

```
Jailbreaking Vision-Language Models via Dissonance-Guided Su!x
Optimization and Image–Phrase Injection
Jiacheng Pi, Zhiguo Y ang, Xingxing Huang, Dongsheng Xu, Ruizhi Zhong, Wenjie Ruan∗
University of Science and Technology of China, School of Computer Science and Technology
{pijch, zhiguoyang, huangxingxing, xds2327659729 }@mail.ustc.edu.cn, rwjie@ustc.edu.cn
Abstract
The integration of vision and language in Vision-Language
Models (VLMs), while enabling multimodal capabilities,
inherently expands their attack surface. Among existing
white-box jailbreak methods, su!x-optimization-based approaches often rely on gradient approximations over discrete token spaces, yielding insu!cient guidance and causing optimization to stagnate in local optima, while imageperturbation-based ones frequently exhibit poor crossmodel transferability. In this work, we introduce DGSIP,a
Dissonance-Guided Su!x Optimization and Image–Phrase
Injection framework. DGSIP leverages predictive dissonance between the target model and an unaligned model
to identify tokens suppressed by safety alignment, using
them as a more e”ective signal than gradient-based cues
for su!x optimization. It further reinforces the attack by
jointly optimizing the content and presentation of phrase
embedded in images to leverage VLMs’ cross-modal sensitivity. Our extensive experiments demonstrate that DGSIP
outperforms prior baselines across multiple safety benchmarks and a range of open-source VLMs (e.g., MiniGPT-4,
InstructBlip and LLaVA). Notably, compared to baselines,
our method exhibits much stronger transferability to commercial black-box VLMs, such as GPT-4o-Mini, Gemini 2.0
Flash and Qwen 2.5-VL. Based upon DGSIP , we empirically reveal critical vulnerabilities in the safeguard mechanisms of current VLMs, highlighting the need for more
robust defense strategies. The implementation is available
on https://github.com/Trusted-LLM/DGSIP .
1. Introduction
Vision-Language Models (VLMs), which couple large pretrained

>>> ABSTRACT:
The integration of vision and language in Vision-Language
Models (VLMs), while enabling multimodal capabilities,
inherently expands their attack surface. Among existing
white-box jailbreak methods, su!x-optimization-based approaches often rely on gradient approximations over discrete token spaces, yielding insu!cient guidance and causing optimization to stagnate in local optima, while imageperturbation-based ones frequently exhibit poor crossmodel transferability. In this work, we introduce DGSIP,a
Dissonance-Guided Su!x Optimization and Image–Phrase
Injection framework. DGSIP leverages predictive dissonance between the target model and an unaligned model
to identify tokens suppressed by safety alignment, using
them as a more e”ective signal than gradient-based cues
for su!x optimization. It further reinforces the attack by
jointly optimizing the content and presentation of phrase
embedded in images to leverage VLMs’ cross-modal sensitivity. Our extensive experiments demonstrate that DGSIP
outperforms prior baselines across multiple safety benchmarks and a range of open-source VLMs (e.g., MiniGPT-4,
InstructBlip and LLaVA). Notably, compared to baselines,
our method exhibits much stronger transferability to commercial black-box VLMs, such as GPT-4o-Mini, Gemini 2.0
Flash and Qwen 2.5-VL. Based upon DGSIP , we empirically reveal critical vulnerabilities in the safeguard mechanisms of current VLMs, highlighting the need for more
robust defense strategies. The implementation is available
on https://github.com/Trusted-LLM/DGSIP .
```



## 【每日推荐】Towards Highly Transferable Vision-Language Attack via Semantic-Augmented.pdf

```
Towards Highly Transferable Vision-Language Attack via Semantic-Augmented
Dynamic Contrastive Interaction
Yuanbo Li1, Tianyang Xu1, Cong Hu1, Tao Zhou1, Xiao-Jun Wu1*, Josef Kittler2
1School of Artificial Intelligence and Computer Science, Jiangnan University
2Centre for Vision, Speech and Signal Processing (CVSSP), University of Surrey
liyuanbo12138@163.com,{tianyang.xu, conghu, taozhou.ai, wu xiaojun}@jiangnan.edu.cn
j.kittler@surrey.ac.uk
Abstract
With the rapid advancement and widespread application of
vision-language pre-training (VLP) models, their vulnerability to adversarial attacks has become a critical concern. In general, the adversarial examples can typically
be designed to exhibit transferable power, attacking not
only different models but also across diverse tasks. However, existing attacks on language-vision models mainly
rely on static cross-modal interactions and focus solely
on disrupting positive image-text pairs, resulting in limited cross-modal disruption and poor transferability. To
address this issue, we propose a Semantic-Augmented Dynamic Contrastive Attack (SADCA) that enhances adversarial transferability through progressive and semantically
guided perturbation. SADCA progressively disrupts crossmodal alignment through dynamic interactions between adversarial images and texts. This is accomplished by SADCA
establishing a contrastive learning mechanism involving
adversarial, positive and negative samples, to reinforce
the semantic inconsistency of the obtained perturbations.
Moreover, we empirically find that input transformations
commonly used in traditional transfer-based attacks also
benefit VLPs, which motivates a semantic augmentation
module that increases the diversity and generalization of
adversarial examples. Extensive experiments on multiple
datasets and models demonstrate that SADCA significantly
improves adversarial transferability and consistently surpasses state-of-the-art methods. The code is released at
https://github.com/LiY

>>> ABSTRACT:
With the rapid advancement and widespread application of
vision-language pre-training (VLP) models, their vulnerability to adversarial attacks has become a critical concern. In general, the adversarial examples can typically
be designed to exhibit transferable power, attacking not
only different models but also across diverse tasks. However, existing attacks on language-vision models mainly
rely on static cross-modal interactions and focus solely
on disrupting positive image-text pairs, resulting in limited cross-modal disruption and poor transferability. To
address this issue, we propose a Semantic-Augmented Dynamic Contrastive Attack (SADCA) that enhances adversarial transferability through progressive and semantically
guided perturbation. SADCA progressively disrupts crossmodal alignment through dynamic interactions between adversarial images and texts. This is accomplished by SADCA
establishing a contrastive learning mechanism involving
adversarial, positive and negative samples, to reinforce
the semantic inconsistency of the obtained perturbations.
Moreover, we empirically find that input transformations
commonly used in traditional transfer-based attacks also
benefit VLPs, which motivates a semantic augmentation
module that increases the diversity and generalization of
adversarial examples. Extensive experiments on multiple
datasets and models demonstrate that SADCA significantly
improves adversarial transferability and consistently surpasses state-of-the-art methods. The code is released at
https://github.com/LiYuanBoJNU/SADCA.
```



## 【每日推荐】Transform to Transfer Boosting Adversarial Attack Transferability on.pdf

```
Transform to Transfer: Boosting Adversarial Attack Transferability on
Vision-Language Pre-training Models
Yang Li1,2 Jia-Li Yin1,2* Luojun Lin2 Wei Lin3
1Fujian Province Key Laboratory of Information Security and Network Systems, Fuzhou, China
2Fuzhou University 3Fujian University of Technology
{241027138, jlyin, ljlin}@fzu.edu.cn, wlin@fjut.edu.cn
Abstract
Vision-Language Pre-training (VLP) models, while achieving state-of-the-art performance on various multimodal
tasks, exhibit significant vulnerability to multimodal adversarial examples. In black-box attack scenarios of VLP models, a key challenge lies in the limited transferability of these
adversarial examples. Existing methods to enhance transferability often suffer from an excessive dependence on the
source model and a reliance on limited and fixed transformation techniques. To overcome these limitations, we propose a novel Transform to Transfer Attack (TTA) method.
Our approach introduces a learnable transformation mechanism that adaptively selects optimal combinations of transformations to maximize input diversity, and incorporates integrated gradients to mitigate over-reliance on the source
model, thereby refining the attack optimization process. Extensive experiments demonstrate that TTA achieves outstanding attack performance in downstream tasks, outperforming current state-of-the-art attack methods across different VLP architectures.
1. Introduction
Vision-Language Pre-training (VLP) models have demonstrated strong performance across various downstream
vision-language tasks—such as image-text retrieval [4, 44],
image captioning [16], visual grounding [6], and visual
entailment [34]—by learning cross-modal representations
that bridge the semantic gap between visual and linguistic
modalities. Despite these successes, recent works have revealed that VLP models remain vulnerable to multimodal
adversarial examples [11, 27, 33, 40]. Moreover, adversarial examples generated for one downstream task can be
trans

>>> ABSTRACT:
Vision-Language Pre-training (VLP) models, while achieving state-of-the-art performance on various multimodal
tasks, exhibit significant vulnerability to multimodal adversarial examples. In black-box attack scenarios of VLP models, a key challenge lies in the limited transferability of these
adversarial examples. Existing methods to enhance transferability often suffer from an excessive dependence on the
source model and a reliance on limited and fixed transformation techniques. To overcome these limitations, we propose a novel Transform to Transfer Attack (TTA) method.
Our approach introduces a learnable transformation mechanism that adaptively selects optimal combinations of transformations to maximize input diversity, and incorporates integrated gradients to mitigate over-reliance on the source
model, thereby refining the attack optimization process. Extensive experiments demonstrate that TTA achieves outstanding attack performance in downstream tasks, outperforming current state-of-the-art attack methods across different VLP architectures.
```



## 【注意力思路获取】Attention Hijacking Response Manipulation Across Queries in Vision-Language Models.pdf

```
Attention Hijacking: Response Manipulation Across
Queries in Vision-Language Models
Zhiqiang Wang1∗ Dongrui Liu2∗ Yan Li1 Zonghao Ying3 Wei Xue1
Wenhan Luo1 Yike Guo1
1 Hong Kong University of Science and Technology 2 Shanghai Jiao Tong University
3 Beihang University
zwangmk@connect.ust.hk
Abstract
Existing adversarial attacks on vision-language models (VLMs) can steer model
outputs toward attacker-specified target responses, but their effectiveness often
degrades when the same perturbed input is paired with different textual queries.
This paper studies cross-query response manipulation, where a single adversarial
example is expected to remain effective across diverse user queries. We first
analyze the limitations of existing attacks and find that successful transfer is closely
associated with preserving an image-dominant attention pattern during response
generation. Motivated by the observation, we proposeAttention Hijacking, a
novel adversarial attack that explicitly steers internal attention distributions toward
a persistent image-dominant pattern. By amplifying the influence of visual tokens
on target response tokens while suppressing the competing influence of textual
tokens, our method reduces the dependence of the manipulated output on the
specific wording of the query. Extensive experiments on widely used VLMs show
that Attention Hijacking substantially improves cross-query transferability across
diverse target responses and unseen queries. The method also extends effectively
to multiple attack scenarios, offering new insights into the role of attention stability
in transferable response manipulation for VLMs.
1 Introduction
Vision language models (VLMs) [23, 4, 9, 25] have recently achieved remarkable progress, driven
by increased data availability and computational power. As these models are increasingly used as
response-generation systems, their outputs are now relied upon in a wide range of applications,
including autonomous driving, public security, he

>>> ABSTRACT:
Existing adversarial attacks on vision-language models (VLMs) can steer model
outputs toward attacker-specified target responses, but their effectiveness often
degrades when the same perturbed input is paired with different textual queries.
This paper studies cross-query response manipulation, where a single adversarial
example is expected to remain effective across diverse user queries. We first
analyze the limitations of existing attacks and find that successful transfer is closely
associated with preserving an image-dominant attention pattern during response
generation. Motivated by the observation, we proposeAttention Hijacking, a
novel adversarial attack that explicitly steers internal attention distributions toward
a persistent image-dominant pattern. By amplifying the influence of visual tokens
on target response tokens while suppressing the competing influence of textual
tokens, our method reduces the dependence of the manipulated output on the
specific wording of the query. Extensive experiments on widely used VLMs show
that Attention Hijacking substantially improves cross-query transferability across
diverse target responses and unseen queries. The method also extends effectively
to multiple attack scenarios, offering new insights into the role of attention stability
in transferable response manipulation for VLMs.
```



## 【注意力思路获取】Causal Attribution via Activation Patching.pdf

```
Causal Attribution via Activation Patching
Amirmohammad Izadi∗, Mohammadali Banayeeanzade*, Alireza Mirrokni*, Hosein Hasani*,
Mobin Bagherian,Faridoun Mehri,and Mahdieh Soleymani Baghshah
Sharif University of Technology
Abstract
Attribution methods for Vision Transformers (ViTs) aim to identify image regions that
influence model predictions, but producing faithful and well-localized attributions
remains challenging. Existing attribution methods face several limitations, with
gradient-based, relevance-propagation, and attention-based methods relying on local
approximations, while perturbation or optimization-based methods intervene on
inputs, tokens, or surrogates rather than internal patch representations. The key
challenge is that class-relevant evidence is formed through interactions between patch
tokens across layers; methods that operate only on input changes, attention weights,
or backward relevance signals may therefore provide indirect proxies for patch
importance rather than directly testing the predictive effect of contextualized patch
representations. We proposeCausal Attribution via Activation Patching (CAAP),
which estimates the contribution of individual image patches to the ViT’s prediction
by directly intervening on internal activations rather than using learned masks or
synthetic perturbation patterns. For each patch, CAAP inserts the corresponding
source-image activations into a neutral target context over an intermediate range
of layers and uses the resulting target-class score as the attribution signal. The
resulting attribution map reflects the causal contribution of patch-associated internal
representations on the model’s prediction. The causal intervention serves as a
principled measure of patch influence by capturing semantic evidence after initial
representation formation, while avoiding late-layer global mixing that can reduce
spatial specificity. Across multiple ViT backbones and standard metrics, CAAP
consistently outperforms existing met

>>> ABSTRACT:
Attribution methods for Vision Transformers (ViTs) aim to identify image regions that
influence model predictions, but producing faithful and well-localized attributions
remains challenging. Existing attribution methods face several limitations, with
gradient-based, relevance-propagation, and attention-based methods relying on local
approximations, while perturbation or optimization-based methods intervene on
inputs, tokens, or surrogates rather than internal patch representations. The key
challenge is that class-relevant evidence is formed through interactions between patch
tokens across layers; methods that operate only on input changes, attention weights,
or backward relevance signals may therefore provide indirect proxies for patch
importance rather than directly testing the predictive effect of contextualized patch
representations. We proposeCausal Attribution via Activation Patching (CAAP),
which estimates the contribution of individual image patches to the ViT’s prediction
by directly intervening on internal activations rather than using learned masks or
synthetic perturbation patterns. For each patch, CAAP inserts the corresponding
source-image activations into a neutral target context over an intermediate range
of layers and uses the resulting target-class score as the attribution signal. The
resulting attribution map reflects the causal contribution of patch-associated internal
representations on the model’s prediction. The causal intervention serves as a
principled measure of patch influence by capturing semantic evidence after initial
representation formation, while avoiding late-layer global mixing that can reduce
spatial specificity. Across multiple ViT backbones and standard metrics, CAAP
consistently outperforms existing methods in various settings and produces more
faithful and localized attributions.
```
