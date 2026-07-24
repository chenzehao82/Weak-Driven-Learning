# NeurIPS 2026 审稿意见汇总

**论文：** Weak-Driven Learning: How Weak Agents make Strong Agents Stronger
**投稿编号：** 3503
**OpenReview ID：** `WAqz1qihuI`
**提交/修改时间：** 2026-04-21 / 2026-05-27
**版本数：** 2
**材料状态：** 3 份 Official Review；无 rebuttal、讨论往返、Meta Review 或最终决定

> 本文根据用户提供的 6 页 OpenReview 浏览器打印 PDF 整理。打印版部分长行的右侧被裁切，因此以下是忠实的中文结构化整理，不臆造被裁掉的英文尾句。页 3–6 的 PAT 内容是自动 LLM 反馈，明确不参与正式评审，不能当作第四位审稿人。

## 一、投稿信息

- **TL;DR：** 强模型在 SFT 饱和后，可通过与自身历史弱 checkpoint 联合训练继续提升；无需额外教师，也不增加部署时推理成本。
- **Primary Area：** Language and multimodal language models
- **Secondary Area：** Deep learning advancements
- **Contribution Type：** General
- **LLM Usage：** Editing
- **LLM Experiment：** Opt in
- **许可证：** CC BY 4.0
- **Supplementary Material：** 已提交 ZIP

## 二、评分总览

| Reviewer | Quality | Clarity | Significance | Originality | Rating | Confidence |
|---|---:|---:|---:|---:|---|---:|
| Gfi2 | 3 | 2 | 2 | 3 | 3: Borderline reject | 3 |
| 5HPc | 3 | 3 | 3 | 4 | 5: Accept | 3 |
| j7NF | 2 | 2 | 2 | 2 | 2: Reject | 3 |

总体评分为 **3 / 5 / 2**，均值 **3.33**。三人置信度均为 3；一位接收、一位边缘拒稿、一位拒稿，分歧明显。三人均报告无或仅有很轻微的伦理问题。

## 三、Reviewer Gfi2

**时间：** 2026-06-24（修改于 2026-07-23）
**建议：** 3 — Borderline reject；**置信度：** 3

### 摘要

WMSS 使用历史弱 checkpoint 改进强模型：按弱强模型的预测熵差选择数据，并混合弱强 logits，以增强难负 token 的训练信号。方法改善了推理和代码表现，但需要额外训练计算，并依赖弱强模型间存在有意义的预测差异。

### 优点

1. 问题设定虽不新，但重要且受到社区关注。
2. 将历史训练状态作为可复用监督信号的想法新颖、有社区价值。
3. 附录以细粒度记录了相关细节。

### 缺点与问题

1. **Agent 命名不准确。** Agent 通常涉及长程任务、记忆、多轮交互和工具使用，而本文实验只是非 agentic 的数学和代码任务。
2. 部分图和可视化中文字过小，无法辨认。
3. 只在 4B–8B 小模型和有限模型族上实验，难以判断对大模型及其他架构的泛化；Qwen3 代码收益从 4B 到 8B 下降尤其值得关注。
4. 数学/代码后训练的事实标准是 RLVR。除非证明 SFT 收益能转移到 RLVR/OPD，或 WMSS 可用于这些阶段，否则现实影响有限。分布收窄还可能削弱后续探索。审稿人建议讨论：
   - *Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead*
   - *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*
5. 应报告验证集过拟合前的 peak checkpoint，而不应只报固定两 epoch。
6. Token 分布收窄是否损害 OOD 表现？需要实证。
7. 缺少 weak-to-strong baseline。
8. 缺少 FLOPs 开销；wall-clock 依赖硬件，不能用于硬件无关比较。

Questions 栏写的是“请参见 weaknesses”。Limitations 填写 “Yes”。

### 匿名与格式风险

- 补充代码包含大量姓名和可识别链接。
- `README.md` 的 arXiv、Hugging Face 等链接会直接暴露作者身份。
- 审稿人建议 AC 检查是否应 desk reject。
- 另有图中文字过小问题。

### 评分

| Quality | Clarity | Significance | Originality | Rating | Confidence |
|---:|---:|---:|---:|---|---:|
| 3: good | 2: not good | 2: not good | 3: good | 3: Borderline reject | 3 |

## 四、Reviewer 5HPc

**时间：** 2026-06-23（修改于 2026-07-23）
**建议：** 5 — Accept；**置信度：** 3

### 摘要

Weak-Driven Learning 使用较弱历史 checkpoint 提供纠正信号。WD-JT 混合弱强 logits，为难例维持训练信号；WD-DS 用熵差选择高价值样本。论文提供难负样本梯度增强的理论支持，并在多个 benchmark 上报告提升，无部署时推理开销。

### 优点

1. 写作清晰，逻辑连贯。
2. 提出了新的大模型后训练思路；方向与常见的强模型蒸馏弱模型相反，具有新颖性和社区价值。
3. 实验充分，技术可靠，论断有较好证据支持。

### 缺点与建议

1. 标题中的 “Agents” 容易误导；方法与工具调用、自主决策等传统 agent 定义无关。
2. 结构上可考虑先介绍面向训练样本的 WD-DS，再介绍面向训练方法的 WD-JT。
3. 所有实验使用 greedy decoding；AIME、AMC 等小 benchmark 应采用多个训练/评测 seed，重复测试并报告均值，以获得可靠比较。

Questions 栏要求回应上述优缺点；Limitations 填写 “yes”。

### 格式问题

Section 4.2 标题（line 163）与上方 Figure 2 caption 间距过小，需要检查会议模板合规性。

### 评分

| Quality | Clarity | Significance | Originality | Rating | Confidence |
|---:|---:|---:|---:|---|---:|
| 3: good | 3: good | 3: good | 4: excellent | 5: Accept | 3 |

## 五、Reviewer j7NF

**时间：** 2026-06-20（修改于 2026-07-23）
**建议：** 2 — Reject；**置信度：** 3

### 摘要

WMSS 用早期 SFT checkpoint 在 MLE 接近收敛后继续提供信号。WD-JT 把弱 logits 混入强 logits，使已学会目标 token 后仍有梯度；WD-DS 根据难度、回退和弱强差异挖掘样本。论文在 Qwen/Gemma 的数学、代码、逻辑任务上与两 epoch SFT 比较。

审稿人的核心判断是：**WMSS 让已经学会的 token 继续收到 SFT 更新，而论文将 sharpening 描述成 correction。**

### 优点

1. 部署时免费、无需新 rollout；最终只部署强模型，并复用已有弱 checkpoint。
2. 简单、易复现；核心是 SFT loss 的小改动加一个 sampler。

审稿人强调部署时免费不等于训练时免费。

### 七项主要缺点

1. **W1：纠正信号可能只是 sharpening。** 弱模型使混合分布保持低置信度，让强模型梯度不消失，本质上是重新施加旧标签梯度。Theorem 1 影响整个非目标分布而非只针对错误干扰项，“纠正/定向”表述过强。
2. **W2：没有排除普通 sharpening 或更长训练。** 缺少 longer-SFT、higher-LR、matched-entropy target、label smoothing。
3. **W3：理论没有证明准确率提高。** 它只证明更平坦 softmax 和更大非目标梯度，未连接到更好决策；Corollary 2 接近重述自身不等式。
4. **W4：小 benchmark 只有单 seed。** Table 1 无误差棒；AIME 30 题，单题变化即可显著改分。
5. **W5：未测多样性成本。** 只报 pass@1；sharpening 可能用覆盖度换 top-1，大 k 的 pass@k 可能更差。
6. **W6：WD-DS 可能只是 uncertainty sampling。** 主导 `H(strong)` 项不用弱模型，weak-driven 部分仅 entropy-gap 分支。
7. **W7：logit mixing 不新。** 与 Deep Mutual Learning、co-distillation、KDCL、Transformer Copilot 相似；新意可能仅是把历史弱 checkpoint 配成纠正分支。

### 五个要求

1. 用 uniform 或相同熵的 temperature-matched distribution 替代弱 logits；验证真实弱模型结构是否必要。
2. 增加 matched-compute SFT、higher-LR SFT、label smoothing。
3. 在数学任务报告相对 SFT 的 pass@k 和 calibration。
4. 明确所有 baseline 是否使用相同训练数据和 step；否则按公平设置重跑 Table 1。
5. 在主干模型报告至少三个训练 seed 和误差棒。

### 局限性

1. 超过 MLE 最优点继续 sharpening 可能损害校准。
2. 只测 pass@1，未处理多样性成本。
3. 只报 wall-clock 而非 FLOPs；WD-JT 每步双模型计算，WD-DS 节省数据后的净成本优势可能有限。
4. Table 1 单次运行有未量化方差。

### 评分

| Quality | Clarity | Significance | Originality | Rating | Confidence |
|---:|---:|---:|---:|---|---:|
| 2: not good | 2: not good | 2: not good | 2: not good | 2: Reject | 3 |

## 六、跨审稿人共识

1. **三人均认为 “Agent” 命名不当或误导。**
2. **三人均关注统计稳定性：** peak checkpoint、多 seed、误差棒。
3. **训练成本不足：** FLOPs、显存、吞吐和双 forward 开销。
4. **后续 RL/RLVR 与分布收窄风险：** 探索能力、pass@k、calibration、OOD。
5. **Baseline 不足或不公平：** weak-to-strong、longer/higher-LR SFT、label smoothing、matched entropy，以及对齐数据和 step。
6. **高风险单点：** 匿名违规可能 desk reject；理论可能只是普通 sharpening；结果与表格一致性需审计。

## 七、Program Chairs 的 PAT 自动 LLM 反馈

### 身份和效力

- 2026-05-05 01:49：通知已收到并处理论文。
- 2026-05-05 02:25：发布自动反馈。
- 2026-05-07 09:58：邀请填写问卷。
- 页面明确声明 PAT 反馈不会用于正式评审，reviewer/AC/PC 不可见，且模型可能幻觉。

### PAT 认可的优点

1. WD-JT 反转传统蒸馏方向，构思简洁、新颖。
2. 三阶段训练动态提供了完整机制叙事。
3. 多任务、多种 3B–8B 模型上有提升，无部署推理开销。
4. WD-DS 可优先选择高价值样本并减少训练量。

### PAT 指出的数据与实验问题

1. **Logic 数据可能泄漏：** 17,205 条合并数据、2,072 条评测保留和 14,882 条最终训练数据之间的算术关系可疑，需证明 split 无交集。
2. **SPIN/SSB 只用 20K，而 WMSS 约用 111K，数据规模不公平。**
3. **Gemma tag stripping 表述像是删除整个 CoT，可能与逐步推理评测 prompt 错配。**
4. **UNDIAL 的 logit noise 除以 `sqrt(L)` 可能把长序列噪声缩到近零。**
5. **根据最终测试集轨迹选 Epoch 3 可能构成测试集泄漏。**
6. **默认配置结果矛盾：** Table 1 为 AIME 20.0/MATH500 71.4，Table 7 为 16.7/68.2，Table 8 为 20.0/73.3。
7. **Qwen2.5-3B AIME 矛盾：** Table 1 为 6.7%，Appendix B.5 称六种配置均为 0。
8. **DPO 的 “highest-scoring rollout” 未说明 score 来源。**
9. 缺 optimizer、global batch、scheduler、warmup、weight decay、full/PEFT 等关键超参数。
10. 应增加 label smoothing、temperature scaling 等简单对照。
11. 应报告显存峰值、吞吐、最大 batch 和 70B+ 可扩展性。
12. 建议补充 WD-DS 相近工作 InstructDiff（2026）。

### PAT 指出的理论问题

1. Appendix C.2/C.3 像是假设弱强模型都更新，但 Algorithm 1 的 weak model 是 frozen；gradient-share crossover 推导与实现不一致。
2. Frozen weak model 在固定 200 条样本上的 Pre/Post logit 统计不应变化，但 Table 10/B.8 报告明显 drift，可能是统计标签、脚本或意外更新错误。
3. Sensitivity ratio 混用 Strong Pre `1240.10` 与 Weak Post `1034.50`；若用两者 Pre 值，应为 `(1240.10/1191.33)^2 ≈ 1.083`，crossover 约 0.49。
4. Proposition 3 从参数梯度推出逐 token logit update，隐含 `K_strong ∝ I` 的 Jacobian Gram matrix 假设。
5. 零均值随机游走不能自然解释单向 null-space mean drift；weight decay 更可能，且交叉熵全局 shift 梯度应精确为零。
6. Theorem 1/Corollary 2 缺显式证明；Equation 24 可能漏 `s_i`；frozen weak 条件下 cross-Hessian 分析可能无关。

### PAT 指出的表格、引用和格式问题

1. Table 1 Math Avg 有舍入错误：`509.5/7≈72.78`、`394.4/7≈56.34`。
2. “mean@3” 容易误解；若是三次训练的 greedy pass@1 均值，应改名。
3. “Logit Variance (sigma)” 的符号应区分标准差与方差。
4. Section 6.3 对 “+ WD-JT” 和 full WMSS 的组合关系表述不清。
5. Algorithm 1 时间下标、UNDIAL epsilon 符号、WD-DS “mastered” 表述需修正。
6. Gemma、TRL、transformers 缺引用；资产许可证未逐项写具体名称。
7. 部分预印本引用、系统名大小写、UNDIAL 重音符号、parenthetical citation 格式需修正。
8. Section 4.2 间距和图中文字大小需检查。

## 八、处理优先级

### P0：有效性或合规

1. 清理补充材料全部身份链接，核查匿名政策。
2. 审计 Logic split，并给出 ID 交集为零的证据。
3. 核查 frozen weak 的 Pre/Post 统计和训练代码。
4. 统一 Table 1/7/8 与 Qwen2.5-3B 结果。
5. 澄清/修正 Gemma tag stripping 和 UNDIAL 噪声。

### P1：核心实验

1. Matched-compute longer SFT、higher-LR SFT、label smoothing。
2. Uniform/matched-entropy/temperature-matched weak-logit 替代。
3. 至少三个训练 seed、误差棒。
4. Pass@k、calibration、OOD、SFT 后 RLVR。
5. 主 baseline 对齐数据、step、token budget。
6. FLOPs、显存、吞吐、wall-clock。

### P2：定位与写作

1. 删除或严格定义 “Agent”，考虑重命名论文和 WMSS。
2. 收缩 correction/难负定向的机制表述，说明与 sharpening 的区别。
3. 重写 frozen weak 条件下的理论。
4. 补相关工作、超参数、许可证和引用。
5. 修正图表、间距、符号、舍入、参考文献。

## 九、材料中没有的内容

该 PDF 不含作者 rebuttal、reviewer-author 往返、AC/SAC Meta Review 或最终决定，不能据此推断最终录用结果。
