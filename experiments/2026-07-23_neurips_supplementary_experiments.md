# NeurIPS 2026 补充实验总计划

**状态：DRAFT / 未获训练授权 / 禁止启动 GPU 实验**
**日期：** 2026-07-23
**对应材料：** `review_neurips.md`
**主任务：** 回应 Reviewer Gfi2、5HPc、j7NF 对机制、统计、公平性、成本与 RLVR 价值的质疑。

> 本文件只整理实验，不代表超参数已获确认。按照项目规则，任何训练或评测启动前，必须逐项确认第八节的全部参数，并完成数据、模型、脚本、GPU、tmux 和日志预检。

## 一、结论先行

补充工作不应直接铺开所有模型和任务。建议按以下顺序推进：

1. **先做有效性审计：** Logic split、frozen weak、表格数字、Gemma 预处理和 UNDIAL 实现。任何一项失败都应先修正论文，不能用更多训练掩盖。
2. **主线只锁定 Qwen3-4B-Base + Math：** 用最强反证实验回答“WMSS 是否只是更久 SFT / 更高 LR / 普通 sharpening”。
3. **再做统计与分布代价：** 三个独立训练 seed、validation-selected peak、pass@k、calibration、OOD。
4. **RLVR 转移放第二阶段：** 它最能回答现实价值，但成本高，不能抢在机制和有效性审计之前。
5. **跨规模、跨架构放最后：** 只在核心机制成立后追加。

建议把实验组织为五个包：

| 包 | 内容 | 优先级 |
|---|---|---|
| A | 有效性审计 | P0，所有训练前 |
| B | Sharpening 排除 + 真实 weak logits 必要性 | P0，核心 |
| C | 三 seed 公平主表 + peak checkpoint | P0，核心 |
| D | pass@k + calibration + OOD | P1 |
| E | RLVR、WD-DS、weak-to-strong、规模与效率 | P1/P2 |

## 二、现有证据覆盖

| 审稿要求 | 当前状态 | 可复用证据 | 缺口 |
|---|---|---|---|
| Matched SFT | 部分覆盖 | `experiments/2026-07-20_qwen3_4b_base_math_sft_ep1.md` 有严谨 1-epoch SFT；Medical 有近似 3-epoch 对比 | 没有 Math 完整 WDL 对同 token/update/FLOPs 的 SFT |
| Higher-LR SFT | 缺失 | Code 有不同 LR 历史结果 | 不是干净的纯 SFT 控制 |
| Label smoothing | 缺失 | 无 | 需新增 |
| 独立训练 multi-seed | 缺失 | 现有训练基本为 seed 42 | 无 mean/std/CI |
| pass@k | 仅有计划 | GRAPE 计划已有重复采样和 bootstrap 设计 | 无 WMSS vs SFT 正式结果 |
| Calibration | 缺失 | 无 | 无 NLL/Brier/ECE/reliability |
| OOD | 缺失 | 有多域实验 | 都是域内训练/测试，不是明确 OOD |
| RLVR | 缺失 | 有 SPIN/DPO | 不属于 RLVR |
| FLOPs/成本 | 部分覆盖 | 有 runtime；Medical 记录双模型约 81s/step、单模型约 20s/step | 无完整 pipeline FLOPs、GPU-hours、峰值显存公平对照 |
| Uniform/matched entropy | 缺失核心对照 | `2026-04-26_qwen25_3b_math_abg_ablation.md` 有部分 alpha/beta/gamma 结果 | 没有真实 weak vs uniform/同熵/打乱 weak |
| 数据公平性 | 部分覆盖 | Math SFT 数据哈希和重叠审计完整 | 主表 baseline 的数据、step、token、compute 尚未统一 |

现有结果只能作为背景证据，不能替代下面的正式补充实验。旧实验文件中的“训练中”状态不得当作当前实时状态。

## 三、包 A：训练前有效性审计

这些项目不是“加分实验”，而是继续实验的门禁。

### A1. Logic split 零泄漏审计

**目的：** 回应 PAT 对 17,205 / 2,072 / 14,882 数量关系的质疑。

**检查：**

1. 原始数据 split、最终训练集、全部评测集的 sample ID 交集。
2. 规范化题面后的 exact overlap。
3. 去除选项顺序、空白和格式差异后的 near-duplicate。
4. 每条最终训练样本到源 split 的 provenance。
5. 重新计算 retention rate 的分母和分子。

**成功判据：**

- 训练集与所有评测集 ID/hash 交集为 0。
- Near-duplicate 清单可人工核查。
- 17,205、2,072、14,882 的关系可由同一份 receipt 重放。
- 若不通过，立即停止引用 Logic 结果并重建数据。

### A2. Frozen weak 不变量审计

**目的：** 回应 Table 10/B.8 中 frozen weak 出现 Pre/Post drift 的问题。

**检查：**

1. `requires_grad=False`。
2. optimizer parameter groups 不包含 weak 参数。
3. 训练前后全部 weak tensor hash 相同。
4. 固定 200 样本、同 tokenizer、同 precision、同 eval mode 重算 logits。
5. 区分“weak checkpoint 变化”和“同 weak 在不同 strong/fused context 下统计变化”。
6. 重算 sensitivity ratio 和 crossover。

**成功判据：**

- Weak 权重逐 tensor 完全不变。
- 相同输入/环境下 weak-only logits 可复现。
- 若 Pre/Post 表实际统计的不是同一对象，修正表头和论文解释。
- 使用正确 Pre 值后重新报告 crossover。

### A3. 结果表一致性审计

**目的：** 解决 Table 1/7/8 默认配置和 Qwen2.5-3B AIME 冲突。

**检查：**

- 每个表格单元追溯到唯一 checkpoint、eval command、parser、seed 和 artifact hash。
- 区分 greedy 单次、三个训练 run 均值和多 rollout mean。
- 重算 Math Avg 和舍入。
- 禁止从手写表格反推结果。

**成功判据：** 同一配置只能对应一个 canonical result；所有论文数字由机器可重放汇总器生成。

### A4. Gemma / UNDIAL / DPO 实现审计

1. Gemma：确认只删除标签还是删除整个 `<think>` 内容，抽查转换前后样本。
2. UNDIAL：画不同序列长度下实际 logit noise std，对比当前 `/sqrt(L)` 与原方法合同。
3. DPO：明确 “highest-scoring rollout” 的 score 来源。
4. 如实现偏离原方法，修复后必须重跑对应 baseline，旧结果标记无效。

### A5. 匿名合规

这不是实验，但属于当前最高风险门禁：核查当时 Supplementary ZIP 的姓名、README、arXiv、Hugging Face、GitHub、badge、commit metadata 和链接跳转。匿名问题不能由补实验抵消。

## 四、包 B：Sharpening 排除与 weak logits 必要性

### B1. 核心问题

Reviewer j7NF 的核心拒稿理由是：WD-JT 可能只是让旧 SFT 梯度继续存在，没有使用真实弱模型中的结构信息。

### B2. 固定实验对象

- **主模型：** `Qwen/Qwen3-4B-Base`
- **主域：** Math
- **训练数据：** `dataset/am_deepseek_r1_filtered_ad.jsonl`
- **主评测：** Math500、AIME2025、AMC23、GSM8K、AQuA、MAWPS、SVAMP
- **主配置来源：** 当前 NeurIPS 论文对应的 canonical WMSS 配置
- **选择原则：** 数据、completion mask、optimizer、scheduler、batch、最大长度和 checkpoint 起点全部一致，只改变被研究因素。

正式启动前必须确认当前论文结果到底对应哪一版 loader、全序列或 completion-only loss、最大长度和 Stage 定义；不得直接沿用 4 月存在回归/简化的配置。

### B3. 对照臂

| ID | 实验臂 | 作用 |
|---|---|---|
| B0 | Published SFT baseline reproduction | 锚定论文原始对照 |
| B1 | Matched-update/token longer SFT | 排除更多监督步数和 token |
| B2 | Matched-FLOPs SFT | 排除 WD-JT 双 forward 的计算预算优势/劣势 |
| B3 | Higher-LR SFT | 排除简单 LR restart / 更强梯度 |
| B4 | Label smoothing SFT | 回应 reviewer 指定 baseline |
| B5 | Uniform weak distribution | 测试无结构分布是否足够 |
| B6 | Per-sample entropy-matched synthetic distribution | 保持熵，去掉真实 weak 结构 |
| B7 | Shuffled weak logits | 保持真实 weak 边际统计，破坏样本对应关系 |
| B8 | Real weak logits WD-JT | 核心方法 |
| B9 | Full WMSS = WD-DS + WD-JT | 论文完整方法 |

B1 和 B2 必须同时存在。“相同 epoch”不是公平计算对照：WD-JT 每 step 包含双模型 forward。

### B4. 两阶段执行策略

**筛选阶段：** 所有 B0–B9 先用一个已确认 seed 跑完整主设置，检查机制排序与实现稳定性。

**确认阶段：** 至少对以下臂跑三个独立训练 seed：

- B1 matched-update SFT
- B2 matched-FLOPs SFT
- B3 higher-LR SFT
- B6/B7 中表现最强的无结构控制
- B8 real weak WD-JT
- B9 full WMSS

如果算力不足，不能用重复 greedy evaluation 冒充独立训练 seed。

### B5. 成功判据

支持核心机制至少需要：

1. B8 在三个训练 seed 上总体优于 B1/B2/B3。
2. B8 优于 B6/B7，说明真实 weak 的样本相关结构有额外贡献。
3. B9 相比 B8 的提升可归因于 WD-DS，而不是更多数据曝光。
4. 提升不只由 AIME/AMC 单题变化推动。
5. 若 synthetic/shuffled control 与 B8 相当，应把论文机制改写为 sharpening/regularization，而不能继续声称 weak-specific correction。

## 五、包 C：三 seed、公平主表与 peak checkpoint

### C1. 独立训练 seed

最低要求是三个独立训练 seed。每个 seed 必须控制：

- Python、NumPy、Torch、CUDA RNG。
- Dataset shuffle / DistributedSampler。
- 模型初始化路径。
- 任何样本选择和 tie-break。
- vLLM request seed 与训练 seed 分开记录。

### C2. Validation-selected peak

1. 训练前固定 validation split；不能使用 AIME、AMC、Math500 等最终测试集选 epoch。
2. 固定 checkpoint cadence。
3. 用 validation 主指标选择每个 run 的 peak。
4. 同时报告 final checkpoint，避免只挑最优点。
5. 所有方法使用相同选择规则。

### C3. 统计报告

- 每个数据集：三 seed mean、std、min/max。
- 主比较：seed-level paired difference。
- 样本级：paired bootstrap 95% CI。
- AIME/AMC：同时报告正确题数，不只报百分比。
- Macro average：明确是否等权；不得让小数据集隐式放大。
- “mean@3” 改写为 “mean over 3 independent training runs”。

### C4. 公平主表

主表至少包含 B1、B2、B3、B8、B9。所有行必须附：

- 训练样本数、唯一样本数。
- optimizer updates。
- supervised completion tokens。
- 总 sequence tokens。
- estimated/measured FLOPs。
- GPU-hours、wall-clock。
- peak GPU memory。
- checkpoint selection rule。

## 六、包 D：pass@k、calibration 与 OOD

### D1. pass@k / 多样性

优先复用 GRAPE 计划中已经设计的逐 request 独立 deterministic seed 和 bootstrap 工具，但要针对 canonical SFT/WMSS checkpoint 正式运行。

**建议报告：**

- greedy pass@1
- sampled pass@1
- pass@4、pass@8、pass@16
- unique answer / unique correct solution 比率
- completion length 和截断率
- paired bootstrap 95% CI

AIME/AMC 因题目少，需要比 Math500 更多 rollout。具体 temperature、top-p、每题次数和最大长度必须经用户确认。

### D2. Calibration

在答案可验证任务上报告：

- completion NLL
- Brier score
- ECE / reliability diagram
- selective accuracy / risk-coverage
- 正确与错误答案的 confidence 分布

必须先固定 sequence confidence 的定义，不能事后在多种定义中挑最有利者。

### D3. OOD

OOD 数据集必须满足：

1. 未用于训练。
2. 未用于 WD-DS 选样。
3. 未用于超参数和 checkpoint 选择。
4. 与训练域存在明确 domain shift 或 difficulty shift。
5. 在实验计划中提前固定数据路径、版本和 parser。

候选数据集需要先盘点本地可用集；未确认前不填具体名称。最低比较 B1/B8/B9，并同时报告 pass@1、pass@k、NLL/ECE 和生成熵。

### D4. 成功判据

- WMSS 的 greedy 收益不能伴随显著 pass@k、校准或 OOD 退化。
- 若存在 trade-off，应在论文中明确报告，而非只保留 pass@1。

## 七、包 E：第二阶段扩展

### E1. SFT 到 RLVR 的转移

| Arm | 初始化 | RLVR |
|---|---|---|
| E1-Base | Base | 相同 RLVR |
| E1-SFT | Matched SFT | 相同 RLVR |
| E1-WMSS | Full WMSS | 相同 RLVR |

必须控制 reward、rollout 数、policy updates、KL、token budget 和总计算。比较初始分数、训练曲线、最终 pass@1/pass@k、探索覆盖和 entropy。理想为三个 seed；若只做一个 seed，只能作为初步证据。

**关键判据：** WMSS 初始化在相同 RLVR 预算后仍优于 SFT，且没有更差的探索覆盖。

### E2. WD-DS 是否只是 uncertainty sampling

固定选择样本数，比较：

1. Random
2. 仅 `H(strong)`
3. 仅 weak-strong entropy gap
4. 完整 WD-DS
5. 与 WD-DS strong-entropy 分布匹配的 sampler

报告样本集合重叠、选中样本特征、最终性能和筛选/训练成本。

### E3. Weak-to-strong 与数据公平性

- 选择一个标准 weak-to-strong/weak supervision baseline。
- SPIN/SSB 若进入主表，必须与 WMSS 使用相同数据池、active sample 数和 token/compute budget；不能继续用 20K 对 111K。
- 对 rollout 型方法同时报告数据生成成本。

### E4. 规模与架构泛化

核心实验成立后，再补：

- Qwen3 4B → 8B → 14B 的收益趋势。
- 至少一个不同模型架构。
- 优先代码任务，因为 reviewer 特别指出 4B 到 8B 收益下降。
- 70B+ 只做可扩展性/显存估算也可，rebuttal 阶段不建议直接承诺完整训练。

### E5. 机制诊断

比较 real weak、entropy-matched、shuffled weak：

- target/non-target logits
- margin
- gradient norm
- 样本级 entropy 与梯度变化
- 参数/日志中的 global shift
- weight decay on/off 对 null-space drift 的影响

这些诊断只能解释机制，不能替代任务指标。

## 八、完整超参数确认清单

下列每一项必须由用户确认后才能生成启动脚本。括号内仅表示当前项目常见值，不代表已批准。

| 类别 | 参数 | 候选/当前信息 | 状态 |
|---|---|---|---|
| 模型 | base model/revision | Qwen3-4B-Base | 待确认 |
| 数据 | canonical train path/hash | AM Math 111,657 | 待确认 |
| 数据 | validation split 比例与 seed | 必须新建且不碰 test | 待确认 |
| Loss | 全序列或 completion-only | 历史两者均出现过 | 必须确认 |
| 长度 | train max sequence length | 4096 或 8192 | 待确认 |
| Batch | per-device batch | 常见 1 | 待确认 |
| Batch | gradient accumulation/global batch | 需与论文 canonical 对齐 | 待确认 |
| 优化 | optimizer | 需从论文配置核验 | 待确认 |
| 优化 | base LR | 常见 1e-5 | 待确认 |
| 优化 | higher-LR 候选 | 例如 2e-5/5e-5，不可自行决定 | 待确认 |
| 优化 | betas/epsilon/weight decay | 需完整列出 | 待确认 |
| 优化 | scheduler/warmup | 需完整列出 | 待确认 |
| 训练 | epoch/update/token budget | 同时定义 update-match 和 FLOPs-match | 待确认 |
| 随机性 | training seeds | 至少 3 个 | 待确认 |
| 保存 | checkpoint cadence | 用于 validation peak | 待确认 |
| WD-JT | lambda / fusion rule | 论文 canonical 值 | 待确认 |
| WD-DS | alpha/beta/gamma 与采样量 | 论文 canonical 值 | 待确认 |
| 控制 | label smoothing epsilon | 不可默认 | 待确认 |
| 控制 | uniform/matched-entropy 构造 | 数学定义和数值容差 | 待确认 |
| Eval | parser/prompt/thinking mode | 必须跨臂一致 | 待确认 |
| Eval | temperature/top-p/max tokens | pass@k 专用 | 待确认 |
| Eval | rollout 数和 request seeds | 按数据集固定 | 待确认 |
| Calibration | sequence confidence 定义/binning | 预注册 | 待确认 |
| OOD | 数据集/version/path | 本地盘点后确认 | 待确认 |
| RLVR | 算法/reward/KL/rollout/update budget | 第二阶段 | 待确认 |
| 系统 | precision/DeepSpeed/FA2 | transformers 保持 4.57.1 | 待确认 |

## 九、资源与执行路径

### 资源估算原则

不在超参数和 canonical pipeline 未确认前给出伪精确总时长。已知参考：

- Qwen3-4B Math 单模型 1 epoch 历史约 5–7 小时。
- Stage3 双模型历史明显更慢。
- B0–B9 全部完整跑三个 seed 将是数十个 8-GPU run，无法在单机上短时间完成。

因此必须采用“单 seed 筛选 → 关键臂三 seed → 评测扩展”的顺序。

### 计划新增入口

遵循“不改已有脚本，优先新建文件”：

- 控制组 launcher：`EnsembleLLM/scripts/run_20260723_neurips_math_controls.sh`
- 统计评测：`EnsembleLLM/scripts/run_20260723_neurips_math_passk_calibration.sh`
- 结果汇总：`EnsembleLLM/scripts/summarize_20260723_neurips_supplement.py`
- 结果目录：`EnsembleLLM/results/20260723_neurips_supplement/`
- 日志目录：`EnsembleLLM/TrainLogs/`
- 权重目录：`checkpoints/2026-07-23_qwen3-4b_<method>_<key-hparams>_neurips-supp/`

以上文件目前均未创建。创建前先锁定实验合同。

### 执行机器

- 当前唯一机器：本机 8 GPU。
- 所有训练/评测必须在 tmux 后台运行。
- 启动前必须执行 tmux、GPU、最近 TrainLogs 三联查。
- 当前其他正式实验可能占用 GPU；补充实验不得与其争抢显存。

## 十、推荐执行顺序

1. A1–A5 全部审计并形成 receipts。
2. 确认第八节全部参数。
3. B0–B9 单 seed 机制筛选。
4. 根据筛选结果冻结三 seed 确认臂，不允许看 test 后增删方法。
5. 完成 C：三 seed、validation peak、公平主表和成本。
6. 对冻结 checkpoint 完成 D：pass@k、calibration、OOD。
7. 根据 rebuttal 需要决定是否启动 E1 RLVR。
8. 最后考虑 WD-DS、weak-to-strong 和规模泛化。

## 十一、停止条件

任一条件触发时，应停止扩展并重新解释论文：

1. Logic 数据确认泄漏。
2. Weak 参数实际发生更新，或 frozen weak drift 无法解释。
3. Canonical 结果表无法追溯或主要数字不能复现。
4. Matched SFT / synthetic weak 与 real weak 无显著差别。
5. WMSS 的 pass@k、calibration、OOD 或 RLVR 明显劣于 SFT。
6. 预计实验无法在 rebuttal 时间和单机预算内形成可信结果。

## 十二、结果

尚未执行。所有指标、checkpoint、日志和失败记录均为空。

### 结果回写要求

任何实验一旦获得用户逐项确认并启动：

1. 启动前把该实验拆成独立 `experiments/YYYY-MM-DD_<name>.md`。
2. 当场登记 checkpoint MANIFEST 和目录内 `meta.md`。
3. 结果或失败立即追加到对应实验文件，不得只在对话中汇报。
4. 主表只接收通过 provenance、hash 和公平预算校验的结果。
