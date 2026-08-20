#!/bin/bash
set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ========== 环境配置 ==========
eval "$(conda shell.bash hook)"
# export HF_ENDPOINT="https://hf-mirror.com"
# Hugging Face uses its standard cache unless HF_HOME is supplied by the user.

# ========== 日志配置 ==========
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/ensemble_${TIMESTAMP}.log"

# 将所有输出同时写入日志文件和终端
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 显示 GPU 使用情况的函数
show_gpu_usage() {
    echo ""
    echo "========== 当前 GPU 使用情况 =========="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "[%s] %s | %s°C, %3s%% | %5s / %5s MB\n", $1, $2, $3, $4, $5, $6}'
    echo "========================================"
    echo ""
}

# ========== 配置参数 ==========
GPU_USE=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$GPU_USE

# 模型和数据路径配置
outdir="${OUTDIR:-$PROJECT_ROOT/weights/ensemble/Qwen3-8B-Base}"
base_model="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
# 默认使用本仓库 dataprocess 脚本生成的数据
stage1_data_path="${TRAIN_DATA_PATH:-$PROJECT_ROOT/dataprocess/am_deepseek_r1_filtered_ad.jsonl}"
data_files="${TRAIN_DATA_PATH:-$PROJECT_ROOT/dataprocess/am_deepseek_r1_filtered_ad.jsonl}"

# 训练参数配置
stage1_epochs=1
stage2_epochs=1
stage3_epochs=1

# BrownBoost 超参数
alpha=0.1
beta=0.8
gamma=0.1
easy_quantile=0.2
hard_quantile=0.8
patience=2
easy_patience=2
lambda_time=1.0
lambda_easy=1.0
sample_multiplier_stage2=1.0
sample_multiplier_stage3=1.0
model_type="wmss"
freeze=false
stage3_name="stage3_fused_brownboost_freeze${freeze}_${model_type}"

# 工作目录
cd "$SCRIPT_DIR"
entropy_dir="$outdir/entropy"

# ========== 辅助函数 ==========
get_latest_checkpoint() {
    local dir=$1
    if [ -d "$dir" ]; then
        local latest=$(find "$dir" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -n "$latest" ]; then
            echo "$latest"
        else
            echo "$dir"
        fi
    else
        echo "$dir"
    fi
}

wait_and_clear_gpu() {
    echo "等待 30 秒以便 GPU 清理..."
    sleep 30
    show_gpu_usage
}

# ========== 开始流程 ==========
echo "=========================================="
echo "开始 Ensemble LLM 三阶段训练 Pipeline"
echo "输出目录: $outdir"
echo "=========================================="

conda activate qwen

# 创建必要的目录
mkdir -p "$entropy_dir"
mkdir -p "$outdir"

# ========== 步骤 0: 计算 base 模型的 entropy_0 ==========
echo ""
echo "=========================================="
echo "步骤 0: 计算 base 模型的 entropy_0"
echo "=========================================="

entropy_0_path="$entropy_dir/entropy_0.jsonl"
conda activate qwen

# 注意：需要确保 run_entropy.py 在路径中
# 如果不在，请修改为正确的路径
# accelerate launch \
#     --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
#     ../ensemble/run_entropy.py compute \
#     --model_path "$base_model" \
#     --data_file "$data_files" \
#     --output_path "$entropy_0_path" \
#     --entropy_field "entropy_0" \
#     --stage "stage0"

# ========== 步骤 1: Stage 1 训练 -> m1 ==========
echo ""
echo "=========================================="
echo "步骤 1: Stage 1 训练 -> m1"
echo "=========================================="
# conda activate qwen
# accelerate launch \
#     --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
#     ../ensemble/ensemble_train.py \
#     --stage 1 \
#     --model-name "$base_model" \
#     --stage1-data-path "$stage1_data_path" \
#     --data-files "$data_files" \
#     --output-dir "$outdir" \
#     --wandb-project "ensemble-math" \
#     --wandb-run-name "qwen3-ensemble" \
#     --per-device-train-batch-size 1 \
#     --grad-accum 32 \
#     --max-seq-length 4096 \
#     --use-chat-template True \
#     --stage1-num-epochs $stage1_epochs \
#     --alpha $alpha \
#     --beta $beta \
#     --gamma $gamma \
#     --easy-quantile $easy_quantile \
#     --hard-quantile $hard_quantile \
#     --patience $patience \
#     --easy-patience $easy_patience \
#     --lambda-time $lambda_time \
#     --lambda-easy $lambda_easy \
#     --sample-multiplier-stage2 $sample_multiplier_stage2 \
#     --sample-multiplier-stage3 $sample_multiplier_stage3 \
#     --entropy-results ""


# ========== 步骤 2: 计算 m1 的 entropy_1 ==========
echo ""
echo "=========================================="
echo "步骤 2: 计算 m1 的 entropy_1"
echo "=========================================="

m1_dir="$outdir/stage1_m1"
m1_checkpoint=$(get_latest_checkpoint "$m1_dir")

entropy_1_path="$entropy_dir/entropy_1.jsonl"
echo "使用 m1 checkpoint: $m1_checkpoint"
conda activate qwen
# accelerate launch \
#     --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
#     ../ensemble/run_entropy.py compute \
#     --model_path "$m1_checkpoint" \
#     --data_file "$data_files" \
#     --output_path "$entropy_1_path" \
#     --entropy_field "entropy_1" \
#     --stage "stage1"

# ========== 步骤 3: 合并 entropy_0 和 entropy_1 ==========
echo ""
echo "=========================================="
echo "步骤 3: 合并 entropy_0 和 entropy_1"
echo "=========================================="

entropy_merged_stage1="$entropy_dir/entropy_merged_stage1.jsonl"

# python ../ensemble/run_entropy.py merge \
#     --input_files "$entropy_0_path" "$entropy_1_path" \
#     --output_path "$entropy_merged_stage1"

echo "合并完成: $entropy_merged_stage1"

# ========== 步骤 3.5: 测试 Stage 1 的 m1 模型 ==========
echo ""
echo "=========================================="
echo "步骤 3.5: 评测 Stage 1 训练的 m1 模型"
echo "=========================================="

# 论文主评测的默认协议：thinking on, seed=42, greedy (T=0, top-p=1), n=1。
# EVAL_DATA_ROOT 可指向一个包含同名子目录的外部数据根目录。本仓库不附带 Math500，
# 因此未提供 math500/test.jsonl 时会明确跳过该项。
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/dataprocess/test_dataset}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PROJECT_ROOT/results}"
EVAL_MATH500_PATH="${EVAL_MATH500_PATH:-$EVAL_DATA_ROOT/math500/test.jsonl}"
EVAL_DATASETS=(
    "amc23|$EVAL_DATA_ROOT/amc23/test.json"
    "math500|$EVAL_MATH500_PATH"
    "aime2025|$EVAL_DATA_ROOT/aime2025/test.json"
    "mawps|$EVAL_DATA_ROOT/mawps/test.json"
    "AQuA|$EVAL_DATA_ROOT/AQuA/test.json"
    "gsm8k|$EVAL_DATA_ROOT/gsm8k/test.json"
    "SVAMP|$EVAL_DATA_ROOT/SVAMP/test.json"
)
EVAL_TP="${EVAL_TP:-1}"
EVAL_MODE="greedy"
EVAL_TEMPERATURE="0"
EVAL_TOP_P="1"
EVAL_NUM_SAMPLES="1"
EVAL_SEED="42"
EVAL_MAX_INPUT_TOKENS="4096"

run_evaluation_suite() {
    local eval_model="$1"
    local eval_label="$2"
    local dataset_config dataset_name dataset_path
    local max_new_tokens max_model_len output_dir
    local failures=0

    for dataset_config in "${EVAL_DATASETS[@]}"; do
        IFS='|' read -r dataset_name dataset_path <<< "$dataset_config"

        if [ ! -f "$dataset_path" ]; then
            echo "  ↷ 跳过 $dataset_name：找不到 $dataset_path"
            continue
        fi

        case "$dataset_name" in
            aime2025|amc23)
                max_new_tokens=8192
                max_model_len=12288
                ;;
            *)
                max_new_tokens=4096
                max_model_len=8192
                ;;
        esac

        output_dir="$EVAL_OUTPUT_ROOT/$eval_label/$dataset_name/paper-greedy-t0-n1-seed42"
        mkdir -p "$output_dir"
        echo "  → 评测数据集: $dataset_name ..."

        if CUDA_VISIBLE_DEVICES=$GPU_USE python "$PROJECT_ROOT/ensemble/eval_vllm_thinking_math.py" \
            --dataset "$dataset_path" \
            --model "$eval_model" \
            --dataset-name "$dataset_name" \
            --tp "$EVAL_TP" \
            --protocol paper \
            --mode "$EVAL_MODE" \
            --temperature "$EVAL_TEMPERATURE" \
            --top-p "$EVAL_TOP_P" \
            --num-samples "$EVAL_NUM_SAMPLES" \
            --thinking \
            --seed "$EVAL_SEED" \
            --max-input-tokens "$EVAL_MAX_INPUT_TOKENS" \
            --max-new-tokens "$max_new_tokens" \
            --max-model-len "$max_model_len" \
            --output-dir "$output_dir"; then
            echo "  ✓ $dataset_name 评测完成"
        else
            echo "⚠️  警告: 数据集 $dataset_name 评测失败"
            failures=1
        fi
        echo ""
    done
    return "$failures"
}

conda activate verl_dev

echo "开始评测 Stage 1 的 m1 模型: $m1_checkpoint"
echo ""

# 如需评测 Stage 1，取消下一行注释。
# run_evaluation_suite "$m1_checkpoint" "stage1_m1"

echo "✓ Stage 1 m1 模型评测完成"
wait_and_clear_gpu

# ========== 步骤 4: 制作 stage0_m0 ==========
echo ""
echo "=========================================="
echo "步骤 4: 复制 base 模型到 stage0_m0"
echo "=========================================="

# 确保 outdir 是绝对路径（脚本已切换到脚本所在目录）
if [[ ! "$outdir" = /* ]]; then
    # 如果是相对路径，转换为绝对路径
    outdir_abs="$(realpath "$outdir")"
else
    outdir_abs="$outdir"
fi

conda activate qwen
python ../ensemble/copymodel.py \
    --model-name "$base_model" \
    --output-dir "$outdir_abs"

if [ $? -ne 0 ]; then
    echo "❌ 复制模型失败"
    exit 1
fi

echo "✓ stage0_m0 模型准备完成: $outdir_abs/stage0_m0"

# ========== 步骤 5: Stage 3 训练 -> 最终模型 ==========
echo ""
echo "=========================================="
echo "步骤 5: Stage 3 训练 -> 最终模型"
echo "=========================================="
conda activate qwen
accelerate launch \
   --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
   ../ensemble/ensemble_train.py \
   --stage 3 \
   --model-name "$base_model" \
   --stage1-data-path "$stage1_data_path" \
   --data-files "$data_files" \
   --output-dir "$outdir" \
   --wandb-project "ensemble-math" \
   --wandb-run-name "qwen3-ensemble" \
   --per-device-train-batch-size 1 \
   --grad-accum 32 \
   --max-seq-length 4096 \
   --use-chat-template True \
   --stage3-num-epochs $stage3_epochs \
   --m1-path "stage0_m0" \
   --m2-path "stage1_m1" \
   --entropy-results "$entropy_merged_stage1" \
   --alpha $alpha \
   --beta $beta \
   --gamma $gamma \
   --easy-quantile $easy_quantile \
   --hard-quantile $hard_quantile \
   --patience $patience \
   --easy-patience $easy_patience \
   --lambda-time $lambda_time \
   --lambda-easy $lambda_easy \
   --sample-multiplier-stage2 $sample_multiplier_stage2 \
   --sample-multiplier-stage3 $sample_multiplier_stage3 \
   --stage3-name $stage3_name \
   --model-type $model_type \
   --freeze-first-model $freeze


# ========== 步骤 6: 提取 current/strong 子模型 ==========
echo ""
echo "=========================================="
echo "步骤 6: 从最终模型中提取 current/strong 子模型"
echo "=========================================="

final_model_dir="$outdir/$stage3_name"
final_checkpoint=$(get_latest_checkpoint "$final_model_dir")

if [ -z "$final_checkpoint" ] || [ "$final_checkpoint" = "$final_model_dir" ]; then
    echo "⚠️  警告: 未找到 checkpoint，使用模型目录: $final_model_dir"
    final_checkpoint="$final_model_dir"
fi

extracted_model_dir="$outdir/${stage3_name}_extracted_m1"
echo "从融合模型提取 current/strong 子模型:"
echo "  - 输入模型: $final_checkpoint"
echo "  - 输出目录: $extracted_model_dir"
echo "  - 子模型索引: 1 (stage1/current branch)"

conda activate qwen
python ../ensemble/extract_submodel.py \
    --input "$final_checkpoint" \
    --output "$extracted_model_dir" \
    --submodel_idx 1 \
    --dtype bfloat16

if [ $? -ne 0 ]; then
    echo "❌ 提取子模型失败"
    exit 1
fi

echo "✓ 子模型提取完成: $extracted_model_dir"
wait_and_clear_gpu

# ========== 步骤 7: 测试提取的模型 ==========
echo ""
echo "=========================================="
echo "步骤 7: 测试提取的 current/strong 子模型"
echo "=========================================="

conda activate verl_dev

echo "开始评测提取的模型: $extracted_model_dir"
echo ""

run_evaluation_suite "$extracted_model_dir" "${stage3_name}_extracted_m1"

wait_and_clear_gpu

# ========== 完成 ==========
echo ""
echo "=========================================="
echo "✓ 全部 Pipeline 完成！"
echo "=========================================="
echo "模型保存位置:"
echo "  - Stage1 (m1): $m1_checkpoint"
echo "  - Stage0 (m0): $outdir/stage0_m0"
echo "  - Stage3 (final): $final_checkpoint"
echo "  - 提取的 current/strong 子模型 (m1): $extracted_model_dir"
echo ""
echo "Entropy 文件:"
echo "  - entropy_0: $entropy_0_path"
echo "  - entropy_1: $entropy_1_path"
echo "  - 合并 (Stage1): $entropy_merged_stage1"
echo ""
echo "评测结果保存在: $EVAL_OUTPUT_ROOT"
echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志文件: $LOG_FILE"
echo "=========================================="
