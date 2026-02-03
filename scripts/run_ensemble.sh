#!/bin/bash
set -e  # 遇到错误立即退出

# ========== 环境配置 ==========
eval "$(conda shell.bash hook)"
# export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/buaa/hf_cache"

# ========== 日志配置 ==========
LOG_DIR="./logs"
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
outdir="../weights/ensemble/Qwen3-4B-Base"
base_model="Qwen/Qwen3-4B-Base"
# 默认使用本仓库 dataprocess 脚本生成的数据
stage1_data_path="/root/buaa/czh/Weak-Driving Learning/dataprocess/am_deepseek_r1_filtered_ad.jsonl"
data_files="/root/buaa/czh/Weak-Driving Learning/dataprocess/am_deepseek_r1_filtered_ad.jsonl"

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
cd "$(dirname "$0")"
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
accelerate launch \
    --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
    ../ensemble/run_entropy.py compute \
    --model_path "$base_model" \
    --data_file "$data_files" \
    --output_path "$entropy_0_path" \
    --entropy_field "entropy_0" \
    --stage "stage0"

# ========== 步骤 1: Stage 1 训练 -> m1 ==========
echo ""
echo "=========================================="
echo "步骤 1: Stage 1 训练 -> m1"
echo "=========================================="
conda activate qwen
accelerate launch \
    --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
    ../ensemble/ensemble_train.py \
    --stage 1 \
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
    --stage1-num-epochs $stage1_epochs \
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
    --entropy-results ""

wait_and_clear_gpu

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
accelerate launch \
    --config_file=/root/buaa/czh/EnsembleLLM/scripts/accelerate_config.yaml \
    ../ensemble/run_entropy.py compute \
    --model_path "$m1_checkpoint" \
    --data_file "$data_files" \
    --output_path "$entropy_1_path" \
    --entropy_field "entropy_1" \
    --stage "stage1"

wait_and_clear_gpu

# ========== 步骤 3: 合并 entropy_0 和 entropy_1 ==========
echo ""
echo "=========================================="
echo "步骤 3: 合并 entropy_0 和 entropy_1"
echo "=========================================="

entropy_merged_stage1="$entropy_dir/entropy_merged_stage1.jsonl"

python ../ensemble/run_entropy.py merge \
    --input_files "$entropy_0_path" "$entropy_1_path" \
    --output_path "$entropy_merged_stage1"

echo "合并完成: $entropy_merged_stage1"
# ========== 步骤 4: 制作 stage0_m0 ==========
python ../ensemble/copymodel.py

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

wait_and_clear_gpu

# ========== 步骤 6: 提取第一个子模型 ==========
echo ""
echo "=========================================="
echo "步骤 6: 从最终模型中提取第一个子模型"
echo "=========================================="

final_model_dir="$outdir/$stage3_name"
final_checkpoint=$(get_latest_checkpoint "$final_model_dir")

if [ -z "$final_checkpoint" ] || [ "$final_checkpoint" = "$final_model_dir" ]; then
    echo "⚠️  警告: 未找到 checkpoint，使用模型目录: $final_model_dir"
    final_checkpoint="$final_model_dir"
fi

extracted_model_dir="$outdir/${stage3_name}_extracted_m0"
echo "从融合模型提取第一个子模型:"
echo "  - 输入模型: $final_checkpoint"
echo "  - 输出目录: $extracted_model_dir"
echo "  - 子模型索引: 0 (第一个模型)"

conda activate qwen
python ../ensemble/extract_submodel.py \
    --input "$final_checkpoint" \
    --output "$extracted_model_dir" \
    --submodel_idx 0 \
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
echo "步骤 7: 测试提取的第一个子模型"
echo "=========================================="

# 测试数据集配置（可以根据需要修改）
EVAL_DATASETS=(
    "amc23|/root/buaa/czh/dataset/test_dataset/amc23/test.json|json"
    "math500|/root/buaa/czh/dataset/test_dataset/math500/test.jsonl|jsonl"
    "aime2025|/root/buaa/czh/dataset/test_dataset/aime2025/test.jsonl|jsonl"
    "mawps|/root/buaa/czh/dataset/test_dataset/mawps/test.json|json"
    "AQuA|/root/buaa/czh/dataset/test_dataset/AQuA/test.json|json"
    "gsm8k|/root/buaa/czh/dataset/test_dataset/gsm8k/test.json|json"
    "SVAMP|/root/buaa/czh/dataset/test_dataset/SVAMP/test.json|json"
)
EVAL_TP=8  # tensor parallel size for evaluation
EVAL_REPEAT=3  # repeat times for evaluation

conda activate verl_dev

echo "开始评测提取的模型: $extracted_model_dir"
echo ""

for dataset_config in "${EVAL_DATASETS[@]}"; do
    IFS='|' read -r dataset_name dataset_path file_type <<< "$dataset_config"
    echo "  → 评测数据集: $dataset_name ..."
    
    CUDA_VISIBLE_DEVICES=$GPU_USE python ../ensemble/eval_vllm_thinking_math.py \
        --dataset "$dataset_path" \
        --model "$extracted_model_dir" \
        --tp $EVAL_TP \
        --epoch "${stage3_name}_extracted_m0" \
        --repeat $EVAL_REPEAT \
        --dataset_name "$dataset_name"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  警告: 数据集 $dataset_name 评测失败"
    else
        echo "  ✓ $dataset_name 评测完成"
    fi
    echo ""
done

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
echo "  - 提取的子模型 (m0): $extracted_model_dir"
echo ""
echo "Entropy 文件:"
echo "  - entropy_0: $entropy_0_path"
echo "  - entropy_1: $entropy_1_path"
echo "  - 合并 (Stage1): $entropy_merged_stage1"
echo ""
echo "评测结果保存在: results/ 目录下"
echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "日志文件: $LOG_FILE"
echo "=========================================="

