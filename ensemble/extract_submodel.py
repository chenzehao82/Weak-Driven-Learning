#!/usr/bin/env python3
"""
从融合模型中提取指定的子模型并保存为独立模型。

用法:
    python extract_submodel.py --input <ensemble_model_path> --output <save_dir> [--submodel_idx 1] [--dtype bfloat16]

示例:
    python extract_submodel.py \
        --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \
        --output weights/llmboost/Qwen3-4B-Base/stage3_m3 \
        --submodel_idx 1
"""

import os
import sys
import argparse

# 延迟导入，先解析参数（这样 --help 可以在没有依赖的情况下工作）
def import_dependencies():
    """延迟导入依赖"""
    import torch
    
    # 添加项目根目录到 sys.path（父目录）
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_current_dir)  # 项目根目录
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    
    # 导入 extract_submodel 函数
    try:
        from utils.fuse_models import extract_submodel
        return extract_submodel, torch
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保:")
        print("  1. 已激活正确的 conda 环境（例如: conda activate qwen）")
        print("  2. utils/ 目录存在于项目根目录")
        print(f"  3. 当前工作目录: {os.getcwd()}")
        print(f"  4. 脚本位置: {_current_dir}")
        print(f"  5. 项目根目录: {_parent_dir}")
        print(f"  6. Python 路径: {sys.path[:3]}")
        sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从融合模型中提取指定的子模型并保存为独立模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取 sub_models.1 (第二个模型)
  python extract_submodel.py \\
      --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \\
      --output weights/llmboost/Qwen3-4B-Base/stage3_m3

  # 提取 sub_models.0 (第一个模型)
  python extract_submodel.py \\
      --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \\
      --output weights/llmboost/Qwen3-4B-Base/stage3_m0 \\
      --submodel_idx 0

  # 指定数据类型
  python extract_submodel.py \\
      --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \\
      --output weights/llmboost/Qwen3-4B-Base/stage3_m3 \\
      --dtype float16
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="/root/buaa/czh/EnsembleLLM/weights/llmboost-code/Qwen3-8B-Base/stage3_fused_brownboost_freezefalse_vote-base/checkpoint-406",
        help="融合模型路径（可以是 checkpoint 目录或模型目录）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/root/buaa/czh/EnsembleLLM/weights/llmboost-code/Qwen3-8B-Base/stage3",
        help="保存提取的子模型的目录路径"
    )
    
    parser.add_argument(
        "--submodel_idx",
        type=int,
        default=1,
        help="要提取的子模型索引（0表示第一个，1表示第二个，默认: 1）"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="保存的权重精度（默认: bfloat16）"
    )
    
    return parser.parse_args()


def get_torch_dtype(dtype_str: str, torch):
    """将字符串转换为 torch dtype"""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def main():
    """主函数"""
    args = parse_args()
    
    # 延迟导入依赖（在解析参数之后）
    extract_submodel, torch = import_dependencies()
    
    # 验证输入路径
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 检查是否是 checkpoint 目录
    if os.path.isdir(args.input):
        # 检查是否有 config.json 或 pytorch_model.bin
        has_config = os.path.exists(os.path.join(args.input, "config.json"))
        has_model = os.path.exists(os.path.join(args.input, "pytorch_model.bin")) or \
                   any(f.startswith("pytorch_model") for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)))
        
        if not (has_config or has_model):
            print(f"⚠️  警告: 输入目录似乎不是有效的模型目录")
            print(f"   未找到 config.json 或 pytorch_model.bin")
            print(f"   继续尝试加载...")
    
    # 转换 dtype
    torch_dtype = get_torch_dtype(args.dtype, torch)
    
    # 打印参数信息
    print("=" * 60)
    print("🔹 提取子模型配置")
    print("=" * 60)
    print(f"  输入模型: {args.input}")
    print(f"  输出目录: {args.output}")
    print(f"  子模型索引: {args.submodel_idx} (sub_models.{args.submodel_idx})")
    print(f"  权重精度: {args.dtype}")
    print("=" * 60)
    print()
    
    # 调用 extract_submodel 函数
    try:
        saved_path = extract_submodel(
            ensemble_model_path=args.input,
            submodel_idx=args.submodel_idx,
            save_dir=args.output,
            torch_dtype=torch_dtype
        )
        
        print()
        print("=" * 60)
        print("✅ 提取完成!")
        print("=" * 60)
        print(f"  保存路径: {saved_path}")
        print(f"  输出目录: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 提取失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

