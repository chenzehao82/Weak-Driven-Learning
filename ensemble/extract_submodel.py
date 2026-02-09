##!/usr/bin/env python3
"""
Extract specified sub-model from ensemble model and save as independent model.

Usage:
    python extract_submodel.py --input <ensemble_model_path> --output <save_dir> [--submodel_idx 1] [--dtype bfloat16]

Example:
    python extract_submodel.py \
        --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \
        --output weights/llmboost/Qwen3-4B-Base/stage3_m3 \
        --submodel_idx 1
"""

import os
import sys
import argparse

## Delay imports, parse parameters first (so --help works without dependencies)
def import_dependencies():
    """Delay import dependencies"""
    import torch
    
    ## Add project root to sys.path (parent directory)
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_current_dir)  ## 
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    
    ## Import extract_submodel function
    try:
        from utils.fuse_models import extract_submodel
        return extract_submodel, torch
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure:")
        print("  1. Correct conda environment is activated (e.g.: conda activate qwen)")
        print("  2. utils/ Directory exists in project root")
        print(f"  3. Current working directory: {os.getcwd()}")
        print(f"  4. Script location: {_current_dir}")
        print(f"  5. Project root: {_parent_dir}")
        print(f"  6. Python path: {sys.path[:3]}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract specified sub-model from ensemble model and save as independent model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  ## Extract sub_models.1 (second model)
  python extract_submodel.py \\
      --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \\
      --output weights/llmboost/Qwen3-4B-Base/stage3_m3

  ## Extract sub_models.0 (first model)
  python extract_submodel.py \\
      --input weights/llmboost/Qwen3-4B-Base/stage3_fused_brownboost/checkpoint-436 \\
      --output weights/llmboost/Qwen3-4B-Base/stage3_m0 \\
      --submodel_idx 0

  ## Specify data type
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
        help="Ensemble model path (can be checkpoint directory or model directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/root/buaa/czh/EnsembleLLM/weights/llmboost-code/Qwen3-8B-Base/stage3",
        help="Save directory for extracted sub-model"
    )
    
    parser.add_argument(
        "--submodel_idx",
        type=int,
        default=1,
        help="Sub-model index to extract (0 for first, 1 for second, default: 1）"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Saved weight precision (default: bfloat16）"
    )
    
    return parser.parse_args()


def get_torch_dtype(dtype_str: str, torch):
    """Convert string to torch dtype"""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def main():
    """Main function"""
    args = parse_args()
    
    ## Delay import dependencies（）
    extract_submodel, torch = import_dependencies()
    
    ## Validate input path
    if not os.path.exists(args.input):
        print(f"❌ Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    ## Check if it's a checkpoint directory
    if os.path.isdir(args.input):
        ## Check whether there is config.json or pytorch_model.bin
        has_config = os.path.exists(os.path.join(args.input, "config.json"))
        has_model = os.path.exists(os.path.join(args.input, "pytorch_model.bin")) or \
                   any(f.startswith("pytorch_model") for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f)))
        
        if not (has_config or has_model):
            print(f"⚠️  Warning: Input directory does not seem to be a valid model directory")
            print(f"   Did not find config.json or pytorch_model.bin")
            print(f"   Continue attempting to load...")
    
    ## Convert dtype
    torch_dtype = get_torch_dtype(args.dtype, torch)
    
    ## Print parameter information
    print("=" * 60)
    print("Extract sub-model configuration")
    print("=" * 60)
    print(f"  Input model: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  Sub-model index: {args.submodel_idx} (sub_models.{args.submodel_idx})")
    print(f"  Weight precision: {args.dtype}")
    print("=" * 60)
    print()
    
    ## Call extract_submodel function
    try:
        saved_path = extract_submodel(
            ensemble_model_path=args.input,
            submodel_idx=args.submodel_idx,
            save_dir=args.output,
            torch_dtype=torch_dtype
        )
        
        print()
        print("=" * 60)
        print("✅ Extraction complete!")
        print("=" * 60)
        print(f"  Save path: {saved_path}")
        print(f"  Output directory: {args.output}")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Extraction failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

