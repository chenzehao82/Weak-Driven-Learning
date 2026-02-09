import argparse
import os
from accelerate import PartialState
from utils.compute_entropy import compute_entropy_for_model, merge_entropy_files

def main():
    parser = argparse.ArgumentParser(description="Standalone Entropy Computer")
    
    subparsers = parser.add_subparsers(dest="command", help="compute or merge")
    
    # : compute
    compute_parser = subparsers.add_parser("compute")
    compute_parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    compute_parser.add_argument("--data_file", type=str, required=True, help="Path to jsonl data file")
    compute_parser.add_argument("--output_path", type=str, required=True, help="Output jsonl path for entropy")
    compute_parser.add_argument("--entropy_field", type=str, default="entropy_0", help="Field name (entropy_0, entropy_1, etc.)")
    compute_parser.add_argument("--stage", type=str, default="stage0", help="stage0/stage1/stage2/stage3")

    # : merge
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input_files", type=str, nargs="+", required=True, help="List of entropy files to merge")
    merge_parser.add_argument("--output_path", type=str, required=True, help="Merged output path")

    args = parser.parse_args()
    distributed_state = PartialState()

    if args.command == "compute":
        compute_entropy_for_model(
            model_path=args.model_path,
            data_files=[args.data_file],
            output_path=args.output_path,
            entropy_field=args.entropy_field,
            distributed_state=distributed_state,
            stage=args.stage
        )
    
    elif args.command == "merge":
        # merge ，
        merge_entropy_files(args.input_files, args.output_path)

if __name__ == "__main__":
    main()
