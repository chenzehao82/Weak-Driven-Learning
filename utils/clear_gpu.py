##!/usr/bin/env python3
"""
GPU 

：
    python clear_gpu.py
"""

import gc
import torch
import time


def clear_gpu_memory():
    """Thoroughly clean GPU memory"""
    print("=" * 60)
    print("Starting GPU memory cleaning...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA device not detected")
        return
    
    ## 
    print("\nBefore cleanup GPU memory usage:")
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: Allocated {mem_allocated:.2f}GB, Reserved {mem_reserved:.2f}GB")
    
    ## 
    print("\nExecuting cleanup operations...")
    
    ## 1. Synchronize all CUDA streams
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
    
    ## 2. Empty cache
    torch.cuda.empty_cache()
    
    ## 3. Force garbage collection
    gc.collect()
    
    ## 4. Empty cache
    time.sleep(1)
    torch.cuda.empty_cache()
    
    ## 5. Another garbage collection
    gc.collect()
    
    ## 6. Final empty
    torch.cuda.empty_cache()
    
    ## 
    print("\nAfter cleanup GPU memory usage:")
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: Allocated {mem_allocated:.2f}GB, Reserved {mem_reserved:.2f}GB")
    
    print("\n" + "=" * 60)
    print("✓ GPU memory cleanup complete")
    print("=" * 60)


if __name__ == "__main__":
    clear_gpu_memory()

