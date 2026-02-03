#!/usr/bin/env python3
"""
GPU 内存清理工具

用法：
    python clear_gpu.py
"""

import gc
import torch
import time


def clear_gpu_memory():
    """彻底清理 GPU 内存"""
    print("=" * 60)
    print("开始清理 GPU 内存...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("未检测到 CUDA 设备")
        return
    
    # 打印清理前的状态
    print("\n清理前 GPU 显存使用情况:")
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: 已分配 {mem_allocated:.2f}GB, 已保留 {mem_reserved:.2f}GB")
    
    # 执行清理
    print("\n执行清理操作...")
    
    # 1. 同步所有 CUDA 流
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
    
    # 2. 清空缓存
    torch.cuda.empty_cache()
    
    # 3. 强制垃圾回收
    gc.collect()
    
    # 4. 再次清空缓存
    time.sleep(1)
    torch.cuda.empty_cache()
    
    # 5. 再次垃圾回收
    gc.collect()
    
    # 6. 最后一次清空
    torch.cuda.empty_cache()
    
    # 打印清理后的状态
    print("\n清理后 GPU 显存使用情况:")
    for i in range(torch.cuda.device_count()):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: 已分配 {mem_allocated:.2f}GB, 已保留 {mem_reserved:.2f}GB")
    
    print("\n" + "=" * 60)
    print("✓ GPU 内存清理完成")
    print("=" * 60)


if __name__ == "__main__":
    clear_gpu_memory()

