#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_matrices_v2 import extract_and_save_matrices

if __name__ == "__main__":
    # 测试处理1.jsonl文件
    extract_and_save_matrices("1.jsonl", "1_adj_matrices")
    print("测试完成！")