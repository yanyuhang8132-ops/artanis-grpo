"""
测试紧凑序列化功能
"""

import sys
import os
import json

# 添加路径
sys.path.append('/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/inference')

def test_compact_serialization():
    """测试紧凑序列化功能"""
    
    from Inference2Sample import convert_to_samples, convert_tensors_to_lists, custom_serializer, NumpyEncoder
    
    print("=== 测试紧凑序列化功能 ===")
    
    inference_path = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/inference_5_samples.jsonl"
    
    # 检查文件是否存在
    if not os.path.exists(inference_path):
        print(f"✗ 推理文件不存在: {inference_path}")
        return False
    
    try:
        # 转换样本
        samples = convert_to_samples(inference_path, max_samples=1)
        print(f"✓ 成功转换 {len(samples)} 个样本")
        
        if not samples:
            print("✗ 没有样本可以测试")
            return False
        
        sample = samples[0]
        
        # 转换tensor为list
        serializable_sample = convert_tensors_to_lists(sample)
        print("✓ 成功转换tensor为list")
        
        # 应用紧凑序列化
        compact_item = json.loads(
            json.dumps(serializable_sample, cls=NumpyEncoder, default=custom_serializer),
            object_hook=custom_serializer
        )
        print("✓ 成功应用紧凑序列化")
        
        # 检查序列化效果
        compact_fields = 0
        for section in ["graph", "non_graph", "labels"]:
            if section in compact_item:
                for field, value in compact_item[section].items():
                    if isinstance(value, str):
                        compact_fields += 1
        
        print(f"✓ 紧凑序列化字段数量: {compact_fields}")
        
        # 保存测试文件
        test_output = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/test_compact.json"
        with open(test_output, 'w', encoding='utf-8') as f:
            f.write("[\n")
            json.dump(compact_item, f, indent=2, separators=(',', ': '))
            f.write("\n]")
        
        print(f"✓ 测试文件已保存: {test_output}")
        
        # 验证文件可读性
        with open(test_output, 'r') as f:
            loaded_data = json.load(f)
        print(f"✓ 文件可正常读取，包含 {len(loaded_data)} 个样本")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_compact_serialization()
    print(f"\n{'=' * 40}")
    print(f"测试结果: {'成功' if success else '失败'}")
    exit(0 if success else 1)