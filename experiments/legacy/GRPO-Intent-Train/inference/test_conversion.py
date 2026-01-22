"""
测试推理结果转换为验证器输入格式
"""

import sys
import os
import json

# 添加项目路径
sys.path.append('/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/inference')
sys.path.append('/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train')

from Inference2Sample import convert_to_samples, convert_tensors_to_lists

def main():
    inference_path = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/inference_5_samples.jsonl"
    
    print("=== 测试推理结果转换 ===")
    
    try:
        # 转换样本
        samples = convert_to_samples(inference_path, max_samples=2)
        print(f"✓ 成功转换 {len(samples)} 个样本")
        
        # 检查样本结构
        for i, sample in enumerate(samples):
            print(f"\n--- 样本 {i+1} ---")
            print(f"图特征: {list(sample['graph'].keys())}")
            print(f"非图特征: {list(sample['non_graph'].keys())}")
            print(f"标签: {list(sample['labels'].keys())}")
            print(f"邻接矩阵形状: {sample['graph']['adj'].shape}")
            print(f"掩码形状: {sample['mask'].shape}")
            print(f"流量数量: {len(sample['non_graph']['traffic_flows'])}")
            
            # 检查生成的配置
            gen_config = sample['non_graph']['generated_config']
            print(f"生成配置 - LB: {len(gen_config.get('lb_rules', {}))}, ACL: {len(gen_config.get('acl_rules', {}))}")
        
        # 保存转换结果
        output_path = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/converted_samples.json"
        serializable_samples = [convert_tensors_to_lists(s) for s in samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 转换结果已保存到: {output_path}")
        
        # 尝试验证器计算
        try:
            from reward_fn import compute_reward
            print("\n=== 测试验证器 ===")
            
            for i, sample in enumerate(samples):
                try:
                    reward = compute_reward(sample)
                    print(f"样本 {i+1} 奖励: {reward:.4f}")
                except Exception as e:
                    print(f"样本 {i+1} 计算奖励失败: {e}")
                    
        except ImportError:
            print("\n! 警告: 无法导入 reward_fn 模块")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())