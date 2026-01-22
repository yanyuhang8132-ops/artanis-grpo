"""
迭代推理和验证脚本
对每条测试数据进行推理，然后验证SLA是否满足，如果不满足则重新推理，最多4次
统计获得通过结果时平均需要多少次推理
"""

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import re
import networkx as nx
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import PeftModel

# 导入 script_v2.py 中的类和函数
import sys
sys.path.append('/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/inference')
from script_v2 import (
    PromptEncoder, 
    IntentModelWithPromptEncoder
)

# 导入 extract_matrices_v2.py 中的函数
sys.path.append('/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output')
from extract_matrices_v2 import (
    parse_topology,
    extract_sla_intent,
    check_sla_satisfaction,
    parse_device_configurations,
    parse_model_output,
    standardize_model_middleboxes,
    compare_rules,
    simulate_acl_effect,
    simulate_slb_effect
)


class IterativeInferenceEvaluator:
    def __init__(self, model_path, adapter_path, test_data_path, max_attempts=4):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.test_data_path = test_data_path
        self.max_attempts = max_attempts
        self.tokenizer = None
        self.model = None
        self.results = []
        
        # 定义四种不同的参数组合
        self.param_combinations = [
            {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9, "do_sample": True},   # 默认参数
            {"max_new_tokens": 512, "temperature": 0.5, "top_p": 0.8, "do_sample": True},   # 更保守
            {"max_new_tokens": 512, "temperature": 0.9, "top_p": 0.95, "do_sample": True},  # 更随机
            {"max_new_tokens": 512, "temperature": 0.3, "top_p": 0.7, "do_sample": True},   # 最保守
        ]
    
    def generation_prompt_func(self, example):
        """生成推理用的prompt"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

你是一个网络配置助手，请根据网络状态和意图生成配置更新。
<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        tokens = self.tokenizer(prompt, add_special_tokens=False)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
        
    def load_model(self):
        """加载模型和tokenizer"""
        print("=== 加载模型 ===")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=False, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        base_model = PeftModel.from_pretrained(base_model, self.adapter_path)
        base_model.print_trainable_parameters()

        self.model = IntentModelWithPromptEncoder(
            base_model, self.tokenizer, prefix_length=10, max_model_length=2048
        )
        self.model.gradient_checkpointing_enable()
        print("模型加载完成")
        
    def load_test_data(self):
        """加载测试数据"""
        print("=== 加载测试数据 ===")
        df = pd.read_json(self.test_data_path)
        ds = Dataset.from_pandas(df)
        
        # 使用实例方法处理数据
        eval_subset = ds.map(self.generation_prompt_func, remove_columns=ds.column_names)
        original_subset = ds
        
        # 选择测试样本数量（可以调整）
        sample_count = min(50, len(eval_subset))  # 先用50条测试
        eval_subset = eval_subset.select(range(sample_count))
        original_subset = original_subset.select(range(sample_count))
        
        print(f"测试样本数量: {len(eval_subset)}")
        return eval_subset, original_subset
        
    def single_inference(self, example, attempt_idx=0):
        """单次推理 - 使用不同的参数组合"""
        # 根据尝试次数选择不同的参数组合
        params = self.param_combinations[attempt_idx % len(self.param_combinations)]
        
        input_ids = torch.tensor([example["input_ids"]], device=self.model.base_model.device)
        attention_mask = torch.tensor([example["attention_mask"]], device=self.model.base_model.device)

        with torch.no_grad():
            outputs = self.model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=params["max_new_tokens"],
                do_sample=params["do_sample"],
                top_p=params["top_p"],
                temperature=params["temperature"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # 提取 assistant 部分
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            generated_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        elif "assistant\n" in generated_text:
            generated_response = generated_text.split("assistant\n")[-1].strip()
        elif len(generated_text) > len(input_text):
            generated_response = generated_text[len(input_text):].strip()
        else:
            generated_response = generated_text

        return generated_response, params
        
    def validate_sla(self, user_input, model_output, adjusted_sla=None):
        """验证SLA是否满足"""
        try:
            # 解析用户输入
            config_match = re.search(r"\[Device Configurations\]\n(.*?)\n\#", user_input, re.DOTALL)
            device_match = re.search(r"\[Device List\]\n(.+?)\n\n", user_input, re.DOTALL)
            topology_match = re.search(r"\[Topology Info\]\n(.+?)\n\#\#", user_input, re.DOTALL)
            
            if not all([config_match, device_match, topology_match]):
                return False, "解析失败", None
                
            device_names = [d.strip() for d in device_match.group(1).split(',')]
            topology_str = topology_match.group(1)
            config_text = config_match.group(1)

            adj_matrix, name_to_idx, ip_to_node = parse_topology(topology_str, device_names)
            
            # 获取SLA需求
            node_count = len(device_names)
            src, dst, original_sla = extract_sla_intent(user_input, node_count)
            if not src or not dst:
                return False, "SLA解析失败", None

            # 使用调整后的SLA或原始SLA
            sla_to_check = adjusted_sla if adjusted_sla is not None else original_sla

            # 解析现有配置和模型输出
            existing_acls, existing_slbs = parse_device_configurations(config_text)
            model_items = parse_model_output(model_output)
            model_acls, model_slbs = standardize_model_middleboxes(model_items)
            new_acls, new_slbs = compare_rules(model_acls, model_slbs, existing_acls, existing_slbs)

            # 模拟中间件效果
            adj_matrix_modified = adj_matrix.copy()
            simulate_acl_effect(adj_matrix_modified, name_to_idx, new_acls, ip_to_node)
            simulate_slb_effect(adj_matrix_modified, name_to_idx, new_slbs, ip_to_node)
            
            # 检查SLA满足情况
            satisfied, paths = check_sla_satisfaction(adj_matrix_modified, name_to_idx, src, dst, sla_to_check)
            
            return satisfied, f"原始SLA: {original_sla}, 检查SLA: {sla_to_check}, 满足: {satisfied}", original_sla
            
        except Exception as e:
            return False, f"验证出错: {str(e)}", None
    
    def iterative_inference_single_sample(self, example, original_example, sample_id):
        """对单个样本进行迭代推理"""
        user_input = original_example['instruction']
        reference = original_example['output']
        
        print(f"\n=== 处理样本 {sample_id} ===")
        print(f"输入摘要: {user_input[:100]}...")
        
        attempts = []
        current_sla = None  # 用于跟踪当前的SLA标准
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"  尝试 {attempt}/{self.max_attempts}")
            
            # 推理 - 传入attempt索引来选择不同的参数组合
            model_output, used_params = self.single_inference(example, attempt - 1)
            
            # 验证 - 如果不是第一次尝试，使用降低后的SLA标准
            is_satisfied, validation_msg, original_sla = self.validate_sla(user_input, model_output, current_sla)
            
            # 如果第一次尝试，记录原始SLA
            if attempt == 1 and original_sla is not None:
                current_sla = original_sla
            
            attempt_result = {
                "attempt": attempt,
                "model_output": model_output,
                "satisfied": is_satisfied,
                "validation_msg": validation_msg,
                "parameters_used": used_params,
                "sla_used": current_sla
            }
            attempts.append(attempt_result)
            
            print(f"    参数: temp={used_params['temperature']}, top_p={used_params['top_p']}")
            print(f"    当前SLA标准: {current_sla}")
            print(f"    结果: {validation_msg}")
            
            if is_satisfied:
                print(f"    ✓ 样本 {sample_id} 在第 {attempt} 次尝试成功")
                break
            else:
                if attempt < self.max_attempts and current_sla is not None:
                    current_sla = max(current_sla - 5, 5)  # 降低5，但不低于5
                    print(f"    SLA标准降低到: {current_sla}")
        else:
            print(f"    ✗ 样本 {sample_id} 在 {self.max_attempts} 次尝试后仍未成功")
        
        return {
            "sample_id": sample_id,
            "user_input": user_input,
            "reference": reference,
            "attempts": attempts,
            "final_success": any(a["satisfied"] for a in attempts),
            "success_attempt": next((a["attempt"] for a in attempts if a["satisfied"]), None),
            "original_sla": original_sla,
            "final_sla": current_sla
        }
    
    def run_evaluation(self):
        """运行完整评估"""
        self.load_model()
        eval_subset, original_subset = self.load_test_data()
        
        print(f"\n=== 开始迭代推理评估 ===")
        print(f"测试样本数: {len(eval_subset)}")
        print(f"最大尝试次数: {self.max_attempts}")
        
        self.model.eval()
        
        # 对每个样本进行迭代推理
        for i, (example, original_example) in enumerate(zip(eval_subset, original_subset)):
            result = self.iterative_inference_single_sample(example, original_example, i + 1)
            self.results.append(result)
        
        # 统计结果
        self.analyze_results()
        
    def analyze_results(self):
        """分析结果"""
        print(f"\n=== 结果分析 ===")
        
        total_samples = len(self.results)
        successful_samples = [r for r in self.results if r["final_success"]]
        success_rate = len(successful_samples) / total_samples
        
        print(f"总样本数: {total_samples}")
        print(f"成功样本数: {len(successful_samples)}")
        print(f"成功率: {success_rate:.2%}")
        
        if successful_samples:
            success_attempts = [r["success_attempt"] for r in successful_samples]
            avg_attempts = np.mean(success_attempts)
            print(f"成功样本平均尝试次数: {avg_attempts:.2f}")
            
            # 按尝试次数统计
            attempt_counts = {}
            for attempt in success_attempts:
                attempt_counts[attempt] = attempt_counts.get(attempt, 0) + 1
            
            print("成功样本按尝试次数分布:")
            for attempt in sorted(attempt_counts.keys()):
                count = attempt_counts[attempt]
                percentage = count / len(successful_samples)
                print(f"  第 {attempt} 次成功: {count} 个样本 ({percentage:.1%})")
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/iterative_results_{timestamp}.json"
        
        summary = {
            "evaluation_info": {
                "total_samples": total_samples,
                "successful_samples": len(successful_samples),
                "success_rate": success_rate,
                "max_attempts": self.max_attempts,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": self.results
        }
        
        if successful_samples:
            summary["evaluation_info"]["avg_attempts_for_success"] = float(np.mean(success_attempts))
            summary["evaluation_info"]["attempt_distribution"] = attempt_counts
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="迭代推理和验证脚本")
    parser.add_argument("--model_path", type=str, 
                       default="INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--adapter_path", type=str,
                       default="INFOCOM26-Intent-Model/output/llama3_network_config_lora/adapter", 
                       help="LoRA适配器路径")
    parser.add_argument("--test_data", type=str,
                       default="INFOCOM26-Intent-Model/data/test_data/test.json",
                       help="测试数据路径")
    parser.add_argument("--max_attempts", type=int, default=4,
                       help="每个样本最大尝试次数")
    
    args = parser.parse_args()
    
    evaluator = IterativeInferenceEvaluator(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        test_data_path=args.test_data,
        max_attempts=args.max_attempts
    )
    
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()