import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


# === 补充函数: generation_prompt_func ===
def generation_prompt_func(example):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

你是一个网络配置助手，请根据网络状态和意图生成配置更新。
<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    tokens = tokenizer(prompt, add_special_tokens=False)
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }


# === Prompt Encoder ===
class PromptEncoder(nn.Module):
    def __init__(self, prefix_length, embedding_dim, hidden_dim=512):
        super().__init__()
        self.prefix_length = prefix_length
        self.embedding = nn.Embedding(prefix_length, embedding_dim)
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, batch_size):
        prefix_tokens = torch.arange(self.prefix_length).unsqueeze(0).expand(batch_size, -1).to(self.embedding.weight.device)
        prefix_embed = self.embedding(prefix_tokens)
        return self.transform(prefix_embed)


# === Wrap model with prompt encoder ===
class IntentModelWithPromptEncoder(nn.Module):
    def __init__(self, base_model, tokenizer, prefix_length=10, max_model_length=2048):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.tokenizer = tokenizer
        self.max_model_length = max_model_length
        self.embedding_dim = base_model.get_input_embeddings().embedding_dim
        self.prompt_encoder = PromptEncoder(prefix_length, self.embedding_dim)
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        prefix_embeddings = self.prompt_encoder(batch_size)  # [B, prefix_len, D]

        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)  # [B, T, D]
        inputs_embeds = torch.cat([prefix_embeddings, inputs_embeds], dim=1)  # [B, T+P, D]

        prefix_attention = torch.ones((batch_size, self.prefix_length), dtype=attention_mask.dtype).to(attention_mask.device)
        attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full((batch_size, self.prefix_length), -100).to(labels.device)
            labels = torch.cat([prefix_labels, labels], dim=1)

        # 拼接后统一截断到模型最大长度（保留末尾）
        total_len = inputs_embeds.shape[1]
        if total_len > self.max_model_length:
            trim = total_len - self.max_model_length
            inputs_embeds = inputs_embeds[:, trim:]
            attention_mask = attention_mask[:, trim:]
            if labels is not None:
                labels = labels[:, trim:]

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()


# === Preprocessing ===
def process_func(example):
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n你是一个网络配置助手，请根据网络状态和意图生成配置更新。\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# === Model Inference ===
def inference_with_fixed_parameters(model, tokenizer, eval_subset_prompt, original_subset, output_file):
    """
    使用固定参数进行模型推理
    """
    # 固定参数
    max_new_tokens = 512
    temperature = 0.7
    top_p = 0.9
    
    model.eval()
    model_outputs = []
    inference_times = []  # 记录每次推理时间
    
    print(f"\n=== 开始推理 ===")
    print(f"使用固定参数: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    
    total_start_time = time.time()  # 记录总开始时间
    
    for i, (example, original_example) in enumerate(zip(eval_subset_prompt, original_subset)):
        input_ids = torch.tensor([example["input_ids"]], device=model.base_model.device)
        attention_mask = torch.tensor([example["attention_mask"]], device=model.base_model.device)

        # 记录单次推理开始时间
        inference_start_time = time.time()
        
        with torch.no_grad():
            outputs = model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 记录单次推理结束时间
        inference_end_time = time.time()
        single_inference_time = inference_end_time - inference_start_time
        inference_times.append(single_inference_time)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # 提取 assistant 部分
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            generated_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        elif "assistant\n" in generated_text:
            generated_response = generated_text.split("assistant\n")[-1].strip()
        elif len(generated_text) > len(input_text):
            generated_response = generated_text[len(input_text):].strip()
        else:
            generated_response = generated_text

        # 获取真实答案
        true_answer = original_example['output']
        
        # 确保true_answer是字符串类型
        if isinstance(true_answer, dict):
            true_answer = str(true_answer)
        elif not isinstance(true_answer, str):
            true_answer = str(true_answer)
        
        # 记录模型输入输出
        model_output_record = {
            "id": i + 1,
            "user_input": original_example['instruction'],
            "reference": true_answer,
            "model_output": generated_response,
            "inference_time": round(single_inference_time, 3)  # 添加单次推理时间
            # "raw_input_text": input_text,
            # "raw_generated_text": generated_text
        }
        model_outputs.append(model_output_record)
        
        print(f"处理样本 #{i+1}/{len(eval_subset_prompt)} (用时: {single_inference_time:.3f}秒)")
        print(f"  输入: {original_example['instruction'][:100]}...")
        print(f"  参考答案: {true_answer[:100]}...")
        print(f"  模型输出: {generated_response[:100]}...")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 计算统计信息
    avg_inference_time = np.mean(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    print(f"\n=== 推理时间统计 ===")
    print(f"总推理时间: {total_time:.3f}秒")
    print(f"平均推理时间: {avg_inference_time:.3f}秒")
    print(f"最短推理时间: {min_inference_time:.3f}秒")
    print(f"最长推理时间: {max_inference_time:.3f}秒")
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in model_outputs:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return len(model_outputs), {
        'total_time': total_time,
        'avg_inference_time': avg_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'all_inference_times': inference_times
    }


# === Main Function ===
def main_inference(output_file=None):
    """
    主推理函数
    """
    global tokenizer
    
    # 设置输出文件
    if output_file is None:
        output_file = f"GRPO-Intent-Train/output/generated_output_30.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("=== 开始模型推理 ===")
    print(f"输出文件: {output_file}")
    
    # 加载模型
    print("\n=== 加载模型 ===")
    model_path = 'src/translator/legacy/INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    base_model = PeftModel.from_pretrained(base_model, "src/translator/legacy/INFOCOM26-Intent-Model/output/llama3_network_config_lora/adapter")

    base_model.print_trainable_parameters()

    model = IntentModelWithPromptEncoder(base_model, tokenizer, prefix_length=10, max_model_length=2048)
    model.gradient_checkpointing_enable()

    # 加载测试数据
    print("\n=== 加载测试数据 ===")
    df = pd.read_json('src/translator/legacy/INFOCOM26-Intent-Model/data/total_dataset/test_30.json')
    ds = Dataset.from_pandas(df)
    
    # 替换原来的数据处理方式，使用纯 prompt 输入
    eval_subset = ds.map(generation_prompt_func, remove_columns=ds.column_names)
    original_subset = ds
    
    # 使用全部测试数据，一共300条
    # eval_subset = eval_subset.select([i for i in list(range(130))])
    # original_subset = original_subset.select([i for i in list(range(130))])
    # print(f"测试样本数量: {len(eval_subset)}")
    
    # 只使用前5条测试数据
    eval_subset = eval_subset.select(range(5))
    original_subset = original_subset.select(range(5))
    print(f"测试样本数量: {len(eval_subset)}")

    # 执行推理
    num_samples, timing_stats = inference_with_fixed_parameters(
        model, tokenizer, eval_subset, original_subset, output_file
    )
    
    # 输出总结
    print(f"\n=== 推理完成 ===")
    print(f"结果已保存到: {output_file}")
    print(f"共处理样本数: {num_samples}")
    
    # 保存总结结果
    summary_file = output_file.replace('.jsonl', '_summary.json')
    summary = {
        'inference_info': {
            'total_samples': len(eval_subset),
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'timing_statistics': {
                'total_time': round(timing_stats['total_time'], 3),
                'average_inference_time': round(timing_stats['avg_inference_time'], 3),
                'min_inference_time': round(timing_stats['min_inference_time'], 3),
                'max_inference_time': round(timing_stats['max_inference_time'], 3)
            }
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"总结结果已保存到: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument("--output_file", type=str, default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    main_inference(output_file=args.output_file)