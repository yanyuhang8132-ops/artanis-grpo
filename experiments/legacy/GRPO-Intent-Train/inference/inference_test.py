"""
推理脚本：加载 LoRA 微调后的 LLaMA 模型，对 test.json 中的一条样本进行推理。
并保存与 ground truth 的对比输出。
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === 模型路径 ===
base_model = "INFOCOM26-Intent-Model/model/LLM-Research/Meta-Llama-3-8B-Instruct"
lora_path = "INFOCOM26-Intent-Model/output/llama3_network_config_lora/adapter"

# === 加载模型和 LoRA 权重 ===
print("[INFO] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("[INFO] Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# === 加载 tokenizer ===
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

# === 加载测试数据（test.json） ===
with open("INFOCOM26-Intent-Model/data/test_data/test.json", "r") as f:
    data = json.load(f)
sample = data[0]
prompt = sample["instruction"] + "\n" + sample["input"] if sample["input"] else sample["instruction"]
ground_truth = sample.get("output", None)

# === 执行推理 ===
print("[INFO] Generating output...")
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=4096,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === 打印与保存结果 ===
print("\n====== PROMPT ======")
print(prompt)
print("\n====== MODEL OUTPUT ======")
print(generated_text)

# 保存生成结果 + GT
output_path = "GRPO-Intent-Train/output/generated_output.json"
with open(output_path, "w") as f:
    json.dump({
        "prompt": prompt,
        "generated_output": generated_text,
        "ground_truth": ground_truth
    }, f, ensure_ascii=False, indent=2)
print(f"\n[INFO] Output saved to {output_path}")
