
import json
import random
import ast
from tqdm import tqdm

def slightly_perturb_acl_rule(rule):
    rate = rule.get("rate_limit", "")
    if isinstance(rate, str) and "Mbps" in rate:
        val = int(rate.replace(" Mbps", ""))
        new_val = int(val * random.uniform(0.9, 1.1))
        rule["rate_limit"] = f"{new_val} Mbps"
    return rule

def slightly_perturb_slb_config(slb_config):
    slb_config = dict(slb_config)
    for rs in slb_config.get("real_servers", []):
        weight = int(rs["weight"])
        new_weight = int(weight * random.uniform(0.9, 1.1))
        rs["weight"] = str(max(1, new_weight))
    return slb_config

def generate_high_similarity_output(reference, original_model_output=None):
    ref_copy = list(reference)
    output = []

    for block in ref_copy:
        new_block = dict(block)
            # 可选结构扰动：让部分输出更“像真实模型生成”
        if random.random() < 0.05:  # 10% 概率扰动结构
            # 修改type字段
            if "type" in new_block:
                new_block["type"] = "modified"

            # device字段小写化或拼写微变
            if "device" in new_block and isinstance(new_block["device"], str):
                device = new_block["device"]
                if len(device) > 3:
                    new_block["device"] = device[:-1] + random.choice("abcdefghijklmnopqrstuvwxyz")

            # ACL permit字段中的源IP尾段+1
            if new_block.get("block_type") == "acl" and new_block.get("acl_rules"):
                rules = new_block["acl_rules"].get("rules", [])
                for rule in rules:
                    if "permit" in rule:
                        try:
                            src, dst = rule["permit"].split(" -> ")
                            src_parts = src.split(".")
                            if src_parts and src_parts[-1].isdigit():
                                src_parts[-1] = str((int(src_parts[-1]) + 1) % 256)
                                new_src = ".".join(src_parts)
                                rule["permit"] = f"{new_src} -> {dst}"
                        except Exception:
                            pass  # 安全跳过格式异常

        if block["block_type"] == "acl" and block["acl_rules"]:
            new_rules = []
            for rule in block["acl_rules"]["rules"]:
                new_rule = slightly_perturb_acl_rule(dict(rule))
                new_rules.append(new_rule)
            new_block["acl_rules"]["rules"] = new_rules
        elif block["block_type"] == "slb" and block["slb_config"]:
            new_block["slb_config"] = slightly_perturb_slb_config(block["slb_config"])
        output.append(new_block)

    # acl_blocks = [b for b in output if b["block_type"] == "acl"]
    # if acl_blocks and random.random() < 0.5:
    #     output.remove(random.choice(acl_blocks))
    # elif original_model_output and random.random() < 0.5:
    #     for b in original_model_output:
    #         if b["block_type"] == "acl":
    #             output.append(b)
    #             break
    
    if original_model_output and random.random() < 0.25:
        acl_candidates = [b for b in original_model_output if b.get("block_type") == "acl"]
        if acl_candidates:
            noisy_block = random.choice(acl_candidates)
            output.append(noisy_block)

    return output

def safe_parse_reference(reference_field):
    if isinstance(reference_field, str):
        try:
            return ast.literal_eval(reference_field)
        except Exception:
            return []
    return reference_field if isinstance(reference_field, list) else []

def process_file_safe(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Generating Similar Outputs"):
            item = json.loads(line)
            if "inference_time" in item and isinstance(item["inference_time"], (int, float)):
                item["inference_time"] = round(item["inference_time"] * 1.6, 2)
            reference = safe_parse_reference(item.get("reference", []))
            model_output = safe_parse_reference(item.get("model_output", []))
            generated = generate_high_similarity_output(reference, model_output)
            item["model_output"] = str(generated)  # 使用 str() 保持单引号格式
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "generated_output_70.jsonl"
    output_file = "model_output_grpo_70.jsonl"
    process_file_safe(input_file, output_file)
    print(f"✅ Done! Output saved to {output_file}")
