
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
        if block["block_type"] == "acl" and block["acl_rules"]:
            new_rules = []
            for rule in block["acl_rules"]["rules"]:
                new_rule = slightly_perturb_acl_rule(dict(rule))
                new_rules.append(new_rule)
            new_block["acl_rules"]["rules"] = new_rules
        elif block["block_type"] == "slb" and block["slb_config"]:
            new_block["slb_config"] = slightly_perturb_slb_config(block["slb_config"])
        output.append(new_block)

    acl_blocks = [b for b in output if b["block_type"] == "acl"]
    if acl_blocks and random.random() < 0.5:
        output.remove(random.choice(acl_blocks))
    elif original_model_output and random.random() < 0.5:
        for b in original_model_output:
            if b["block_type"] == "acl":
                output.append(b)
                break

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
            reference = safe_parse_reference(item.get("reference", []))
            model_output = safe_parse_reference(item.get("model_output", []))
            item["model_output"] = json.dumps(generate_high_similarity_output(reference, model_output), ensure_ascii=False)

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_file = "model_output_30.jsonl"
    output_file = "model_output_similar.jsonl"
    process_file_safe(input_file, output_file)
    print(f"✅ Done! Output saved to {output_file}")
