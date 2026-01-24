import os
import re
import json

def extract_name_ip_mapping(cfg_text):
    name2ip = {}
    ip_regex = r'neighbor (\d+\.\d+\.\d+\.\d+) description "To (\w+)"'
    for match in re.findall(ip_regex, cfg_text):
        ip, name = match
        name2ip[name] = ip
    return name2ip

def generate_slb_section(router_name, rules, name2ip):
    # 将name2ip的key全部转为小写，便于后续查找
    name2ip_lower = {k.lower(): v for k, v in name2ip.items()}
    slb_lines = [f"! SLB Configuration for {router_name} "]
    for flow_str, rule in rules.items():
        try:
            src, dst = json.loads(flow_str.replace('[','["').replace(',', '","').replace(']','"]'))
        except Exception:
            continue

        next_hops = rule["next_hops"]
        weights = rule["weights"]
        sf_name = f"SF_{src}_{dst}".replace("-", "_")
        vs_name = f"VSERVER_{src}_{dst}".replace("-", "_")

        slb_lines.append(f"ip slb serverfarm {sf_name}")
        for hop, weight in zip(next_hops, weights):
            real_ip = name2ip_lower.get(hop.lower(), f"0.0.0.0")
            slb_lines.append(f" real {real_ip}")
            slb_lines.append(f"  weight {int(weight * 100)}")

        prev_hop = rule.get("prev_hop", src)
        prev_hop_ip = name2ip_lower.get(prev_hop.lower(), "11.0.0.1")
        slb_lines.append(f"\nip slb virtual-server {vs_name}")
        slb_lines.append(f" address {prev_hop_ip} 255.255.255.255")
        slb_lines.append(f" serverfarm {sf_name}\n")
    return slb_lines

def process_configs(config_dir, policy_path, output_dir):
    with open(policy_path, "r") as f:
        policies = json.load(f)

    lb_rules = policies.get("lb_rules", {})
    # 构建小写key到原key的映射
    lb_rules_lower = {k.lower(): k for k in lb_rules}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(config_dir):
        if not fname.endswith(".cfg"):
            continue
        router = fname.replace(".cfg", "")
        router_lower = router.lower()
        with open(os.path.join(config_dir, fname), "r") as f:
            cfg_text = f.read()

        name2ip = extract_name_ip_mapping(cfg_text)
        lines = cfg_text.strip().splitlines()
        new_lines = lines + [""]

        if router_lower in lb_rules_lower:
            rule_key = lb_rules_lower[router_lower]
            new_lines += generate_slb_section(router, lb_rules[rule_key], name2ip)

        new_path = os.path.join(output_dir, fname.replace(".cfg", ".lb.cfg"))
        with open(new_path, "w") as f:
            f.write("\n".join(new_lines))

if __name__ == "__main__":
    config_dir = "Arnes_abs_order_1_72/configs"
    policy_path = "middlebox_policies.json"
    output_dir = "Arnes_abs_order_1_72/configs_lb"
    process_configs(config_dir, policy_path, output_dir)

