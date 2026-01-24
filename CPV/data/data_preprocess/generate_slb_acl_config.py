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

def generate_acl_section(router_name, rules, name2ip, flows_path, acl_base=100, cfg_text=None):
    # 读取flows_with_paths.json，建立(src, dst)到flow的映射
    with open(flows_path, "r") as f:
        flows = json.load(f)
    flow_map = {}
    for flow in flows:
        key = (flow["src"], flow["dst"])
        flow_map[key] = flow
    # 新增：解析interface与ip和description的映射
    iface_ip_map = {}
    iface_desc_map = {}
    if cfg_text:
        current_iface = None
        current_ip = None
        current_desc = None
        for line in cfg_text.splitlines():
            line = line.strip()
            if line.startswith('interface '):
                current_iface = line.split()[1]
                current_ip = None
                current_desc = None
            elif line.startswith('ip address '):
                m = re.match(r'ip address ([0-9.]+) ', line)
                if m and current_iface:
                    iface_ip_map[m.group(1)] = current_iface
            elif line.startswith('description '):
                m = re.match(r'description "?To ([^"\n]+)"?', line)
                if m and current_iface:
                    iface_desc_map[m.group(1).lower()] = current_iface
    acl_lines = [f"! ACL Configuration for {router_name}"]
    acl_id = acl_base
    for flow_str, rule in rules.items():
        try:
            src, dst = json.loads(flow_str.replace('[','[\"').replace(',', '\",\"').replace(']','\"]'))
        except Exception:
            continue
        rate_limit = rule["rate_limit"] * 1000000  # Mbps to bps
        # 查找flows_with_paths.json中的接口ip
        flow_info = flow_map.get((src, dst))
        src_ip = "11.0.0.1"
        dst_ip = "100.0.0.1"
        iface = "Fa0/0"
        if flow_info:
            if f"src_interface_ip_{router_name.lower()}" in flow_info and f"dst_interface_ip_{router_name.lower()}" in flow_info:
                src_ip = flow_info[f"src_interface_ip_{router_name.lower()}"]
                dst_ip = flow_info[f"dst_interface_ip_{router_name.lower()}"]
            else:
                src_ip = flow_info.get("src_interface_ip", src_ip)
                dst_ip = flow_info.get("dst_interface_ip", dst_ip)
            # 查找acl节点在path中的下一个节点
            path = flow_info.get("path", [])
            if router_name in path:
                idx = path.index(router_name)
                if idx + 1 < len(path):
                    next_hop = path[idx + 1].lower()
                    iface = iface_desc_map.get(next_hop, iface)
        acl_name = f"ACL{acl_id}"
        class_name = f"SLA{acl_id}"
        acl_lines.append(f"ip access-list extended {acl_name}")
        acl_lines.append(f" permit ip host {src_ip} host {dst_ip}")
        acl_lines.append(f"class-map match-any {class_name}")
        acl_lines.append(f" match access-group name {acl_name}")
        acl_lines.append(f"policy-map ACL_LIMIT_{acl_id}")
        acl_lines.append(f" class {class_name}")
        acl_lines.append(f"  police {rate_limit}")
        acl_lines.append(f"interface {iface}")
        acl_lines.append(f" service-policy input ACL_LIMIT_{acl_id}\n")
        acl_id += 1
    return acl_lines

def process_configs(config_dir, policy_path, output_dir):
    with open(policy_path, "r") as f:
        policies = json.load(f)

    lb_rules = policies.get("lb_rules", {})
    acl_rules = policies.get("acl_rules", {})

    lb_rules_lower = {k.lower(): k for k in lb_rules}
    acl_rules_lower = {k.lower(): k for k in acl_rules}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flows_path = os.path.join(os.path.dirname(policy_path), "flows_with_paths.json")

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

        if router_lower in acl_rules_lower:
            rule_key = acl_rules_lower[router_lower]
            new_lines += generate_acl_section(router, acl_rules[rule_key], name2ip, flows_path, cfg_text=cfg_text)

        new_path = os.path.join(output_dir, fname.replace(".cfg", ".lb.cfg"))
        with open(new_path, "w") as f:
            f.write("\n".join(new_lines))

if __name__ == "__main__":
    config_dir = "Colt_abs_order_1_610/configs"
    policy_path = "Colt_middlebox_policies.json"
    output_dir = "Colt_abs_order_1_610/configs_lb"
    process_configs(config_dir, policy_path, output_dir)
