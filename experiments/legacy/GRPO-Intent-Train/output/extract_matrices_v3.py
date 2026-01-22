import json
import numpy as np
import pandas as pd
import re
import os
import networkx as nx

def parse_topology(topology_str, device_list):
    matrix_size = len(device_list)
    name_to_idx = {name: idx for idx, name in enumerate(device_list)}
    adj_matrix = np.zeros((matrix_size, matrix_size))
    ip_to_node = {}

    for line in topology_str.strip().split("\n"):
        # 支持两种格式：带接口信息和不带接口信息
        match = re.match(r"(.+?) \((?:[\w/]+ )?([\d.]+)\) ↔ (.+?) \((?:[\w/]+ )?([\d.]+)\)(?: \[(\d+\.?\d*) Mbps\])?", line.strip())
        if match:
            node_a, ip_a, node_b, ip_b, load = match.groups()
            idx_a = name_to_idx[node_a.strip()]
            idx_b = name_to_idx[node_b.strip()]
            weight = float(load) if load else 0.0
            adj_matrix[idx_a][idx_b] = weight
            adj_matrix[idx_b][idx_a] = weight
            ip_to_node[ip_a] = node_a.strip()
            ip_to_node[ip_b] = node_b.strip()
    return adj_matrix, name_to_idx, ip_to_node

def extract_sla_intent(user_input, node_count=None):
    if node_count is not None:
        if node_count < 25:
            multiplier = 1.0
        elif 25 <= node_count <= 50:
            multiplier = 1.0
        elif 50 <= node_count <= 90:
            multiplier = 1.0
        else:  # node_count > 90
            multiplier = 0.55
    else:
        multiplier = 1.0  # 默认倍数

    # 首先尝试匹配Mbps格式（1.jsonl）
    match = re.search(r"Deploy a service flow from (.+?) to (.+?), with SLA guaranteeing a minimum bandwidth of (\d+) Mbps", user_input)
    if match:
        src, dst, sla = match.groups()
        return src.strip(), dst.strip(), float(sla) * multiplier
    
    # 然后尝试匹配Gbps格式（generated_output.jsonl）
    match = re.search(r"Deploy a service flow from (.+?) to (.+?), with SLA guaranteeing a minimum bandwidth of (\d+) Gbps", user_input)
    if match:
        src, dst, sla = match.groups()
        return src.strip(), dst.strip(), float(sla) * multiplier
    
    return None, None, None

def check_sla_satisfaction(adj_matrix, name_to_idx, src, dst, sla, k=1, max_capacity=None, node_count=None):
    if src not in name_to_idx or dst not in name_to_idx:
        return False, []

    # 如果没有指定max_capacity，根据node_count确定
    if max_capacity is None and node_count is not None:
        if node_count < 25:
            max_capacity = 300
        elif 25 <= node_count <= 50:
            max_capacity = 300
        elif 50 <= node_count <= 90:
            max_capacity = 300
        else:  # node_count > 90
            max_capacity = 500
    elif max_capacity is None:
        max_capacity = 300  # 默认值

    idx_to_name = {v: k for k, v in name_to_idx.items()}
    G = nx.Graph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=1, capacity=adj_matrix[i][j])

    try:
        paths = list(nx.shortest_simple_paths(G, name_to_idx[src], name_to_idx[dst], weight="weight"))
        satisfied = False
        checked_paths = []
        for path in paths[:k]:
            min_bw = min(G[u][v]["capacity"] for u, v in zip(path[:-1], path[1:]))
            max_bw = max(G[u][v]["capacity"] for u, v in zip(path[:-1], path[1:]))
            checked_paths.append(([idx_to_name[i] for i in path], min_bw, max_bw))
            if min_bw >= sla and max_bw <= max_capacity:
                satisfied = True
        return satisfied, checked_paths
    except:
        return False, []


def parse_device_configurations(text):
    acl_rules = []
    slb_rules = []
    device_blocks = re.findall(r'DEVICE: (.*?)\n(\[.*?)(?=(\nDEVICE:|\Z))', text, re.DOTALL)
    for device, block, _ in device_blocks:
        if '[ACL]' in block:
            # 支持两种格式的接口信息
            acl_entries = re.findall(
                r'ACL_\[(.*?),(.*?)\]:\s+- Permit ([\d.]+) -> ([\d.]+)\s+- Rate Limit: (\d+) Mbps\s+- Applied On: (\S+)',
                block)
            for src_node, dst_node, ip_src, ip_dst, rate, iface in acl_entries:
                acl_rules.append({
                    "device": device.strip(),
                    "src_node": src_node.strip(),
                    "dst_node": dst_node.strip(),
                    "src_ip": ip_src,
                    "dst_ip": ip_dst,
                    "rate_limit": float(rate),
                    "interface": iface.strip()
                })
        if '[SLB]' in block:
            # 匹配SLB配置，支持不同的格式
            slb_entries = re.findall(
                r'Serverfarm SF_\[(.*?),(.*?)\]:\s+- Real ([\d.]+) with weight (\d+)\s+- Real ([\d.]+) with weight (\d+)\s+Virtual-Server VS_\[.*?\]:\s+- IP: ([\d.]+)',
                block, re.DOTALL)
            for src_node, dst_node, ip1, w1, ip2, w2, vip in slb_entries:
                slb_rules.append({
                    "device": device.strip(),
                    "src_node": src_node.strip(),
                    "dst_node": dst_node.strip(),
                    "virtual_ip": vip,
                    "real_servers": [
                        {"ip": ip1, "weight": int(w1)},
                        {"ip": ip2, "weight": int(w2)}
                    ]
                })
    return acl_rules, slb_rules

# def parse_model_output(model_output_text):
#     try:
#         return json.loads(model_output_text.replace("'", '"'))
#     except json.JSONDecodeError:
#         return []

import ast

def parse_model_output(model_output_text):
    try:
        return ast.literal_eval(model_output_text)
    except Exception as e:
        print("解析失败：", e)
        return []


def standardize_model_middleboxes(model_items):
    acl_rules = []
    slb_rules = []
    for item in model_items:
        if item['block_type'] == 'acl' and item['acl_rules']:
            for rule in item['acl_rules']['rules']:
                ip_src, ip_dst = rule['permit'].split(" -> ")
                acl_rules.append({
                    "device": item['device'],
                    "src_ip": ip_src.strip(),
                    "dst_ip": ip_dst.strip(),
                    "rate_limit": float(rule['rate_limit'].split()[0]),
                    "interface": rule['interface']
                })
        elif item['block_type'] == 'slb' and item['slb_config']:
            slb_rules.append({
                "device": item['device'],
                "virtual_ip": item['slb_config']['virtual_ip'].split('/')[0],
                "real_servers": [
                    {"ip": rs['ip'], "weight": int(rs['weight'])}
                    for rs in item['slb_config']['real_servers']
                ]
            })
    return acl_rules, slb_rules

def compare_rules(model_acls, model_slbs, existing_acls, existing_slbs):
    def rule_key(rule):
        return (rule.get('device'), rule.get('src_ip'), rule.get('dst_ip'), rule.get('rate_limit'))
    def slb_key(rule):
        return (rule.get('device'), rule.get('virtual_ip'),
                tuple(sorted((rs['ip'], rs['weight']) for rs in rule['real_servers'])))
    existing_acl_keys = set(rule_key(r) for r in existing_acls)
    new_model_acls = [r for r in model_acls if rule_key(r) not in existing_acl_keys]
    existing_slb_keys = set(slb_key(r) for r in existing_slbs)
    new_model_slbs = [r for r in model_slbs if slb_key(r) not in existing_slb_keys]

    return new_model_acls, new_model_slbs


def simulate_acl_effect(adj_matrix, name_to_idx, acl_rules, ip_to_node, node_count=None):
    # 根据节点数量确定ACL倍率
    if node_count is not None:
        if node_count < 25:
            acl_multiplier = 1.0
        elif 25 <= node_count <= 50:
            acl_multiplier = 1.2
        elif 50 <= node_count <= 90:
            acl_multiplier = 1.2
        else:  # node_count > 90
            acl_multiplier = 1.2
    else:
        acl_multiplier = 1.0  # 默认倍率

    G = nx.Graph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=1)
    for rule in acl_rules:
        src_ip, dst_ip = rule["src_ip"], rule["dst_ip"]
        middlebox = rule["device"]
        rate = rule["rate_limit"] * acl_multiplier  # 应用倍率
        if src_ip not in ip_to_node or dst_ip not in ip_to_node or middlebox not in name_to_idx:
            continue
        src_node = ip_to_node[src_ip]
        dst_node = ip_to_node[dst_ip]
        try:
            paths = list(nx.all_shortest_paths(G, name_to_idx[src_node], name_to_idx[dst_node]))
            for path in paths:
                if name_to_idx[middlebox] in path:
                    for u, v in zip(path[:-1], path[1:]):
                        adj_matrix[u][v] += rate
                        adj_matrix[v][u] += rate
                    break
        except:
            continue

def simulate_slb_effect(adj_matrix, name_to_idx, slb_rules, ip_to_node):
    G = nx.Graph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=1)
    for rule in slb_rules:
        total_bw = 30.0
        src = rule.get("src_node")
        dst = rule.get("dst_node")
        middlebox = rule.get("device")
        if not all(n in name_to_idx for n in [src, dst, middlebox]):
            continue
        try:
            path1 = nx.shortest_path(G, name_to_idx[src], name_to_idx[middlebox])
            for u, v in zip(path1[:-1], path1[1:]):
                adj_matrix[u][v] += total_bw
                adj_matrix[v][u] += total_bw
            for server in rule["real_servers"]:
                real_ip = server["ip"]
                weight = server["weight"] / 100.0
                if real_ip not in ip_to_node:
                    continue
                real_node = ip_to_node[real_ip]
                try:
                    path2 = nx.shortest_path(G, name_to_idx[middlebox], name_to_idx[real_node])
                    for u, v in zip(path2[:-1], path2[1:]):
                        adj_matrix[u][v] += total_bw * weight
                        adj_matrix[v][u] += total_bw * weight
                except:
                    continue
        except:
            continue

def extract_and_save_matrices(jsonl_path, output_dir):
    with open(jsonl_path, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()]
    
    print(f"总样本数: {len(samples)}")
    
    os.makedirs(output_dir, exist_ok=True)
    sla_results = []
    sla_results_modified = []
    
    # 统计跳过的样本数
    parse_failed_count = 0
    sla_failed_count = 0
    processed_count = 0

    for idx, sample in enumerate(samples):
        user_input = sample["user_input"]
        model_output_text = sample.get("model_output", "")
        config_match = re.search(r"\[Device Configurations\]\n(.*?)\n\#", user_input, re.DOTALL)
        device_match = re.search(r"\[Device List\]\n(.+?)\n\n", user_input, re.DOTALL)
        topology_match = re.search(r"\[Topology Info\]\n(.+?)\n\#\#", user_input, re.DOTALL)
        if not all([config_match, device_match, topology_match]):
            parse_failed_count += 1
            print(f"样本 {idx} 解析失败: config={bool(config_match)}, device={bool(device_match)}, topology={bool(topology_match)}")
            continue
        device_names = [d.strip() for d in device_match.group(1).split(',')]
        topology_str = topology_match.group(1)
        config_text = config_match.group(1)

        adj_matrix, name_to_idx, ip_to_node = parse_topology(topology_str, device_names)
        
        # 保存原始邻接矩阵
        df_original = pd.DataFrame(adj_matrix, index=device_names, columns=device_names).round(2)
        df_original.to_csv(os.path.join(output_dir, f"adj_matrix_original_{idx}.csv"))

        # 传入节点数量来计算SLA
        node_count = len(device_names)
        src, dst, sla = extract_sla_intent(user_input, node_count)
        if not src or not dst:
            sla_failed_count += 1
            print(f"样本 {idx} SLA解析失败: src={src}, dst={dst}")
            continue

        processed_count += 1

        satisfied, paths = check_sla_satisfaction(adj_matrix.copy(), name_to_idx, src, dst, sla, node_count=node_count)
        path_bws = [round(p[1], 2) for p in paths]
        sla_results.append({
            "sample_id": idx, "source": src, "destination": dst, "sla": sla,
            "satisfied": satisfied, "min_bw_top_paths": path_bws
        })

        existing_acls, existing_slbs = parse_device_configurations(config_text)
        model_items = parse_model_output(model_output_text)
        model_acls, model_slbs = standardize_model_middleboxes(model_items)
        new_acls, new_slbs = compare_rules(model_acls, model_slbs, existing_acls, existing_slbs)

        adj_matrix_modified = adj_matrix.copy()
        simulate_acl_effect(adj_matrix_modified, name_to_idx, new_acls, ip_to_node, node_count)
        simulate_slb_effect(adj_matrix_modified, name_to_idx, new_slbs, ip_to_node)
        
        # 保存增加中间件信息后的邻接矩阵
        df_modified = pd.DataFrame(adj_matrix_modified, index=device_names, columns=device_names).round(2)
        df_modified.to_csv(os.path.join(output_dir, f"adj_matrix_modified_{idx}.csv"))

        satisfied_m, paths_m = check_sla_satisfaction(adj_matrix_modified, name_to_idx, src, dst, sla, node_count=node_count)
        path_min_bws_m = [round(p[1], 2) for p in paths_m]
        path_max_bws_m = [round(p[2], 2) for p in paths_m]
        
        # 找出未满足SLA的链路
        unsatisfied_links = ""
        if paths_m and not satisfied_m:
            # 只检查最短路径（k=1）
            shortest_path_nodes = paths_m[0][0]
            shortest_path_min_bw = paths_m[0][1]
            if shortest_path_min_bw < sla:
                # 找出带宽小于SLA的链路
                links_below_sla = []
                for i in range(len(shortest_path_nodes) - 1):
                    node_a = shortest_path_nodes[i]
                    node_b = shortest_path_nodes[i + 1]
                    idx_a = name_to_idx[node_a]
                    idx_b = name_to_idx[node_b]
                    link_bw = adj_matrix_modified[idx_a][idx_b]
                    if link_bw < sla:
                        links_below_sla.append(f"{node_a}-{node_b}({link_bw:.1f})")
                unsatisfied_links = "; ".join(links_below_sla)
        
        sla_results_modified.append({
            "sample_id": idx, "source": src, "destination": dst, "sla": sla,
            "satisfied": satisfied_m, "min_bw_top_paths": path_min_bws_m, "max_bw_top_paths": path_max_bws_m,
            "unsatisfied_links": unsatisfied_links
        })

    # 构造 DataFrame
    df_orig = pd.DataFrame(sla_results)
    df_mod = pd.DataFrame(sla_results_modified)

    # 过滤有效样例
    valid_orig = df_orig[df_orig["min_bw_top_paths"].astype(str) != "[]"]
    valid_mod = df_mod[df_mod["min_bw_top_paths"].astype(str) != "[]"]
    # print(f"原始样例数量: {len(df_orig)}, 有效样例数量: {len(valid_orig)}")

    # 计算满足率
    rate_orig = valid_orig["satisfied"].mean()
    rate_mod = valid_mod["satisfied"].mean()

    # 构造统计行
    summary_orig = pd.DataFrame([{
        "sample_id": "summary",
        "source": "",
        "destination": "",
        "sla": "",
        "satisfied": f"{rate_orig:.2%}",
        "min_bw_top_paths": "",
        "max_bw_top_paths": ""
    }])
    summary_mod = pd.DataFrame([{
        "sample_id": "summary",
        "source": "",
        "destination": "",
        "sla": "",
        "satisfied": f"{rate_mod:.2%}",
        "min_bw_top_paths": "",
        "max_bw_top_paths": "",
        "unsatisfied_links": ""
    }])

    # 拼接并保存
    final_orig = pd.concat([summary_orig, df_orig], ignore_index=True)
    final_mod = pd.concat([summary_mod, df_mod], ignore_index=True)
    # final_orig.to_csv("100_sla_check_results.csv", index=False)
    final_mod.to_csv("grpo_70_sla_check_results_modified_v.csv", index=False)

    print(f"解析失败样本数: {parse_failed_count}")
    print(f"SLA提取失败样本数: {sla_failed_count}")
    print(f"成功处理样本数: {processed_count}")
    print(f"原始有效样例数量: {len(valid_orig)}")
    print(f"修改后有效样例数量: {len(valid_mod)}")
    print("所有结果已保存")

if __name__ == "__main__":
    extract_and_save_matrices("model_output_grpo_70.jsonl", "adj_matrices")
