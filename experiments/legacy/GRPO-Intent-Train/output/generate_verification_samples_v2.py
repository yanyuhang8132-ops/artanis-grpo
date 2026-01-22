import json
import numpy as np
import pandas as pd
import torch
import networkx as nx
from extract_matrices_v2 import parse_topology, parse_device_configurations, parse_model_output, standardize_model_middleboxes, compare_rules, simulate_acl_effect, simulate_slb_effect
from utils.Equivalence_Class import generate_equivalence_classes

import re

def parse_user_input_fields(user_input):
    device_match = re.search(r"## \[Device List\]\n(.+?)\n\n", user_input, re.DOTALL)
    topology_match = re.search(r"## \[Topology Info\]\n(.+?)\n##", user_input, re.DOTALL)
    config_match = re.search(r"## \[Device Configurations\]\n(.+?)\n##", user_input, re.DOTALL)

    devices = [d.strip() for d in device_match.group(1).split(",")] if device_match else []
    topology = topology_match.group(1).strip() if topology_match else ""
    configs = config_match.group(1).strip() if config_match else ""

    return devices, topology, configs



def generate_traffic_flows(G, name_to_idx, acl_map, slb_map, num_flows=10, max_sla=100):
    print("acl_map:", acl_map)
    print("slb_map:", slb_map)
    nodes = list(name_to_idx.keys())
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    traffic_flows = []
    traffic_pairs = []

    for _ in range(num_flows):
        src, dst = np.random.choice(nodes, 2, replace=False)
        src_id = name_to_idx[src]
        dst_id = name_to_idx[dst]
        sla = np.random.randint(20, max_sla)

        lb_vector = [0.0] * 5  # 固定为5个元素
        acl_rules = []

        try:
            path = nx.shortest_path(G, source=src_id, target=dst_id)
        except:
            continue

        for node_id in path:
            dev = idx_to_name[node_id]
            if dev in slb_map:
                lb_index = slb_map[dev]
                if lb_index < 5:  # 确保不超出lb_vector的范围
                    lb_vector[lb_index] = 1.0  # 假设默认经过该LB分流
            if dev in acl_map:
                rate = acl_map[dev]
                acl_rules.append([node_id, rate])

        traffic_flows.append([src_id, dst_id, sla, lb_vector, acl_rules])
        traffic_pairs.append([src_id, dst_id, path, sla])  # 添加路径和带宽信息

    return traffic_flows, traffic_pairs


def extract_flow_configurations_v2(existing_acls, existing_slbs, new_acls, new_slbs):
    """从现有和新增的中间件配置中提取流量配置信息（修正版）"""
    all_acls = existing_acls + new_acls
    all_slbs = existing_slbs + new_slbs
    
    # 提取所有流量的起终点信息
    flow_endpoints = set()
    
    # 从ACL规则中提取流量端点
    for acl in all_acls:
        # 处理现有配置格式（来自parse_device_configurations）
        if 'src_ip' in acl and 'dst_ip' in acl:
            src_ip = acl['src_ip']
            dst_ip = acl['dst_ip']
            flow_endpoints.add((src_ip, dst_ip))
        # 处理新增配置格式（来自standardize_model_middleboxes）
        elif 'rules' in acl and acl['rules']:
            for rule in acl['rules']:
                if 'permit' in rule:
                    permit_str = rule['permit']
                    # 解析 "10.0.0.35 -> 10.0.0.60" 格式
                    if ' -> ' in permit_str:
                        src_ip, dst_ip = permit_str.split(' -> ')
                        flow_endpoints.add((src_ip.strip(), dst_ip.strip()))
    
    # 从SLB规则中提取流量端点（根据virtual_ip作为目标）
    for slb in all_slbs:
        # 处理现有配置格式（来自parse_device_configurations）
        if 'virtual_ip' in slb and 'real_servers' in slb:
            virtual_ip = slb['virtual_ip']
            for server in slb['real_servers']:
                src_ip = server['ip']
                flow_endpoints.add((src_ip, virtual_ip))
        # 处理新增配置格式（来自standardize_model_middleboxes）
        elif 'slb_config' in slb and slb['slb_config']:
            config = slb['slb_config']
            if 'virtual_ip' in config and 'real_servers' in config:
                virtual_ip = config['virtual_ip'].split(' ')[0]  # 去掉子网掩码
                for server in config['real_servers']:
                    src_ip = server['ip']
                    flow_endpoints.add((src_ip, virtual_ip))
    
    return list(flow_endpoints), all_acls, all_slbs

def generate_sample(entry, sample_id):
    from ast import literal_eval

    # 提取数据
    user_input = entry["user_input"]
    model_output = literal_eval(entry["model_output"])
    device_names, topology_str, config_text = parse_user_input_fields(user_input)


    # 建图
    adj_matrix, name_to_idx, ip_to_node = parse_topology(topology_str, device_names)
    existing_acls, existing_slbs = parse_device_configurations(config_text)
    model_acls_raw, model_slbs_raw = standardize_model_middleboxes(model_output)
    new_acls, new_slbs = compare_rules(model_acls_raw, model_slbs_raw, existing_acls, existing_slbs)
    
    # 统一合并中间件配置，避免多次赋值
    all_acls = existing_acls + new_acls
    all_slbs = existing_slbs + new_slbs
    print("=== 详细调试信息 ===")
    print("existing_acls:", existing_acls)
    print("new_acls:", new_acls)
    print("existing_slbs:", existing_slbs)
    print("new_slbs:", new_slbs)
    
    # 模拟中间件效果
    simulate_acl_effect(adj_matrix, name_to_idx, new_acls, ip_to_node)  # 只模拟新增的
    simulate_slb_effect(adj_matrix, name_to_idx, new_slbs, ip_to_node)  # 只模拟新增的

    # 构建图
    G = nx.DiGraph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=1)

    N = len(device_names)
    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    in_degree = np.zeros(N)
    out_degree = np.zeros(N)
    betw = np.zeros(N)
    for i, name in enumerate(device_names):
        in_degree[i] = out_degree[i] = degrees.get(i, 0)
        betw[i] = betweenness.get(i, 0)
    routes_df = pd.read_csv("routes.csv")
    # 构造等价类
    forwarding_graphs, ec_networks = generate_equivalence_classes(G, routes_df)
    ec_counts = [len(nets) for nets in ec_networks]

    # 修正版：提取流量配置信息
    flow_endpoints, _, _ = extract_flow_configurations_v2(existing_acls, existing_slbs, new_acls, new_slbs)

    # 构建ACL映射：设备名 -> 限速值
    acl_map = {}
    print("\n=== ACL 映射构建过程 ===")
    for idx, rule in enumerate(all_acls):
        device = rule.get('device')
        rate_limit = rule.get('rate_limit')
        print(f"ACL规则 {idx}: device={device}, rate_limit={rate_limit}, 完整规则={rule}")
        if device and rate_limit is not None:  # 修复：使用 is not None 而不是布尔检查
            acl_map[device] = rate_limit
            print(f"  -> 添加到 acl_map: {device} = {rate_limit}")
        else:
            print(f"  -> 跳过: device={device}, rate_limit={rate_limit}")
    
    # 构建SLB映射：设备名 -> 索引
    slb_map = {}
    print("\n=== SLB 映射构建过程 ===")
    for idx, rule in enumerate(all_slbs):
        device = rule.get('device')
        print(f"SLB规则 {idx}: device={device}, 完整规则={rule}")
        if device:
            slb_map[device] = idx
            print(f"  -> 添加到 slb_map: {device} = {idx}")
        else:
            print(f"  -> 跳过: device={device}")
    
    print("\n=== 最终映射结果 ===")
    print("acl_map:", acl_map)
    print("slb_map:", slb_map)
    print("acl_map 设备数量:", len(acl_map))
    print("slb_map 设备数量:", len(slb_map))
    print("all_acls 规则数量:", len(all_acls))
    print("all_slbs 规则数量:", len(all_slbs))

    # 构造流量
    traffic_flows, traffic_pairs = generate_traffic_flows(G, name_to_idx, acl_map, slb_map, num_flows=10)

    # 构造 mask（全部为有效节点）
    mask = [1] * N

    # 拼装 sample
    sample = {
        "graph_features": {
            "adjacency": adj_matrix.tolist(),
            "in_degree": in_degree.tolist(),
            "out_degree": out_degree.tolist(),
            "betweenness": betw.tolist(),
            "ec_counts": ec_counts,
            "ec_networks": ec_networks
        },
        "non_graph_features": {
            "traffic_flows": traffic_flows
        },
        "mask": mask
    }
    return sample

def main():
    with open("inference_5_samples.jsonl", "r") as fin:
        lines = [json.loads(line) for line in fin]

    results = []
    for i, entry in enumerate(lines):
        sample = generate_sample(entry, i)
        results.append(sample)

    with open("verification_samples.jsonl", "w") as fout:
        for item in results:
            fout.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
