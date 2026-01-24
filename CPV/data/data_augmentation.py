import csv
import random
import numpy as np
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import copy
from link_load_matrix import calculate_link_load_matrix

# 1. 动态故障参数计算
def calculate_failure_limits(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    return int(0.5 * num_nodes), int(0.2 * num_edges)

def detect_forwarding_loops(G, traffic_pairs):
    """
    基于业务流的实际转发路径构建子图，再判断是否形成 loop。
    返回每个节点是否出现在任一流量路径环中的标签。
    """
    F = nx.DiGraph()
    for src, dst, path in traffic_pairs:
        for i in range(len(path) - 1):
            F.add_edge(path[i], path[i+1])

    loops = set()
    for cycle in nx.simple_cycles(F):
        loops.update(cycle)

    return [1 if node in loops else 0 for node in G.nodes()]


def calculate_load_balancing_optimized(G, traffic_pairs, load_balancing_threshold=0.1, max_paths=1, sample_ratio=0.1):
    """优化后的负载均衡标签生成函数"""
    nodes = list(G.nodes())
    N = len(nodes)
    load_balancing_matrix = np.zeros((N, N), dtype=int)
    
    # 计算每条链路的负载
    link_utilization = defaultdict(int)
    for src, dst, path, sla_bw in traffic_pairs:
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_utilization[link] += 1
    
    # 采样节点对
    sampled_pairs = []
    for i in range(N):
        for j in range(N):
            if i != j and random.random() < sample_ratio:  # 随机采样 10% 的节点对
                sampled_pairs.append((i, j))
    
    # 计算采样节点对的负载均衡标签
    for i, j in sampled_pairs:
        src, dst = nodes[i], nodes[j]
        
        # 获取从 src 到 dst 的最短路径（最多 max_paths 条）
        try:
            paths = list(nx.shortest_simple_paths(G, src, dst))[:max_paths]
        except nx.NetworkXNoPath:
            paths = []
        
        if not paths:
            load_balancing_matrix[i][j] = 0  # 如果没有路径，负载均衡标签为 0
            continue
        
        # 计算每条路径的链路负载
        path_loads = []
        for path in paths:
            path_load = 0
            for k in range(len(path) - 1):
                link = (path[k], path[k + 1])
                path_load += link_utilization.get(link, 0)
            path_loads.append(path_load)
        
        # 计算方差
        var_load = np.var(path_loads) if path_loads else 0
        load_balancing_matrix[i][j] = 1 if var_load <= load_balancing_threshold else 0
    
    return load_balancing_matrix.tolist()

def calculate_interface_utilization(G, traffic_pairs, interface_utilization_threshold=0.9):
    """计算每个节点的接口利用率标签，形状为 [N_max]"""
    nodes = list(G.nodes())
    N = len(nodes)
    node_utilization = defaultdict(int)  # 记录每个节点的最大接口利用率

    # 统计每条链路的流量
    link_utilization = defaultdict(int)
    for src, dst, path, sla_bw in traffic_pairs:
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            link_utilization[link] += 1

    # 计算每个节点的接口利用率
    for node in nodes:
        # 遍历节点的所有出接口
        for neighbor in G.successors(node):
            link = (node, neighbor)
            util = link_utilization.get(link, 0)
            if util > node_utilization[node]:
                node_utilization[node] = util

    # 生成标签：1 表示至少有一个接口利用率超过阈值，否则为 0
    high_util_labels = [
        1 if node_utilization.get(node, 0) >= interface_utilization_threshold else 0
        for node in nodes
    ]

    return high_util_labels

def detect_routing_blackholes(
    forwarding_graphs,
    nodes,
    unreachable_ratio_threshold=0.6,
    min_unreachable_count=2
):
    """判定某目的节点是否为blackhole，必须满足两个条件：
    1）源节点中超过一定比例不可达；
    2）至少有一定数量的源不可达（排除极小子图）"""
    blackhole_nodes = set()
    for fg in forwarding_graphs.values():
        fg_nodes = list(fg.nodes())
        for dst in fg_nodes:
            unreachable_count = 0
            total_sources = 0
            for src in fg_nodes:
                if src == dst:
                    continue
                total_sources += 1
                if not nx.has_path(fg, src, dst):
                    unreachable_count += 1
            ratio = unreachable_count / total_sources if total_sources > 0 else 0
            if unreachable_count >= min_unreachable_count and ratio >= unreachable_ratio_threshold:
                blackhole_nodes.add(dst)
    return [1 if node in blackhole_nodes else 0 for node in nodes]



def inject_fake_loops(ec_features, forwarding_graphs, loop_ratio=0.1):
    import uuid
    ec_features = copy.deepcopy(ec_features)
    forwarding_graphs = copy.deepcopy(forwarding_graphs)
    all_ec_keys = list(forwarding_graphs.keys())
    num_loops = max(1, int(loop_ratio * len(all_ec_keys)))
    chosen_ecs = random.sample(all_ec_keys, num_loops)
    for ec in chosen_ecs:
        fg = forwarding_graphs[ec].copy()  # 复制子图，避免影响其它场景
        nodes = list(fg.nodes())
        if len(nodes) > 1:
            fake_prefix = f"fake_{uuid.uuid4()}"
            fg.add_edge(nodes[-1], nodes[0])
            for n in nodes:
                ec_features.setdefault(n, []).append(fake_prefix)
            forwarding_graphs[ec] = fg  # 回写
    return ec_features, forwarding_graphs

def generate_labels(G, traffic_pairs, interface_utilization_threshold=0.8, load_balancing_threshold=0.1, forwarding_graphs_path=None, forwarding_graphs_override=None,
                    flows_path=None, policy_path=None, adj_path=None):
    sla_labels = []
    if flows_path and policy_path:
        with open(flows_path, "r") as f:
            flow_data = json.load(f)
        with open(policy_path, "r") as f:
            policy_data = json.load(f)

        acl_rules = policy_data.get("acl_rules", {})

        # 构造 ACL 规则索引表（node → {(src, dst): rate_limit}）
        acl_dict = {}
        for node, rules in acl_rules.items():
            node = node.lower()
            acl_dict[node] = {}
            for flow_key, rule in rules.items():
                src, dst = flow_key.strip("[]").split(",")
                acl_dict[node][(src.lower(), dst.lower())] = rule["rate_limit"]

        # 检查每条 flow 是否被限速
        for flow in flow_data:
            src = flow["src"].lower()
            dst = flow["dst"].lower()
            sla_bw = flow["sla_bw"]
            path = [p.lower() for p in flow["path"]]

            violated = False
            for node in path:
                rate = acl_dict.get(node, {}).get((src, dst), float("inf"))
                if rate < sla_bw:
                    violated = True
                    break
            sla_labels.append(0 if violated else 1)
    
    nodes = list(G.nodes())
    # 1. Router Isolation
    isolated_labels = [int(G.in_degree(node) + G.out_degree(node) == 0) for node in nodes]

    # 2. Router Loop（基于EC传播图）
    loop_nodes = set()
    forwarding_graphs = None
    if forwarding_graphs_override is not None:
        forwarding_graphs = forwarding_graphs_override
    elif forwarding_graphs_path is not None:
        import pickle
        with open(forwarding_graphs_path, "rb") as f:
            forwarding_graphs = pickle.load(f)
    if forwarding_graphs is not None:
        for fg in forwarding_graphs.values():
            for cycle in nx.simple_cycles(fg):
                loop_nodes.update(cycle)
    loop_labels = [1 if node in loop_nodes else 0 for node in nodes]
      # 2.5 Routing Black Hole（控制平面）
    blackhole_labels = [0] * len(nodes)
    if forwarding_graphs is not None:
        # blackhole_labels = detect_routing_blackholes(forwarding_graphs, nodes)
        blackhole_labels = detect_routing_blackholes(
            forwarding_graphs, nodes,
            unreachable_ratio_threshold=0.8,
            min_unreachable_count=1
        )

    # 3. Router Reachability
    reachability_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            reachability_matrix[i][j] = 1 if nx.has_path(G, src, dst) else 0

    # 4. Interface High Utilization (优化后的函数)
    high_util_labels = calculate_interface_utilization(G, traffic_pairs, interface_utilization_threshold)

    # 5. Load Balancing (优化后的函数)
    load_balancing_labels = calculate_load_balancing_optimized(G, traffic_pairs, load_balancing_threshold)

    # ==== 链路过载标签生成 ====
    link_overload_matrix = None
    if flows_path and policy_path and adj_path:
        load_matrix, node_names = calculate_link_load_matrix(flows_path, policy_path, adj_path)
        overload_threshold = 150  # 可根据实际情况调整
        N = len(node_names)
        link_overload_matrix = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                if load_matrix[i, j] != -1 and load_matrix[i, j] > overload_threshold:
                    link_overload_matrix[i, j] = 1
    else:
        # 若没有数据，返回None或全0矩阵
        N = len(G.nodes())
        link_overload_matrix = np.zeros((N, N), dtype=int)

    return {
        "isolated": isolated_labels,
        "loop": loop_labels,
        "blackhole": blackhole_labels,
        "reachability": reachability_matrix.tolist(),
        "high_utilization": high_util_labels,
        "load_balancing": load_balancing_labels,
        "link_overload_pairs": link_overload_matrix.tolist(),
        "sla_labels": sla_labels
    }

# 2. 故障注入与特征更新
def generate_failure_scenario(G, interface_encoding, traffic_pairs, scenario_id, ec_features, policy_data, forwarding_graphs_path=None, loop_ratio=0.1, flows_path=None, policy_path=None, adj_path=None):
    # 初始化数据结构
    adjacency_matrix = nx.to_pandas_adjacency(G, dtype=int)
    nodes = adjacency_matrix.index.tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    N = len(nodes)
    
    # 动态计算故障参数
    k_node_max, k_link_max = calculate_failure_limits(G)
    k_node = random.randint(1, k_node_max) if k_node_max > 0 else 0
    k_link = random.randint(1, k_link_max) if k_link_max > 0 else 0

    # 应用节点故障
    failed_nodes = random.sample(nodes, k=k_node) if k_node > 0 else []
    adj_faulty = adjacency_matrix.copy()
    for node in failed_nodes:
        adj_faulty.loc[node, :] = 0
        adj_faulty.loc[:, node] = 0

    # 应用链路故障
    valid_links = [(src, dst) for src in nodes for dst in nodes 
                   if adj_faulty.loc[src, dst] == 1 
                   and src not in failed_nodes 
                   and dst not in failed_nodes]
    failed_links = random.sample(valid_links, min(k_link, len(valid_links))) if k_link > 0 else []
    for src, dst in failed_links:
        adj_faulty.loc[src, dst] = 0

    # 构建故障后的图
    G_faulty = nx.from_pandas_adjacency(adj_faulty, create_using=nx.DiGraph)

    # 加载原始forwarding_graphs
    forwarding_graphs = None
    if forwarding_graphs_path is not None:
        import pickle
        with open(forwarding_graphs_path, "rb") as f:
            forwarding_graphs = pickle.load(f)
    # 注入虚假环路
    ec_features_aug, forwarding_graphs_aug = inject_fake_loops(ec_features, forwarding_graphs, loop_ratio=loop_ratio) if forwarding_graphs else (ec_features, None)

    # 生成图特征
    graph_features = {
        "adjacency": adj_faulty.values.tolist(),
        "in_degree": [G_faulty.in_degree(n) for n in nodes],
        "out_degree": [G_faulty.out_degree(n) for n in nodes],
        "betweenness": list(nx.betweenness_centrality(G_faulty).values()),
        "ec_counts": [len(ec_features_aug.get(n, [])) for n in nodes],  # EC数量
        "ec_networks": [ec_features_aug.get(n, []) for n in nodes]
    }

    # 生成非图特征
    interface_dim = len(next(iter(interface_encoding.values()))) if interface_encoding else 0
    valid_pairs = []
    for src_idx, dst_idx, path_idx, sla_bw in traffic_pairs:
        src = idx_to_node.get(src_idx)
        dst = idx_to_node.get(dst_idx)
        if src in nodes and dst in nodes:
            valid_pairs.append((src, dst, path_idx, sla_bw))

    traffic_flows = [(src_idx, dst_idx, path_idx, sla_bw) for src_idx, dst_idx, path_idx, sla_bw in traffic_pairs if src_idx in idx_to_node and dst_idx in idx_to_node]
    # 调用增强函数
    enhanced_flows = enhance_flows_with_policies(traffic_flows, policy_data, idx_to_node, node_to_idx)
    non_graph_features = {
        "interfaces": [interface_encoding.get(n, [0]*interface_dim) for n in nodes],
        "traffic_flows": enhanced_flows
    }

    # 生成标签
    labels = generate_labels(G_faulty, valid_pairs, forwarding_graphs_path=forwarding_graphs_path, forwarding_graphs_override=forwarding_graphs_aug,
                            flows_path=flows_path, policy_path=policy_path, adj_path=adj_path)

    return {
        "graph_features": graph_features,
        "non_graph_features": non_graph_features,
        "labels": labels,
        "meta": {
            "original_nodes": nodes,
            "failed_nodes": failed_nodes,
            "failed_links": failed_links,
            "scenario_id": scenario_id
        }
    }

def generate_test_scenario(G, interface_encoding, traffic_pairs, scenario_id, ec_features, forwarding_graphs_path=None, loop_ratio=0.1, flows_path=None, policy_path=None, adj_path=None):
    # 初始化数据结构
    adjacency_matrix = nx.to_pandas_adjacency(G, dtype=int)
    nodes = adjacency_matrix.index.tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    N = len(nodes)

    # 构建无故障的图
    G_faulty = G.copy()

    # 加载原始forwarding_graphs
    forwarding_graphs = None
    if forwarding_graphs_path is not None:
        import pickle
        with open(forwarding_graphs_path, "rb") as f:
            forwarding_graphs = pickle.load(f)
    ec_features_aug, forwarding_graphs_aug = inject_fake_loops(ec_features, forwarding_graphs, loop_ratio=loop_ratio) if forwarding_graphs else (ec_features, None)

    # 生成图特征
    graph_features = {
        "adjacency": adjacency_matrix.values.tolist(),
        "in_degree": [G_faulty.in_degree(n) for n in nodes],
        "out_degree": [G_faulty.out_degree(n) for n in nodes],
        "betweenness": list(nx.betweenness_centrality(G_faulty).values()),
        "ec_counts": [len(ec_features_aug.get(n, [])) for n in nodes],
        "ec_networks": [ec_features_aug.get(n, []) for n in nodes]
    }

    # 生成非图特征
    interface_dim = len(next(iter(interface_encoding.values()))) if interface_encoding else 0
    valid_pairs = []
    for src_idx, dst_idx, path_idx in traffic_pairs:
        src = idx_to_node.get(src_idx)
        dst = idx_to_node.get(dst_idx)
        if src in nodes and dst in nodes:
            valid_pairs.append((src, dst, path_idx))
    
    non_graph_features = {
        "interfaces": [interface_encoding.get(n, [0]*interface_dim) for n in nodes],
        "traffic_flows": [(src_idx, dst_idx) for src_idx, dst_idx, _ in traffic_pairs if src_idx in idx_to_node and dst_idx in idx_to_node]
    }

    # 生成标签
    labels = generate_labels(G_faulty, valid_pairs, forwarding_graphs_path=forwarding_graphs_path, forwarding_graphs_override=forwarding_graphs_aug)

    return {
        "graph_features": graph_features,
        "non_graph_features": non_graph_features,
        "labels": labels,
        "meta": {
            "original_nodes": nodes,
            "failed_nodes": [],
            "failed_links": [],
            "scenario_id": scenario_id
        }
    }


# 3. 填充与掩码生成
def pad_and_mask(features_list, N_max, M_max):
    """改进后的填充和掩码生成函数"""
    padded_data = []
    
    for features in features_list:
        # 获取关键参数
        meta = features["meta"]
        original_nodes = meta["original_nodes"]
        failed_nodes = meta["failed_nodes"]
        original_size = len(original_nodes)
        
        # ====== 生成基础掩码 ======
        mask = [1] * original_size + [0] * (N_max - original_size)
        
        # ====== 应用节点故障标记 ======
        for node in failed_nodes:
            if node in original_nodes:
                idx = original_nodes.index(node)
                mask[idx] = 0  # 将故障节点位置置 0
        
        # ====== 图特征填充 ======
        # 邻接矩阵
        adj_padded = np.zeros((N_max, N_max))
        adj_padded[:original_size, :original_size] = features["graph_features"]["adjacency"]
        features["graph_features"]["adjacency"] = adj_padded.tolist()
        
        # 其他节点级特征
        for key in ["in_degree", "out_degree", "betweenness", "ec_counts"]:
            features["graph_features"][key] += [0] * (N_max - original_size)
        
        # 对ec_networks进行填充
        features["graph_features"]["ec_networks"] += [[]] * (N_max - original_size)
        
        # ====== 非图特征填充 ======
        # 接口编码填充到统一的维度 [N_max, M_max]
        interface_dim = len(features["non_graph_features"]["interfaces"][0])
        interfaces_padded = np.zeros((N_max, M_max))
        interfaces_padded[:original_size, :interface_dim] = features["non_graph_features"]["interfaces"]
        features["non_graph_features"]["interfaces"] = interfaces_padded.tolist()
        # 流量对过滤
        features["non_graph_features"]["traffic_flows"] = [
            (src, dst, sla_bw, lb, acl) for src, dst, sla_bw, lb, acl in features["non_graph_features"]["traffic_flows"]
            if src < N_max and dst < N_max
        ]
        
        # ====== 标签填充 ======
        # 接口利用率标签补全到 N_max
        features["labels"]["high_utilization"] += [0] * (N_max - original_size)
        # 负载均衡标签补全到 N_max
        features["labels"]["load_balancing"] += [0] * (N_max - original_size)
        # loop 标签补全到 N_max
        features["labels"]["loop"] += [0] * (N_max - original_size)
        # 黑洞标签补全到 N_max
        features["labels"]["blackhole"] += [0] * (N_max - original_size)

        # SLA标签补全（flow级，按流数量补）
        if "sla_labels" in features["labels"]:
            num_flows = len(features["non_graph_features"]["traffic_flows"])
            features["labels"]["sla_labels"] += [1] * (num_flows - len(features["labels"]["sla_labels"]))

        
        # ====== 存储最终掩码 ======
        features["mask"] = mask
        padded_data.append(features)
    
    return padded_data

def custom_serializer(obj):
    """自定义序列化函数，确保特定字段紧凑存储"""
    if isinstance(obj, dict):
        # 对特定字段进行紧凑处理
        compact_fields = {
            "graph_features": ["adjacency", "in_degree", "out_degree", 
                              "betweenness", "ec_counts", "ec_networks"],
            "non_graph_features": ["interfaces", "traffic_flows"],
            "labels": ["isolated", "loop", "blackhole", "reachability", 
                      "high_utilization", "load_balancing","link_overload_pairs", "sla_labels"],
            "meta": ["original_nodes", "failed_nodes", "failed_links"],
        }
        
        for section, fields in compact_fields.items():
            if section in obj:
                for field in fields:
                    if field in obj[section]:
                        # 确保这些字段以紧凑形式存储
                        obj[section][field] = json.dumps(
                            obj[section][field], 
                            separators=(',', ':'),
                            cls=NumpyEncoder
                        )
        # 单独处理 mask 字段
        if "mask" in obj:
            obj["mask"] = json.dumps(obj["mask"], separators=(',', ':'))
    return obj

def enhance_flows_with_policies(flows, policy_data, idx_to_node, node_to_idx):
    """
    将LB和ACL策略信息整合到流量特征中
    参数:
        flows: [(src_id, dst_id, path_ids, sla_bw), ...] 
        policy_data: 包含LB和ACL规则的字典(使用节点名称)
        idx_to_node: 字典{节点索引: 节点名称}
        node_to_idx: 字典{节点名称: 节点索引}
    返回格式: [src_id, dst_id, sla_bw, lb_vector, acl_rules]
             其中acl_rules格式: [(限速节点索引, 速率限制), ...]
    """
    enhanced_flows = []
    
    # ====== LB规则处理部分 ====== (保持不变)
    all_next_hops = set()
    for node_rules in policy_data.get('lb_rules', {}).values():
        for rule in node_rules.values():
            all_next_hops.update(rule['next_hops'])
    
    next_hop_name_to_idx = {nh.lower(): idx for idx, nh in enumerate(sorted(all_next_hops))}
    lb_vector_size = len(next_hop_name_to_idx)
    
    lb_rules_info = []
    for node_name, rules in policy_data.get('lb_rules', {}).items():
        node_idx = node_to_idx.get(node_name.lower())
        if node_idx is not None:
            for flow_key, rule in rules.items():
                src_dst = flow_key.strip("[]").split(",")
                if len(src_dst) == 2:
                    src_name, dst_name = src_dst
                    src_idx = node_to_idx.get(src_name.strip().lower())
                    dst_idx = node_to_idx.get(dst_name.strip().lower())
                    
                    if src_idx is not None and dst_idx is not None:
                        lb_rules_info.append({
                            'node_idx': node_idx,
                            'src_idx': src_idx,
                            'dst_idx': dst_idx,
                            'next_hops': [node_to_idx[nh.lower()] for nh in rule['next_hops'] 
                                        if nh.lower() in node_to_idx],
                            'weights': rule['weights']
                        })

    # ====== ACL规则处理部分 ====== (简化版)
    acl_rules_info = []
    for node_name, rules in policy_data.get('acl_rules', {}).items():
        node_idx = node_to_idx.get(node_name.lower())
        if node_idx is not None:
            for flow_key, rule in rules.items():
                src_dst = flow_key.strip("[]").split(",")
                if len(src_dst) == 2:
                    src_name, dst_name = src_dst
                    src_idx = node_to_idx.get(src_name.strip().lower())
                    dst_idx = node_to_idx.get(dst_name.strip().lower())
                    
                    if src_idx is not None and dst_idx is not None:
                        acl_rules_info.append({
                            'node_idx': node_idx,  # 应用限速的节点
                            'src_idx': src_idx,    # 流量的源
                            'dst_idx': dst_idx,    # 流量的目的
                            'rate_limit': rule['rate_limit']
                        })

    # ====== 流量处理部分 ======
    for src_idx, dst_idx, path_indices, sla_bw in flows:
        # 初始化LB向量
        lb_vector = [0.0] * lb_vector_size
        acl_nodes = []  # 存储应用限速的节点和限速值
        
        # 查找匹配的LB规则 (保持不变)
        for rule in lb_rules_info:
            if src_idx == rule['src_idx'] and dst_idx == rule['dst_idx']:
                if rule['node_idx'] in path_indices:
                    for nh_idx, weight in zip(rule['next_hops'], rule['weights']):
                        nh_name = idx_to_node[nh_idx].lower()
                        if nh_name in next_hop_name_to_idx:
                            lb_vector[next_hop_name_to_idx[nh_name]] = weight
        
        # 查找匹配的ACL规则 (简化版)
        for rule in acl_rules_info:
            if (src_idx == rule['src_idx'] and 
                dst_idx == rule['dst_idx'] and
                rule['node_idx'] in path_indices):  # 只需检查节点是否在路径中
                
                acl_nodes.append((
                    rule['node_idx'],  # 限速节点索引
                    rule['rate_limit']  # 速率限制值
                ))
        
        # 构建最终特征向量
        enhanced_vector = [src_idx, dst_idx, sla_bw, lb_vector, acl_nodes]
        enhanced_flows.append(enhanced_vector)
    
    return enhanced_flows

# 4. 主流程
def main():
    topologies = {
        # "Arnes": {
        #     "adjacency": "./Arnes_abs_order_1_72/adjacency_matrix.csv",
        #     "interfaces": "./Arnes_abs_order_1_72/interface_onehot_encoding.json",
        #     "traffic": "./Arnes_abs_order_1_72/flows_with_paths.json",
        #     "middlebox": "./Arnes_abs_order_1_72/middlebox_policies.json",
        #     "equivalence_classes": "./Arnes_abs_order_1_72/equivalence_classes.csv"
        # },
        "Uninett2011": {
            "adjacency": "./Uninett2011_abs_order_1_225/adjacency_matrix.csv",
            "interfaces": "./Uninett2011_abs_order_1_225/interface_onehot_encoding.json",
            "traffic": "./Uninett2011_abs_order_1_225/flows_with_paths.json",
            "middlebox": "./Uninett2011_abs_order_1_225/middlebox_policies.json",
            "equivalence_classes": "./Uninett2011_abs_order_1_225/equivalence_classes.csv"        
        },
        # "Columbus": {"adjacency": "./Columbus_abs_order_1_655/adjacency_matrix.csv",
        #             "interfaces": "./Columbus_abs_order_1_655/interface_onehot_encoding.json",
        #             "traffic": "./Columbus_abs_order_1_655/traffic_pairs.json",
        #             "equivalence_classes": "./Columbus_abs_order_1_655/equivalence_classes.csv"},
        # "UsCarrie": {"adjacency": "./UsCarrie_abs_simple_2_43/adjacency_matrix.csv", 
        #         "interfaces": "./UsCarrie_abs_simple_2_43/interface_onehot_encoding.json",
        #         "traffic": "./UsCarrie_abs_simple_2_43/traffic_pairs.json",
        #         "equivalence_classes": "./UsCarrie_abs_simple_2_43/equivalence_classes.csv"}
    }

    # 确定全局最大节点数和最大接口数
    N_max = max([pd.read_csv(v["adjacency"], index_col=0).shape[0] for v in topologies.values()])
    M_max = max([len(next(iter(json.load(open(v["interfaces"])).values()))) for v in topologies.values()])

    # N_max = 72
    # M_max = 173

    all_scenarios = []
    for topology_name, paths in topologies.items():
        # 加载数据
        adjacency = pd.read_csv(paths["adjacency"], index_col=0).rename(columns=str.lower, index=str.lower)
        G = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)
        with open(paths["interfaces"], "r") as f:
            interface_encoding = json.load(f)
        with open(paths["traffic"], "r") as f:
            flows = json.load(f)
        with open(paths["middlebox"], "r") as f:
            policy_data = json.load(f)
        # 构造(src_idx, dst_idx, path_idx)三元组，全部用节点索引
        node_to_idx = {n: i for i, n in enumerate(G.nodes())}
        traffic_pairs = []
        for flow in flows:
            if "src" in flow and "dst" in flow and "path" in flow and "sla_bw" in flow:
                src = flow["src"].lower()
                dst = flow["dst"].lower()
                path = [p.lower() for p in flow["path"]]
                sla_bw = flow["sla_bw"]  # Get the bandwidth requirement
                if src in node_to_idx and dst in node_to_idx and all(p in node_to_idx for p in path):
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    path_idx = [node_to_idx[p] for p in path]
                    traffic_pairs.append((src_idx, dst_idx, path_idx, sla_bw)) 
                    
        ec_features = defaultdict(list)
        with open(paths["equivalence_classes"], "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                next_hop = row["Next_Hop"]
                networks = row["Networks"].split("|")
                if "_" in next_hop:  # 只处理节点_接口格式
                    node = next_hop.split("_")[0].lower()
                    ec_features[node].extend(networks)

        # 并行生成场景
        with ProcessPoolExecutor(max_workers=8) as executor:
            forwarding_graphs_path = paths["traffic"].replace("flows_with_paths.json", "forwarding_graphs.pkl")
            func = partial(generate_failure_scenario, G, interface_encoding, traffic_pairs, ec_features=ec_features, policy_data=policy_data, forwarding_graphs_path=forwarding_graphs_path, flows_path=paths["traffic"], policy_path=paths["middlebox"], adj_path=paths["adjacency"])
            scenarios = list(tqdm(executor.map(func, range(100)), total=100, desc=f"生成 {topology_name} 场景"))
        
        # 输出场景里的流量对信息
        # for s in scenarios:
        #     print(f"Scenario ID: {s['meta']['scenario_id']}, Traffic Flows: {s['non_graph_features']['traffic_flows']}")

        # 填充和添加掩码
        padded_scenarios = pad_and_mask(scenarios, N_max, M_max)
        all_scenarios.extend(padded_scenarios)

    # 保存结果
    with open("uninett_augmented_dataset_all.json", "w") as f:
        f.write("[\n")  # 添加文件开头的中括号
        for i, scenario in enumerate(all_scenarios):
            compact_item = json.loads(
                json.dumps(scenario, cls=NumpyEncoder, default=custom_serializer),
                object_hook=custom_serializer
            )
            json.dump(compact_item, f, indent=2, separators=(',', ': '))
            if i < len(all_scenarios) - 1:
                f.write(",\n")
        f.write("\n]")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()
