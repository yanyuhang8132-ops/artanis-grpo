import json
import random
import numpy as np
import networkx as nx
import pandas as pd
import time



def load_adjacency_matrix(filepath):
    df = pd.read_csv(filepath, index_col=0)
    node_names = list(df.columns)
    matrix = df.values
    return matrix, node_names

def find_lb_candidate(path, degrees, used_lbs):
    for i in range(1, len(path) - 2):  # 保证LB后至少还有两跳
        node = path[i]
        if degrees[node] >= 3 and node not in used_lbs:
            return node
    return None

def gen_weight_pair():
    candidates = [(i, 100 - i) for i in range(30, 71, 5)]  # e.g. 30:70 to 70:30
    return [round(a / 100, 2) for a in random.choice(candidates)]

def parse_cfg_interface_ips(cfg_folder, node_names):
    """
    解析每个节点的cfg文件，返回如下结构：
    {
        'Koper': {
            'Ljubljana': '10.0.0.43',
            ...
        },
        ...
    }
    """
    import os
    import re
    node2peer2ip = {}
    for node in node_names:
        cfg_path = os.path.join(cfg_folder, f"{node}.cfg")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        peer2ip = {}
        current_iface = None
        current_ip = None
        current_desc = None
        for line in lines:
            line = line.strip()
            if line.startswith('interface '):
                current_iface = line.split()[1]
                current_ip = None
                current_desc = None
            elif line.startswith('ip address '):
                m = re.match(r'ip address ([0-9.]+) ', line)
                if m:
                    current_ip = m.group(1)
            elif line.startswith('description '):
                m = re.match(r'description "?To ([^"\n]+)"?', line)
                if m:
                    current_desc = m.group(1)
            elif line == '!':
                if current_desc and current_ip:
                    peer2ip[current_desc] = current_ip
                current_iface = None
                current_ip = None
                current_desc = None
        node2peer2ip[node] = peer2ip
    return node2peer2ip

def generate_flows_and_middleboxes(
    adj_path, 
    num_flows=20000, 
    num_lb_flows=5000, 
    num_acl_flows=5000, 
    min_bw=10, 
    max_bw=30, 
    cfg_folder=None,
    max_link_capacity=4000  # 新增参数，控制最大链路负载
):
    adj_matrix, node_names = load_adjacency_matrix(adj_path)
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    degrees = dict(G.degree())
    index2name = {i: name for i, name in enumerate(node_names)}
    name2index = {name: i for i, name in index2name.items()}

    # 新增：解析cfg接口IP
    node2peer2ip = parse_cfg_interface_ips(cfg_folder, node_names) if cfg_folder else {}

    all_nodes = list(G.nodes)
    flows = []
    used_pairs = set()
    used_lbs = set()
    lb_rules = {}
    acl_rules = {}
    
    # 新增：链路负载跟踪字典
    link_load = {}  # 格式: (node1_index, node2_index) -> 总负载

    # Generate basic flows with load control
    while len(flows) < num_flows:
        src, dst = random.sample(all_nodes, 2)
        if (src, dst) in used_pairs or not nx.has_path(G, src, dst):
            continue
        
        path = nx.shortest_path(G, src, dst)
        
        # 计算路径上所有链路的最小剩余容量
        min_remaining = float('inf')
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            current_load = link_load.get((u, v), 0)
            remaining = max_link_capacity - current_load
            min_remaining = min(min_remaining, remaining)
        
        # 如果最小剩余容量不足，跳过该流
        if min_remaining < min_bw:
            continue
        
        # 确定实际分配的带宽（不超过剩余容量和max_bw）
        actual_bw = random.randint(
            min_bw, 
            min(max_bw, min_remaining))
        
        flow = {
            "flow_id": f"F{len(flows)+1}",
            "src": index2name[src],
            "dst": index2name[dst],
            "sla_bw": actual_bw,
            "path": [index2name[i] for i in path]
        }
        
        # 更新链路负载
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            link_load[(u, v)] = link_load.get((u, v), 0) + actual_bw
        
        # 新增：查找src到path[1]的本地接口IP，dst收到path[-2]的本地接口IP
        def get_path_ips(path_names):
            src_ip = None
            dst_ip = None
            if node2peer2ip and len(path_names) >= 2:
                src_node = path_names[0]
                next_hop = path_names[1]
                # 大小写不敏感查找src_ip
                src_peer2ip = node2peer2ip.get(src_node, {})
                src_ip = None
                for peer, ip in src_peer2ip.items():
                    if peer.lower() == next_hop.lower():
                        src_ip = ip
                        break
                if src_ip is None:
                    print(f"[DEBUG] src_ip is None | path={path_names} | src_node={src_node} | next_hop={next_hop} | src_node2peer2ip={src_peer2ip}")
                dst_node = path_names[-1]
                prev_hop = path_names[-2]
                dst_peer2ip = node2peer2ip.get(dst_node, {})
                dst_ip = None
                for peer, ip in dst_peer2ip.items():
                    if peer.lower() == prev_hop.lower():
                        dst_ip = ip
                        break
                if dst_ip is None:
                    print(f"[DEBUG] dst_ip is None | path={path_names} | dst_node={dst_node} | prev_hop={prev_hop} | dst_node2peer2ip={dst_peer2ip}")
            return src_ip, dst_ip
        
        src_ip, dst_ip = get_path_ips([index2name[i] for i in path])
        flow["src_interface_ip"] = src_ip
        flow["dst_interface_ip"] = dst_ip
        
        flows.append(flow)
        used_pairs.add((src, dst))

    # Strong attempt to construct exactly num_lb_flows with LB
    candidate_flows = flows.copy()
    random.shuffle(candidate_flows)
    lb_flows = []

    for flow in candidate_flows:
        if len(lb_flows) >= num_lb_flows:
            break

        path_full = [name2index[n] for n in flow["path"]]
        lb_node = find_lb_candidate(path_full, degrees, used_lbs)
        if lb_node is None:
            continue
        lb_name = index2name[lb_node]

        lb_idx = path_full.index(lb_node)
        if lb_idx + 1 >= len(path_full):
            continue
        original_next = path_full[lb_idx + 1]
        prev_hop = path_full[lb_idx - 1] if lb_idx > 0 else None
        dst = path_full[-1]
        visited_set = set(path_full[:lb_idx + 1])

        alt_candidates = [
            n for n in G.successors(lb_node)
            if n not in {original_next, prev_hop} and n not in visited_set
        ]

        for alt_next in alt_candidates:
            try:
                subgraph = G.copy()
                subgraph.remove_nodes_from(visited_set)
                if nx.has_path(subgraph, alt_next, dst):
                    alt_tail = nx.shortest_path(subgraph, alt_next, dst)
                    prefix = path_full[:lb_idx + 1]
                    
                    # 检查两条路径的链路负载
                    def check_path_capacity(path_indices, new_bw):
                        for i in range(len(path_indices)-1):
                            u, v = path_indices[i], path_indices[i+1]
                            if link_load.get((u, v), 0) + new_bw > max_link_capacity:
                                return False
                        return True
                    
                    # 获取权重并计算两条路径的带宽分配
                    weights = gen_weight_pair()
                    original_bw = flow["sla_bw"]
                    path1_bw = int(original_bw * weights[0])  # 主路径分配
                    path2_bw = original_bw - path1_bw         # 备用路径分配
                    
                    # 检查两条路径容量
                    if (check_path_capacity(path_full, path1_bw) and 
                        check_path_capacity(prefix + [alt_next] + alt_tail[1:], path2_bw)):
                        
                        # 创建分流配置
                        flow["path1"] = [index2name[i] for i in path_full]
                        flow["path2"] = [index2name[i] for i in prefix + [alt_next] + alt_tail[1:]]
                        
                        # 为path1和path2分别写入接口ip
                        for pkey in ["path1", "path2"]:
                            if pkey in flow:
                                src_ip, dst_ip = get_path_ips(flow[pkey])
                                flow[f"src_interface_ip_{pkey}"] = src_ip
                                flow[f"dst_interface_ip_{pkey}"] = dst_ip

                        key = f"[{flow['src']},{flow['dst']}]"
                        lb_rules.setdefault(lb_name, {})
                        lb_rules[lb_name][key] = {
                            "next_hops": [index2name[original_next], index2name[alt_next]],
                            "weights": weights,
                            "prev_hop": index2name[prev_hop] if prev_hop is not None else None
                        }

                        # 更新链路负载
                        for i in range(len(path_full)-1):
                            u, v = path_full[i], path_full[i+1]
                            link_load[(u, v)] = link_load.get((u, v), 0) + path1_bw
                        
                        for i in range(len(prefix + [alt_next] + alt_tail[1:])-1):
                            u, v = (prefix + [alt_next] + alt_tail[1:])[i], (prefix + [alt_next] + alt_tail[1:])[i+1]
                            link_load[(u, v)] = link_load.get((u, v), 0) + path2_bw

                        # 添加ACL规则
                        tail_idx = flow["path2"].index(lb_name) + 1
                        acl_candidates = flow["path2"][tail_idx:]
                        # 过滤掉lb节点、起点和终点
                        acl_candidates = [n for n in acl_candidates if n != flow["src"] and n != flow["dst"] and name2index[n] not in used_lbs]
                        if acl_candidates:
                            acl_node = random.choice(acl_candidates)
                            acl_rules.setdefault(acl_node, {})
                            acl_ratio = weights[1]  # path2 对应 ACL 路径
                            acl_rules[acl_node][key] = {
                                "rate_limit": int(flow["sla_bw"] * acl_ratio)
                            }

                        lb_flows.append(flow)
                        used_lbs.add(lb_node)
                        break
            except nx.NetworkXNoPath:
                continue

    # Extra ACL for non-LB flows
    non_lb_flows = [f for f in flows if f not in lb_flows]
    extra_acl_flows = random.sample(non_lb_flows, min(num_acl_flows, len(non_lb_flows)))
    for flow in extra_acl_flows:
        key = f"[{flow['src']},{flow['dst']}]"
        path_nodes = flow["path"]
        middle_nodes = path_nodes[1:-1]
        # 过滤掉已被选中的lb节点
        middle_nodes = [n for n in middle_nodes if name2index[n] not in used_lbs]
        if not middle_nodes:
            continue
        acl_node = random.choice(middle_nodes)
        acl_rules.setdefault(acl_node, {})
        acl_rules[acl_node][key] = {
            "rate_limit": flow["sla_bw"]
        }

    # 最终链路负载检查
    def print_link_load_stats():
        if not link_load:
            print("无链路负载数据")
            return
        
        max_load = max(link_load.values())
        min_load = min(link_load.values())
        overloaded = sum(1 for load in link_load.values() if load > max_link_capacity)
        
        print(f"\n链路负载统计:")
        print(f"- 最大负载: {max_load:.1f}")
        print(f"- 最小负载: {min_load:.1f}")
        print(f"- 过载链路数: {overloaded} (超过{max_link_capacity})")
        print(f"- 总链路数: {len(link_load)}")
    
    print_link_load_stats()

    # Write flows
    with open("Colt_flows_with_paths.json", "w") as f:
        json.dump(flows, f, indent=2)

    # Write middlebox policies
    policies = {
        "scenario_id": 0,
        "lb_rules": lb_rules,
        "acl_rules": acl_rules
    }
    with open("Colt_middlebox_policies.json", "w") as f:
        json.dump(policies, f, indent=2)
    
    

if __name__ == "__main__":
    # 开始计时
    start_time = time.time()
    
    current_file = "Colt_abs_order_1_610"
    cfg_folder = f"./{current_file}/configs"
    generate_flows_and_middleboxes(
        f"./{current_file}/adjacency_matrix.csv", 
        cfg_folder=cfg_folder,
        max_link_capacity=50000000  # 可以调整这个值来控制最大链路负载
    )
    print(f"结果保存完成，耗时: {time.time() - start_time:.2f}秒")