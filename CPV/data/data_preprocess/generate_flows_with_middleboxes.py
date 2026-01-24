import json
import random
import numpy as np
import networkx as nx
import pandas as pd

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

def generate_flows_and_middleboxes(adj_path, num_flows=30, num_lb_flows=5, num_acl_flows=5, min_bw=30, max_bw=100, cfg_folder=None):
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

    # Generate basic flows
    while len(flows) < num_flows:
        src, dst = random.sample(all_nodes, 2)
        if (src, dst) in used_pairs or not nx.has_path(G, src, dst):
            continue
        path = nx.shortest_path(G, src, dst)
        sla_bw = random.randint(min_bw, max_bw)
        flow = {
            "flow_id": f"F{len(flows)+1}",
            "src": index2name[src],
            "dst": index2name[dst],
            "sla_bw": sla_bw,
            "path": [index2name[i] for i in path]
        }
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
                    flow["path1"] = [index2name[i] for i in path_full]
                    flow["path2"] = [index2name[i] for i in prefix + [alt_next] + alt_tail[1:]]
                    # 新增：为path1和path2分别写入接口ip
                    for pkey in ["path1", "path2"]:
                        if pkey in flow:
                            src_ip, dst_ip = get_path_ips(flow[pkey])
                            flow[f"src_interface_ip_{pkey}"] = src_ip
                            flow[f"dst_interface_ip_{pkey}"] = dst_ip

                    key = f"[{flow['src']},{flow['dst']}]"
                    lb_rules.setdefault(lb_name, {})
                    lb_rules[lb_name][key] = {
                        "next_hops": [index2name[original_next], index2name[alt_next]],
                        "weights": gen_weight_pair(),
                        "prev_hop": index2name[prev_hop] if prev_hop is not None else None
                    }

                    tail_idx = flow["path2"].index(lb_name) + 1
                    acl_candidates = flow["path2"][tail_idx:]
                    # 过滤掉lb节点、起点和终点
                    acl_candidates = [n for n in acl_candidates if n != flow["src"] and n != flow["dst"] and name2index[n] not in used_lbs]
                    if acl_candidates:
                        acl_node = random.choice(acl_candidates)
                        acl_rules.setdefault(acl_node, {})
                        weights = lb_rules[lb_name][key]["weights"]
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

    # Write flows
    with open("flows_with_paths.json", "w") as f:
        json.dump(flows, f, indent=2)

    # Write middlebox policies
    policies = {
        "scenario_id": 0,
        "lb_rules": lb_rules,
        "acl_rules": acl_rules
    }
    with open("middlebox_policies.json", "w") as f:
        json.dump(policies, f, indent=2)

if __name__ == "__main__":
    current_file = "Arnes_abs_order_1_72"
    cfg_folder = f"./{current_file}/configs"
    generate_flows_and_middleboxes(f"./{current_file}/adjacency_matrix.csv", cfg_folder=cfg_folder)
