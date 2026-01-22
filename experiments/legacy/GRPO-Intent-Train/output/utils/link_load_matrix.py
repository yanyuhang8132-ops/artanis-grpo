import json
import pandas as pd
import numpy as np

def calculate_link_load_matrix(flows_path, policy_path, adj_path, output_path=None):
    # 读取节点名
    adj_df = pd.read_csv(adj_path, index_col=0)
    nodes = [n.lower() for n in adj_df.index]
    node2idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # 初始化负载矩阵
    load_matrix = np.zeros((N, N))

    # 读取流量和策略
    with open(flows_path, "r") as f:
        flows = json.load(f)
    with open(policy_path, "r") as f:
        policies = json.load(f)
    lb_rules = policies.get("lb_rules", {})

    # 辅助：获取路径上的链路对
    get_links = lambda path: [(path[i], path[i+1]) for i in range(len(path)-1)] if len(path) > 1 else []

    def to_lower_path(path):
        return [p.lower() for p in path]

    def match_lb(flow, lb_rules):
        src = flow.get("src", "").lower()
        dst = flow.get("dst", "").lower()
        for lb_node, rules in lb_rules.items():
            for key, rule in rules.items():
                try:
                    src_rule, dst_rule = json.loads(key.replace('[','[\"').replace(',', '\",\"').replace(']','\"]'))
                    if src == src_rule.lower() and dst == dst_rule.lower():
                        return lb_node.lower(), rule
                except Exception:
                    continue
        return None, None

    for flow in flows:
        bw = flow.get("sla_bw", 0)
        src = flow.get("src", "").lower()
        dst = flow.get("dst", "").lower()
        lb_node, lb_rule = match_lb(flow, lb_rules)
        if lb_node and lb_rule and "path1" in flow and "path2" in flow:
            # 1. src到LB节点前的链路全部加原始带宽
            path = to_lower_path(flow["path"])
            if lb_node in path:
                lb_idx = path.index(lb_node)
                pre_lb_path = path[:lb_idx+1]
                for u, v in get_links(pre_lb_path):
                    if u in node2idx and v in node2idx:
                        load_matrix[node2idx[u], node2idx[v]] += bw
            # 2. LB节点后分流
            for i, pkey in enumerate(["path1", "path2"]):
                if pkey in flow:
                    weight = lb_rule["weights"][i] if i < len(lb_rule["weights"]) else 0.5
                    p = to_lower_path(flow[pkey])
                    if lb_node in p:
                        lb_idx = p.index(lb_node)
                        post_lb_path = p[lb_idx:]
                        for u, v in get_links(post_lb_path):
                            if u in node2idx and v in node2idx:
                                load_matrix[node2idx[u], node2idx[v]] += bw * weight
        else:
            # 普通流，直接全路径加原始带宽
            if "path" in flow:
                path = to_lower_path(flow["path"])
                for u, v in get_links(path):
                    if u in node2idx and v in node2idx:
                        load_matrix[node2idx[u], node2idx[v]] += bw

    # 读取原始邻接矩阵，标记物理不连通的链路为-1
    adj_bool = adj_df.values.astype(int)
    for i in range(N):
        for j in range(N):
            if adj_bool[i, j] == 0 and load_matrix[i, j] == 0:
                load_matrix[i, j] = -1
    load_df = pd.DataFrame(load_matrix, index=nodes, columns=nodes)
    if output_path:
        load_df.to_csv(output_path)
    # 返回负载矩阵和节点名
    return load_matrix, nodes

if __name__ == "__main__":
    flows_path = "./flows_with_paths.json"
    policy_path = "./middlebox_policies.json"
    adj_path = "./Arnes_abs_order_1_72/adjacency_matrix.csv"
    output_path = "./Arnes_abs_order_1_72/link_load_matrix.csv"
    load_matrix, nodes = calculate_link_load_matrix(flows_path, policy_path, adj_path, output_path)
    print(f"链路负载邻接矩阵已生成：{output_path}")
    positive_loads = load_matrix[load_matrix > 0]
    if positive_loads.size > 0:
        print(f"最大链路带宽: {positive_loads.max()}")
        print(f"最小链路带宽: {positive_loads.min()}")
        print(f"所有链路负载总和: {positive_loads.sum()}")
        print(f"平均链路带宽: {positive_loads.mean()}")
    else:
        print("没有正向链路带宽数据。")


