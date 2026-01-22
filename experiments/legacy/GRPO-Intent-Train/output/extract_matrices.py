import json
import numpy as np
import pandas as pd
import re
import os
import networkx as nx

def parse_topology(topology_str, device_list):
    """
    从Topology Info字符串中解析所有连接和负载
    返回邻接矩阵和设备名称到索引的映射
    """
    matrix_size = len(device_list)
    name_to_idx = {name: idx for idx, name in enumerate(device_list)}
    adj_matrix = np.zeros((matrix_size, matrix_size))

    for line in topology_str.strip().split("\n"):
        match = re.match(r"(.+?) \(([\d.]+)\) ↔ (.+?) \(([\d.]+)\)(?: \[(\d+\.?\d*) Mbps\])?", line.strip())
        if match:
            node_a, ip_a, node_b, ip_b, load = match.groups()
            idx_a = name_to_idx[node_a.strip()]
            idx_b = name_to_idx[node_b.strip()]
            weight = float(load) if load else 0.0
            adj_matrix[idx_a][idx_b] = weight
            adj_matrix[idx_b][idx_a] = weight
    return adj_matrix, name_to_idx

def extract_sla_intent(user_input):
    match = re.search(r"Deploy a service flow from (.+?) to (.+?), with SLA guaranteeing a minimum bandwidth of (\d+)", user_input)
    if match:
        src, dst, sla = match.groups()
        return src.strip(), dst.strip(), float(sla)
    return None, None, None

def check_sla_satisfaction(adj_matrix, name_to_idx, src, dst, sla, k=3):
    if src not in name_to_idx or dst not in name_to_idx:
        return False, []

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
            checked_paths.append(([idx_to_name[i] for i in path], min_bw))
            if min_bw >= sla:
                satisfied = True
        return satisfied, checked_paths
    except:
        return False, []

def extract_and_save_matrices(jsonl_path, output_dir):
    with open(jsonl_path, 'r') as f:
        samples = [json.loads(line) for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)
    sla_results = []

    for idx, sample in enumerate(samples):
        user_input = sample["user_input"]
        device_match = re.search(r"\[Device List\]\n(.+?)\n\n", user_input, re.DOTALL)
        if not device_match:
            print(f"Sample {idx}: device list not found.")
            continue
        device_names = [d.strip() for d in device_match.group(1).split(',')]

        topology_match = re.search(r"\[Topology Info\]\n(.+?)\n##", user_input, re.DOTALL)
        if not topology_match:
            print(f"Sample {idx}: topology info not found.")
            continue
        topology_str = topology_match.group(1)

        # 生成邻接矩阵
        adj_matrix, name_to_idx = parse_topology(topology_str, device_names)

        # 保存邻接矩阵 CSV（带表头，保留两位小数）
        df = pd.DataFrame(adj_matrix, index=device_names, columns=device_names).round(2)
        df.to_csv(os.path.join(output_dir, f"adj_matrix_{idx}.csv"))

        # 提取意图 + 检查路径SLA
        src, dst, sla = extract_sla_intent(user_input)
        sla *= 2
        if src and dst:
            satisfied, paths = check_sla_satisfaction(adj_matrix, name_to_idx, src, dst, sla)
            path_bws = [round(p[1], 2) for p in paths]

            # 提取第一条满足SLA的路径的链路带宽细节
            link_details = []
            for path, min_bw in paths:
                if min_bw >= sla:
                    for i in range(len(path) - 1):
                        n1, n2 = path[i], path[i + 1]
                        bw = adj_matrix[name_to_idx[n1]][name_to_idx[n2]]
                        link_details.append((f"{n1} -> {n2}", round(bw, 2)))
                    break  # 只提取第一条满足的路径

            sla_results.append({
                "sample_id": idx,
                "source": src,
                "destination": dst,
                "sla": sla,
                "satisfied": satisfied,
                "min_bw_top_paths": path_bws,
                "satisfying_path_links": link_details
            })


    # 保存SLA检测结果
    if sla_results:
        df_result = pd.DataFrame(sla_results)
        df_result.to_csv(os.path.join(output_dir, "sla_check_results.csv"), index=False)
        print("SLA检测结果已保存为 sla_check_results.csv")

    print("邻接矩阵与SLA检测处理完毕。")

# 示例调用（脚本入口）
if __name__ == "__main__":
    extract_and_save_matrices("inference_5_samples.jsonl", "adj_matrices")
