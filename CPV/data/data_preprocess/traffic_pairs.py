import csv
import random
import networkx as nx
import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# 1. 构建前缀到连通分量的映射（基于强连通分量）
def build_prefix_node_sets(file_path):
    prefix_sets = {}
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 提取基础节点名称（忽略 Equivalence_Class 中的接口名称）
            src = row["Source"].strip().lower()
            dst = row["Destination"].strip().lower()
            
            # 使用 Source 和 Destination 构建连通分量
            if src not in prefix_sets:
                prefix_sets[src] = set()
            prefix_sets[src].add(dst)
    
    # 过滤节点数 ≥2 的连通分量
    valid_prefixes = {src: {src, *dsts} for src, dsts in prefix_sets.items() if len(dsts) >= 1}
    return valid_prefixes

# 2. 生成流量对（含路径信息、失败计数器、方向性处理）
def generate_traffic_pairs(G, prefix_sets, num_pairs=100, max_attempts=1000):
    traffic_pairs = set()
    nodes = list(G.nodes())  # 邻接矩阵中的基础节点列表
    node_to_index = {node: index for index, node in enumerate(nodes)}  # 节点名到索引的映射
    
    with tqdm(total=num_pairs, desc="生成流量对") as pbar:
        attempt = 0
        while len(traffic_pairs) < num_pairs and attempt < max_attempts:
            # 1. 随机选择一个源节点（基于前缀的有效性）
            src = random.choice(nodes)
            
            # 2. 从该源节点的可达目标中随机选择
            if src in prefix_sets:
                possible_dsts = list(prefix_sets[src])
                dst = random.choice(possible_dsts)
            else:
                dst = random.choice(nodes)
            
            # 3. 检查节点对是否有效
            if src == dst:
                continue
            
            # 4. 检查路径有效性
            if nx.has_path(G, src, dst):
                src_index = node_to_index[src]
                dst_index = node_to_index[dst]
                traffic_pairs.add((src_index, dst_index))
                pbar.update(1)
            
            attempt += 1
    
    return list(traffic_pairs)[:num_pairs]
# 3. 主流程
if __name__ == "__main__":
    # 步骤1：构建邻接矩阵拓扑图（基础节点名称）
    current_file = "Arnes_abs_order_1_72"
    adjacency_df = pd.read_csv(f"./{current_file}/adjacency_matrix.csv", index_col=0)
    nodes = [node.strip().lower() for node in adjacency_df.index]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for src in adjacency_df.index:
        src_lower = src.strip().lower()
        for dst in adjacency_df.columns:
            dst_lower = dst.strip().lower()
            if adjacency_df.loc[src, dst] == 1:
                G.add_edge(src_lower, dst_lower)
    
    # 步骤2：构建前缀到节点集合的映射（仅基础节点）
    prefix_sets = build_prefix_node_sets(f"./{current_file}/forwarding_graphs.csv")
    
    # 步骤3：生成流量对
    traffic_pairs = generate_traffic_pairs(G, prefix_sets, num_pairs=100)
    
    # 步骤4：保存结果
    with open(f"./{current_file}/traffic_pairs.json", "w") as f:
        json.dump(traffic_pairs, f, indent=2)
    
    print(f"已生成 {len(traffic_pairs)} 个流量对，基于基础节点名称。")