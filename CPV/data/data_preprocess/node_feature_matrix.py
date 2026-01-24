import pandas as pd
import csv

# 1. 读取邻接矩阵并处理节点名称
current_file = "UsCarrie_abs_simple_2_43"
adjacency_df = pd.read_csv(f"./{current_file}/adjacency_matrix.csv", index_col=0)
nodes = adjacency_df.index.str.strip().str.lower().tolist()  # 统一为小写
adjacency_vectors = {node.lower(): adjacency_df.loc[node].values.astype(int) for node in adjacency_df.index}

# 2. 读取节点的入度和出度
degrees = {}
with open(f"./{current_file}/degrees_output.txt", "r") as f:
    current_node = None
    for line in f:
        line = line.strip()
        if line.startswith("设备:"):
            current_node = line.split(": ")[1].strip().lower()
            degrees[current_node] = {"in": 0, "out": 0}
        elif "入度:" in line:
            degrees[current_node]["in"] = int(line.split(": ")[1])
        elif "出度:" in line:
            degrees[current_node]["out"] = int(line.split(": ")[1])

# 3. 读取介数中心性
betweenness = {}
with open(f"./{current_file}/betweenness_centrality_output.txt", "r") as f:
    for line in f:
        if "设备:" in line:
            parts = line.strip().split(", ")
            node = parts[0].split(": ")[1].strip().lower()
            cb = float(parts[1].split(": ")[1])
            betweenness[node] = cb

# 4. 统计等价类（EC）特征
ec_counts = {}
with open(f"./{current_file}/equivalence_classes.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        next_hop = row["Next_Hop"]
        if "_" in next_hop:  # 只处理节点_接口格式
            node = next_hop.split("_")[0].lower()
            ec_counts[node] = ec_counts.get(node, 0) + 1

# 5. 构建特征矩阵 H⁰ = [A, d_in, d_out, CB, EC]
feature_matrix = []
for node in nodes:
    node_lower = node.lower()
    
    # 邻接向量（A_i*）
    adj_vector = adjacency_vectors.get(node_lower, [0]*len(nodes))
    
    # 入度（d_in）和出度（d_out）
    in_deg = degrees.get(node_lower, {}).get("in", 0)
    out_deg = degrees.get(node_lower, {}).get("out", 0)
    
    # 介数中心性（CB）
    cb = betweenness.get(node_lower, 0.0)
    
    # 等价类（EC）的具体信息
    ec_info = []
    with open(f"./{current_file}/equivalence_classes.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            next_hop = row["Next_Hop"]
            if "_" in next_hop:  # 只处理节点_接口格式
                ec_node = next_hop.split("_")[0].lower()
                if ec_node == node_lower:
                    ec_info.append({
                        "Next_Hop": next_hop,
                        "Networks": row["Networks"]
                    })
    
    # 组合特征向量
    feature_vector = list(adj_vector) + [in_deg, out_deg, cb, ec_info]
    feature_matrix.append(feature_vector)

# 6. 保存为CSV文件
columns = adjacency_df.columns.tolist() + ["d_in", "d_out", "CB", "EC"]
feature_df = pd.DataFrame(feature_matrix, index=adjacency_df.index, columns=columns)
feature_df.to_csv(f"./{current_file}/node_features_H0.csv")

print("特征矩阵已保存为 node_features_H0.csv")

# 7. 以字典形式存储特征矩阵
feature_matrix_dict = {}
for node in nodes:
    node_lower = node.lower()
    
    # 邻接向量（A_i*）
    adj_vector = adjacency_vectors.get(node_lower, [0] * len(nodes))
    
    # 入度（d_in）和出度（d_out）
    in_deg = degrees.get(node_lower, {}).get("in", 0)
    out_deg = degrees.get(node_lower, {}).get("out", 0)
    
    # 介数中心性（CB）
    cb = betweenness.get(node_lower, 0.0)
    
    # 组合特征向量
    feature_matrix_dict[node_lower] = {
        "adjacency_vector": adj_vector,  # 邻接向量
        "in_degree": in_deg,            # 入度
        "out_degree": out_deg,          # 出度
        "betweenness_centrality": cb,   # 介数中心性
        "equivalence_classes": ec_info  # 等价类信息
    }

# ------------------------------
# 8. 输出字典形式特征矩阵
# ------------------------------
# import pprint
# pprint.pprint(feature_matrix_dict)

# ------------------------------
# 9. 保存为JSON文件
# ------------------------------
import json
import numpy as np

# 将 NumPy 数组转换为 Python 列表
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 NumPy 数组转换为列表
    return obj

# 递归处理字典中的 NumPy 数组
def make_json_serializable(data):
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_serializable(item) for item in data]
    else:
        return convert_to_serializable(data)

# 处理特征矩阵字典
serializable_feature_matrix = make_json_serializable(feature_matrix_dict)

# 保存为 JSON 文件
with open(f"./{current_file}/node_features_H0.json", "w") as f:
    json.dump(serializable_feature_matrix, f, indent=2)

print("特征矩阵已保存为 node_features_H0.json")