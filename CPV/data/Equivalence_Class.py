import pandas as pd
import networkx as nx
import numpy as np
import pickle
import time

# 开始计时
start_time = time.time()

# 读取路由信息文件
print("正在读取路由信息文件...")
current_file = "Arnes_abs_order_1_72"
routes_file = f"./{current_file}/{current_file} routes.csv"
routes_df = pd.read_csv(routes_file)
print(f"路由信息读取完成，耗时: {time.time() - start_time:.2f}秒")

# 读取接口信息文件
step_start = time.time()
print("正在读取接口信息文件...")
interfaces_file = f"./{current_file}/{current_file} formatted_interfaces.csv"
interfaces_df = pd.read_csv(interfaces_file)
print(f"接口信息读取完成，耗时: {time.time() - step_start:.2f}秒")

# 1: 构建接口到对端设备的映射
step_start = time.time()
print("正在构建接口到对端设备的映射...")
interface_to_peer = {}
for _, row in interfaces_df.iterrows():
    device = row["Device"]
    interface = row["Interface"]
    description = row["Description"].strip('"')  # 去除双引号
    peer_device = description.split("To ")[-1].replace("TT", " ").strip()  # 处理特殊字符
    interface_key = f"{device}_{interface}"
    interface_to_peer[interface_key] = peer_device
print(f"接口映射构建完成，耗时: {time.time() - step_start:.2f}秒")

# 2: 增强IP到设备的映射
step_start = time.time()
print("正在构建IP到设备的映射...")
ip_to_device = {}
for _, row in interfaces_df.iterrows():
    device = row["Device"]
    ip_address = row["IP Address"].split("/")[0]
    ip_to_device[ip_address] = device
print(f"IP映射构建完成，耗时: {time.time() - step_start:.2f}秒")

# 读取邻接矩阵文件
step_start = time.time()
print("正在读取邻接矩阵文件...")
adjacency_matrix_file = f"./{current_file}/adjacency_matrix.csv"
adjacency_df = pd.read_csv(adjacency_matrix_file, index_col=0)
devices = adjacency_df.index.tolist()
adjacency_matrix = adjacency_df.values
print(f"邻接矩阵读取完成，耗时: {time.time() - step_start:.2f}秒")

# 创建有向图
step_start = time.time()
print("正在创建有向图...")
G = nx.DiGraph()
G.add_nodes_from(devices)

# 添加边（基于邻接矩阵）
for i, src in enumerate(devices):
    for j, dst in enumerate(devices):
        if adjacency_matrix[i, j] == 1:
            G.add_edge(src, dst)
print(f"有向图创建完成，耗时: {time.time() - step_start:.2f}秒")

# 3: 增强等价类构建
step_start = time.time()
print("正在构建等价类...")
def build_equivalence_classes(routes_df):
    equivalence_classes = {}
    for _, row in routes_df.iterrows():
        network = row["Network"]
        next_hop = row["Next_Hop"]
        node = row["Node"]
        
        # 处理接口类型的Next_Hop
        if next_hop.startswith("interface"):
            interface_name = next_hop.split()[-1]
            next_hop_key = f"{node}_{interface_name}"  # 唯一标识符
        else:
            next_hop_key = next_hop  # 其他类型直接使用
            
        if next_hop_key not in equivalence_classes:
            equivalence_classes[next_hop_key] = set()
        equivalence_classes[next_hop_key].add(network)
    
    # 转换为列表
    return {k: list(v) for k, v in equivalence_classes.items()}

equivalence_classes = build_equivalence_classes(routes_df)
print(f"等价类构建完成，耗时: {time.time() - step_start:.2f}秒")

# 4: 增强转发图生成
step_start = time.time()
print("正在构建转发图...")
def build_forwarding_graphs(G, equivalence_classes, routes_df):
    forwarding_graphs = {}
    
    for eq_key, networks in equivalence_classes.items():
        fg = nx.DiGraph()
        
        # 解析等价类类型
        if "_" in eq_key:  # 接口类型
            src_device, interface = eq_key.split("_", 1)
            peer_device = interface_to_peer.get(eq_key, None)
            
            if peer_device and peer_device in G:
                fg.add_edge(src_device, peer_device)
                
        else:  # IP或其他类型
            # 获取所有相关路由条目
            relevant_routes = routes_df[routes_df["Next_Hop"] == eq_key]
            
            for _, row in relevant_routes.iterrows():
                src = row["Node"]
                if eq_key.startswith("ip"):
                    next_hop_ip = eq_key.split()[-1]
                    dst = ip_to_device.get(next_hop_ip, None)
                else:
                    dst = eq_key  # 其他情况直接使用
                
                if dst and dst in G:
                    fg.add_edge(src, dst)
        
        forwarding_graphs[eq_key] = fg
    
    return forwarding_graphs

forwarding_graphs = build_forwarding_graphs(G, equivalence_classes, routes_df)
print(f"转发图构建完成，耗时: {time.time() - step_start:.2f}秒")

# ================== 结果保存 ==================
step_start = time.time()
print("正在保存结果文件...")
# 保存等价类
equivalence_classes_df = pd.DataFrame({
    "Next_Hop": equivalence_classes.keys(),
    "Networks": ["|".join(nets) for nets in equivalence_classes.values()]
})
equivalence_classes_df.to_csv(f"./{current_file}/equivalence_classes.csv", index=False)

# 保存转发图
forwarding_edges = []
for eq_key, fg in forwarding_graphs.items():
    for (u, v) in fg.edges():
        forwarding_edges.append({
            "Equivalence_Class": eq_key,
            "Source": u,
            "Destination": v
        })
pd.DataFrame(forwarding_edges).to_csv(f"./{current_file}/forwarding_graphs.csv", index=False)

# 新增：保存forwarding_graphs为pickle，供后续数据标签生成用
with open(f"./{current_file}/forwarding_graphs.pkl", "wb") as f:
    pickle.dump(forwarding_graphs, f)
print(f"结果保存完成，耗时: {time.time() - step_start:.2f}秒")

print(f"\n总计耗时: {time.time() - start_time:.2f}秒")
print("处理完成！生成的等价类包含以下条目：")
print(equivalence_classes_df.head())