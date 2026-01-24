import numpy as np
import networkx as nx

# 读取邻接矩阵文件
current_file = "UsCarrie_abs_simple_2_43"
adjacency_matrix_file = f"./{current_file}/adjacency_matrix.csv"

# 读取文件内容
with open(adjacency_matrix_file, "r") as f:
    lines = f.readlines()

# 提取设备名称（第一行和第一列）
header = lines[0].strip().split(",")[1:]  # 第一行，去掉第一个空值
devices = [line.strip().split(",")[0] for line in lines[1:]]  # 第一列

# 检查设备名称是否一致
if header != devices:
    print("警告：第一行和第一列的设备名称不一致！")
    print("第一行设备名称:", header)
    print("第一列设备名称:", devices)
    devices = header  # 默认使用第一行的设备名称

# 提取数值部分（跳过第一行和第一列）
adjacency_matrix = np.array(
    [line.strip().split(",")[1:] for line in lines[1:]], dtype=int
)

# 创建有向图
G = nx.DiGraph()

# 添加节点和边
for i in range(len(devices)):
    for j in range(len(devices)):
        if adjacency_matrix[i, j] == 1:  # 如果存在边
            G.add_edge(devices[i], devices[j])

# 计算介数中心性
betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

# 打印结果
print("节点介数中心性：")
for node, centrality in betweenness_centrality.items():
    print(f"设备: {node}, 介数中心性: {centrality:.4f}")

# 计算介数中心性的平均值
mean_betweenness_centrality = np.mean(list(betweenness_centrality.values()))
print(f"平均介数中心性: {mean_betweenness_centrality:.4f}")

# 保存结果到文件
output_file = f"./{current_file}/betweenness_centrality_output.txt"
with open(output_file, "w") as f:
    f.write("节点介数中心性：\n")
    for node, centrality in betweenness_centrality.items():
        f.write(f"设备: {node}, 介数中心性: {centrality:.4f}\n")