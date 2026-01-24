import pandas as pd
import networkx as nx
import os

# 定义文件路径
input_folder = "./UsCarrie_abs_simple_2_43/"
output_folder = input_folder  # 输出文件夹与输入文件夹相同

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 读取接口信息文件
interfaces_file = os.path.join(input_folder, "UsCarrie_abs_simple_2_43 formatted_interfaces.csv")

# 创建有向图
G = nx.DiGraph()

# 使用pandas读取CSV文件
df = pd.read_csv(interfaces_file, sep=',')  # 使用逗号分隔

# 遍历每一行数据
for index, row in df.iterrows():
    device = row['Device']  # 源设备名称
    description = row['Description']  # 接口描述
    
    # 统一转换为大写首字母
    device = device.capitalize()  # 例如，ajdovscina -> Ajdovscina
    
    # 提取目标设备名称
    if "To " in description:
        dst = description.split("To ")[1].strip().strip('"')  # 提取目标设备名称，例如 "Divaca"
        dst = dst.capitalize()  # 统一转换为大写首字母
        
        # 添加有向边（设备到目标设备）
        if device and dst:
            G.add_edge(device, dst)

# 获取所有设备的排序列
nodes = sorted(G.nodes())

# 生成邻接矩阵
adjacency_matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()

# 打印邻接矩阵
print("邻接矩阵：")
print(adjacency_matrix)

# 打印设备名称与邻接矩阵的对应关系
print("\n设备名称与邻接矩阵的对应关系：")
for i, node in enumerate(nodes):
    print(f"Index {i}: {node}")

# 保存邻接矩阵为CSV文件（方便后续使用）
output_file = os.path.join(output_folder, "adjacency_matrix.csv")
pd.DataFrame(adjacency_matrix, index=nodes, columns=nodes).to_csv(output_file)