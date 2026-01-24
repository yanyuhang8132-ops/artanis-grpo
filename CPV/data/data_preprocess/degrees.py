import numpy as np
import pandas as pd

# 读取接口信息文件
current_file = "UsCarrie_abs_simple_2_43"
interfaces_file = f"./{current_file}/{current_file} formatted_interfaces.csv"
df = pd.read_csv(interfaces_file, sep=",")

# 提取所有唯一的设备名称
devices = df["Device"].unique()

# 对设备名称进行排序
devices = sorted(devices)

# 生成设备名称与索引的映射
device_index_mapping = [f"Index {i}: {device.capitalize()}" for i, device in enumerate(devices)]

# 保存映射到文件
output_file = f"./{current_file}/device_index_mapping.txt"
with open(output_file, "w") as f:
    for line in device_index_mapping:
        f.write(line + "\n")

print(f"已生成设备名称与索引的映射文件: {output_file}")

# 读取邻接矩阵文件
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

# 计算每个节点的入度和出度
num_nodes = adjacency_matrix.shape[0]
in_degrees = np.sum(adjacency_matrix, axis=0)  # 入度：列求和
out_degrees = np.sum(adjacency_matrix, axis=1)  # 出度：行求和

# 打印结果
print("节点入度和出度：")
for i in range(num_nodes):
    print(f"设备: {devices[i]}")
    print(f"  入度: {int(in_degrees[i])}")
    print(f"  出度: {int(out_degrees[i])}")

# 保存结果到文件
output_file = f"./{current_file}/degrees_output.txt"
with open(output_file, "w") as f:
    f.write("节点入度和出度：\n")
    for i in range(num_nodes):
        f.write(f"设备: {devices[i]}\n")
        f.write(f"  入度: {int(in_degrees[i])}\n")
        f.write(f"  出度: {int(out_degrees[i])}\n")