import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json

# 1. 读取接口数据并生成全局唯一接口标识符
current_file = "Arnes_abs_order_1_72"
interfaces_df = pd.read_csv(f"./{current_file}/{current_file} formatted_interfaces.csv")

# 生成全局唯一接口标识符：设备名_接口名（例如 "ajdovscina_FastEthernet0/0"）
interfaces_df["Unique_Interface"] = (
    interfaces_df["Device"].str.lower() + "_" + interfaces_df["Interface"]
)

# 2. 训练 One-Hot 编码器
unique_interfaces = interfaces_df["Unique_Interface"].unique().reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(unique_interfaces)

# 3. 为每个设备生成接口的 One-Hot 编码向量
# 按设备分组，合并唯一接口列表
device_interfaces = (
    interfaces_df.groupby("Device")["Unique_Interface"]
    .apply(list)
    .to_dict()
)

# 生成编码向量（多接口的向量按位取或）
onehot_features = {}
for device, interfaces in device_interfaces.items():
    encoded = encoder.transform([[interface] for interface in interfaces])
    merged_vector = encoded.any(axis=0).astype(int)
    onehot_features[device] = merged_vector.tolist()

# 4. 保存编码结果
# 保存接口编码
with open(f"./{current_file}/interface_onehot_encoding.json", "w") as f:
    json.dump(onehot_features, f, indent=2)

# 保存编码器类别（可选，用于后续解释）
interface_categories = encoder.categories_[0].tolist()
with open(f"./{current_file}/interface_categories.json", "w") as f:
    json.dump(interface_categories, f, indent=2)

print("接口 One-Hot 编码已保存，且已区分不同设备的同名接口！")
print("编码维度（唯一接口总数）:", len(encoder.categories_[0]))
