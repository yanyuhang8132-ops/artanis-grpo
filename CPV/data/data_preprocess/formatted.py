import pandas as pd

NETWORKNAME="UsCarrie_abs_simple_2_43"
# 读取接口信息
interfaces = pd.read_csv(NETWORKNAME+" interfaces.csv")

# 提取接口信息
interfaces_formatted = interfaces[["Interface", "Primary_Address", "Description"]].copy()
interfaces_formatted["Device"] = interfaces_formatted["Interface"].apply(lambda x: x.split("[")[0])
interfaces_formatted["Interface"] = interfaces_formatted["Interface"].apply(lambda x: x.split("[")[1].rstrip("]"))
interfaces_formatted["IP Address"] = interfaces_formatted["Primary_Address"].apply(lambda x: x.split("/")[0])
interfaces_formatted["Subnet Mask"] = interfaces_formatted["Primary_Address"].apply(lambda x: "/" + x.split("/")[1])

# 保存格式化后的接口信息
interfaces_formatted[["Device", "Interface", "IP Address", "Subnet Mask", "Description"]].to_csv(NETWORKNAME+" formatted_interfaces.csv", index=False)
print("格式化后的接口信息已保存到 formatted_interfaces.csv")

# 读取转发表信息
routes = pd.read_csv(NETWORKNAME+" routes.csv")

# 提取转发表信息
routes_formatted = routes[["Node", "Network", "Next_Hop", "Protocol"]].copy()
routes_formatted["Next_Hop"] = routes_formatted["Next_Hop"].apply(lambda x: x.replace("interface ", "").split(" ")[0])

# 保存格式化后的转发表信息
routes_formatted.to_csv(NETWORKNAME+" formatted_routes.csv", index=False)
print("格式化后的转发表信息已保存到 formatted_routes.csv")