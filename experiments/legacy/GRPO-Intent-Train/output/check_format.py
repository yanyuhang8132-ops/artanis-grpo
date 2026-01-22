#!/usr/bin/env python3
import json
import re

# 读取1.jsonl的第一个样本来检查格式
with open("1.jsonl", 'r') as f:
    first_sample = json.loads(f.readline())

user_input = first_sample["user_input"]

# 检查网络拓扑格式
topology_match = re.search(r"\[Topology Info\]\n(.+?)\n\#\#", user_input, re.DOTALL)
if topology_match:
    topology_str = topology_match.group(1)
    print("拓扑信息样例：")
    print(topology_str.split('\n')[0])  # 打印第一行
    print(topology_str.split('\n')[1])  # 打印第二行
    
    # 测试正则表达式
    line = topology_str.split('\n')[0]
    match = re.match(r"(.+?) \((?:[\w/]+ )?([\d.]+)\) ↔ (.+?) \((?:[\w/]+ )?([\d.]+)\)(?: \[(\d+\.?\d*) Mbps\])?", line.strip())
    if match:
        print("✅ 正则表达式匹配成功")
        print(f"节点A: {match.group(1)}, IP_A: {match.group(2)}")
        print(f"节点B: {match.group(3)}, IP_B: {match.group(4)}")
        print(f"带宽: {match.group(5)}")
    else:
        print("❌ 正则表达式匹配失败")

# 检查SLA意图格式
sla_match = re.search(r"Deploy a service flow from (.+?) to (.+?), with SLA guaranteeing a minimum bandwidth of (\d+) Mbps", user_input)
if sla_match:
    print("✅ SLA意图匹配成功")
    print(f"源: {sla_match.group(1)}, 目标: {sla_match.group(2)}, SLA: {sla_match.group(3)} Mbps")
else:
    print("❌ SLA意图匹配失败")