"""
将 inference_5_samples.jsonl 的模型输出解析并转换为验证器输入格式
"""

import sys
import os
import re
import json
import torch
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_network_topology(user_input):
    """从用户输入中解析网络拓扑信息"""
    devices = []
    topology_links = []
    ip_to_device = {}  # IP到设备名称的映射
    
    # 解析设备列表
    device_list_match = re.search(r"\[Device List\]\n(.+?)\n\n", user_input, re.DOTALL)
    if device_list_match:
        device_text = device_list_match.group(1)
        devices = [d.strip() for d in device_text.split(',')]
    
    # 解析拓扑连接
    topology_match = re.search(r"\[Topology Info\]\n(.+?)\n## \[Device Configurations\]", user_input, re.DOTALL)
    if topology_match:
        topology_text = topology_match.group(1)
        for line in topology_text.split('\n'):
            if '↔' in line:
                # 解析连接信息，例如: Ajdovscina (10.0.0.69) ↔ Divaca (10.0.0.68)
                match = re.match(r'(\w+) \(([\d\.]+)\) ↔ (\w+) \(([\d\.]+)\)', line)
                if match:
                    src_device, src_ip, dst_device, dst_ip = match.groups()
                    topology_links.append({
                        'src_device': src_device,
                        'src_ip': src_ip,
                        'dst_device': dst_device,
                        'dst_ip': dst_ip
                    })
                    # 建立IP到设备的映射
                    ip_to_device[src_ip] = src_device
                    ip_to_device[dst_ip] = dst_device
    
    return devices, topology_links, ip_to_device


def build_graph_from_topology(devices, topology_links):
    """根据拓扑信息构建网络图"""
    G = nx.DiGraph()
    
    # 添加节点
    for device in devices:
        G.add_node(device.lower())
    
    # 添加边
    for link in topology_links:
        src = link['src_device'].lower()
        dst = link['dst_device'].lower()
        if src in G.nodes() and dst in G.nodes():
            G.add_edge(src, dst)
            G.add_edge(dst, src)  # 无向图，双向连接
    
    return G


def parse_existing_configs(user_input):
    """解析用户输入中的现有配置"""
    existing_configs = {'lb_rules': {}, 'acl_rules': {}}
    
    # 解析设备配置部分
    config_match = re.search(r"## \[Device Configurations\]\n(.+?)(?=\n\n# === Intent Description ===)", user_input, re.DOTALL)
    if config_match:
        config_text = config_match.group(1)
        
        # 按设备分割
        device_configs = re.split(r'## DEVICE: (\w+)', config_text)[1:]  # 跳过第一个空元素
        
        for i in range(0, len(device_configs), 2):
            if i+1 < len(device_configs):
                device_name = device_configs[i]
                device_config = device_configs[i+1]
                
                # 解析SLB配置
                slb_matches = re.findall(r'Serverfarm SF_\[(.+?)\]:\n(.+?)Virtual-Server VS_\[.+?\]:', device_config, re.DOTALL)
                for flow_key, serverfarm_config in slb_matches:
                    real_matches = re.findall(r'Real ([\d\.]+) with weight (\d+)', serverfarm_config)
                    if real_matches:
                        if device_name not in existing_configs['lb_rules']:
                            existing_configs['lb_rules'][device_name] = {}
                        existing_configs['lb_rules'][device_name][f"[{flow_key}]"] = {
                            'next_hops': [ip for ip, _ in real_matches],
                            'weights': [float(weight)/100.0 for _, weight in real_matches]
                        }
                
                # 解析ACL配置
                acl_matches = re.findall(r'ACL_\[(.+?)\]:\n.+?Permit ([\d\.]+) -> ([\d\.]+)\n.+?Rate Limit: (\d+) Mbps', device_config, re.DOTALL)
                for flow_key, src_ip, dst_ip, rate_limit in acl_matches:
                    if device_name not in existing_configs['acl_rules']:
                        existing_configs['acl_rules'][device_name] = {}
                    existing_configs['acl_rules'][device_name][f"[{src_ip},{dst_ip}]"] = {
                        'rate_limit': float(rate_limit)
                    }
    
    return existing_configs


def create_network_features_like_augmentation(devices, topology_links, policy_configs):
    """参考data_augmentation.py的方法生成网络特征"""
    # 构建网络图
    G = nx.DiGraph()
    
    # 添加节点
    for device in devices:
        G.add_node(device.lower())
    
    # 添加边
    for link in topology_links:
        src = link['src_device'].lower()
        dst = link['dst_device'].lower()
        if src in G.nodes() and dst in G.nodes():
            G.add_edge(src, dst)
            G.add_edge(dst, src)  # 无向图，双向连接
    
    nodes = list(G.nodes())
    N_max = max(75, len(nodes))  # 确保足够的维度
    
    # 创建节点到索引的映射
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 生成图特征（类似data_augmentation.py）
    adjacency_matrix = nx.to_pandas_adjacency(G, nodelist=nodes, dtype=int)
    
    graph_features = {
        "adjacency": adjacency_matrix.values.tolist(),
        "in_degree": [G.in_degree(n) for n in nodes],
        "out_degree": [G.out_degree(n) for n in nodes],
        "betweenness": list(nx.betweenness_centrality(G).values()),
        "ec_counts": [len([]) for n in nodes],  # 简化EC特征
        "ec_networks": [[] for n in nodes]
    }
    
    # 生成流量特征（参考enhance_flows_with_policies）
    traffic_flows = generate_enhanced_traffic_flows(nodes, policy_configs, node_to_idx, idx_to_node)
    
    non_graph_features = {
        "interfaces": [[0]*100 for _ in nodes],  # 简化接口特征
        "traffic_flows": traffic_flows
    }
    
    # 生成标签（简化版本）
    labels = {
        "isolated": [0] * len(nodes),
        "loop": [0] * len(nodes),
        "blackhole": [0] * len(nodes),
        "reachability": [[1 if i==j else 0 for j in range(len(nodes))] for i in range(len(nodes))],
        "high_utilization": [0] * len(nodes),
        "load_balancing": [[0]*len(nodes) for _ in range(len(nodes))],
        "link_overload_pairs": [[0]*len(nodes) for _ in range(len(nodes))],
        "sla_labels": [1] * len(traffic_flows)  # 假设都满足SLA
    }
    
    # 掩码
    mask = [1] * len(nodes) + [0] * (N_max - len(nodes))
    
    return {
        "graph_features": graph_features,
        "non_graph_features": non_graph_features,
        "labels": labels,
        "mask": mask,
        "meta": {
            "original_nodes": nodes,
            "failed_nodes": [],
            "failed_links": []
        }
    }


def generate_enhanced_traffic_flows(nodes, policy_configs, node_to_idx, idx_to_node, num_flows=30):
    """生成增强的流量特征（参考enhance_flows_with_policies）"""
    import random
    traffic_flows = []
    
    # 合并现有配置和生成的配置
    all_lb_rules = policy_configs.get('lb_rules', {})
    all_acl_rules = policy_configs.get('acl_rules', {})
    
    # 收集所有可能的next_hop用于LB向量
    all_next_hops = set()
    for device_rules in all_lb_rules.values():
        for rule in device_rules.values():
            all_next_hops.update(rule.get('next_hops', []))
    
    next_hop_to_idx = {nh: idx for idx, nh in enumerate(sorted(all_next_hops))}
    lb_vector_size = max(5, len(next_hop_to_idx))  # 至少5维
    
    # 生成流量对
    for _ in range(num_flows):
        src_node = random.choice(nodes)
        dst_node = random.choice(nodes)
        
        if src_node != dst_node:
            src_idx = node_to_idx[src_node]
            dst_idx = node_to_idx[dst_node]
            sla_bw = random.randint(10, 200)
            
            # 生成负载均衡向量
            lb_vector = [0.0] * lb_vector_size
            for device, rules in all_lb_rules.items():
                for flow_key, rule in rules.items():
                    # 为对应的next_hop设置权重
                    for i, nh in enumerate(rule.get('next_hops', [])):
                        if nh in next_hop_to_idx:
                            idx = next_hop_to_idx[nh]
                            if idx < len(lb_vector):
                                weights = rule.get('weights', [])
                                if i < len(weights):
                                    lb_vector[idx] = weights[i]
            
            # 生成ACL信息：(节点索引, 限速值)对的列表
            acl_nodes = []
            for device, rules in all_acl_rules.items():
                device_idx = node_to_idx.get(device.lower())
                if device_idx is not None:
                    for flow_key, rule in rules.items():
                        acl_nodes.append((device_idx, rule['rate_limit']))
            
            # 构建增强的流量特征 [src_idx, dst_idx, sla_bw, lb_vector, acl_nodes]
            enhanced_flow = [src_idx, dst_idx, sla_bw, lb_vector, acl_nodes]
            traffic_flows.append(enhanced_flow)
    
    return traffic_flows


def pad_and_mask_like_augmentation(features, N_max=75):
    """参考data_augmentation.py的pad_and_mask函数"""
    original_size = len(features["meta"]["original_nodes"])
    
    # 填充图特征
    for key in ["in_degree", "out_degree", "betweenness", "ec_counts"]:
        features["graph_features"][key] += [0] * (N_max - original_size)
    
    features["graph_features"]["ec_networks"] += [[]] * (N_max - original_size)
    
    # 填充邻接矩阵
    adj_padded = np.zeros((N_max, N_max))
    adj_padded[:original_size, :original_size] = features["graph_features"]["adjacency"]
    features["graph_features"]["adjacency"] = adj_padded.tolist()
    
    # 填充非图特征
    interface_padded = np.zeros((N_max, 100))
    interface_padded[:original_size, :] = features["non_graph_features"]["interfaces"]
    features["non_graph_features"]["interfaces"] = interface_padded.tolist()
    
    # 填充标签
    for key in ["isolated", "loop", "blackhole", "high_utilization"]:
        features["labels"][key] += [0] * (N_max - original_size)
    
    # 填充矩阵标签
    for key in ["reachability", "load_balancing", "link_overload_pairs"]:
        matrix_padded = np.zeros((N_max, N_max))
        matrix_padded[:original_size, :original_size] = features["labels"][key]
        features["labels"][key] = matrix_padded.tolist()
    
    # 更新掩码
    features["mask"] = [1] * original_size + [0] * (N_max - original_size)
    
    return features


def convert_to_samples(inference_path, max_samples=5):
    """
    将推理结果转换为验证器输入格式
    完全基于用户输入和模型输出，参考data_augmentation.py的处理方式
    """
    # 读取推理结果
    with open(inference_path, "r") as f:
        inference_data = [json.loads(line) for line in f]

    samples = []
    for i, inf in enumerate(inference_data[:max_samples]):
        user_input = inf.get('user_input', '')
        model_output = inf.get('model_output', '')
        
        # 1. 解析网络拓扑和现有配置
        devices, topology_links, ip_to_device = parse_network_topology(user_input)
        existing_configs = parse_existing_configs(user_input)
        
        # 2. 解析模型输出的新配置
        generated_configs = parse_model_output_configs(model_output)
        
        # 3. 合并配置
        all_configs = {
            'lb_rules': {**existing_configs.get('lb_rules', {}), **generated_configs.get('lb_rules', {})},
            'acl_rules': {**existing_configs.get('acl_rules', {}), **generated_configs.get('acl_rules', {})}
        }
        
        # 4. 生成网络特征（参考data_augmentation.py）
        features = create_network_features_like_augmentation(devices, topology_links, all_configs)
        
        # 5. 填充和掩码处理
        features = pad_and_mask_like_augmentation(features, N_max=75)
        
        # 6. 转换为tensor格式
        N_max = 75
        sample = {
            "graph": {
                "adj": torch.tensor(features["graph_features"]["adjacency"], dtype=torch.float),
                "in_degree": torch.tensor(features["graph_features"]["in_degree"], dtype=torch.float),
                "out_degree": torch.tensor(features["graph_features"]["out_degree"], dtype=torch.float),
                "betweenness": torch.tensor(features["graph_features"]["betweenness"], dtype=torch.float),
                "ec_counts": torch.tensor(features["graph_features"]["ec_counts"], dtype=torch.float),
                "ec_networks": features["graph_features"]["ec_networks"]
            },
            "non_graph": {
                "traffic_flows": features["non_graph_features"]["traffic_flows"],
                "generated_config": generated_configs,  # 只保留新生成的配置
                "interfaces": torch.tensor(features["non_graph_features"]["interfaces"], dtype=torch.float)
            },
            "labels": {
                "isolated": torch.tensor(features["labels"]["isolated"], dtype=torch.float),
                "loop": torch.tensor(features["labels"]["loop"], dtype=torch.float),
                "blackhole": torch.tensor(features["labels"]["blackhole"], dtype=torch.float),
                "reachability": torch.tensor(features["labels"]["reachability"], dtype=torch.float),
                "link_overload_pairs": torch.tensor(features["labels"]["link_overload_pairs"], dtype=torch.float),
                "sla_labels": torch.tensor(features["labels"]["sla_labels"], dtype=torch.float)
            },
            "mask": torch.tensor(features["mask"], dtype=torch.bool),
            "meta": {
                "scenario_id": i,
                "original_nodes": features["meta"]["original_nodes"],
                "devices": devices,
                "topology_links": topology_links,
                "existing_configs": existing_configs,
                "generated_configs": generated_configs
            }
        }

        samples.append(sample)
        print(f"成功转换样本 {i+1}: {len(devices)} 个设备, {len(features['non_graph_features']['traffic_flows'])} 个流量")
        print(f"  - 现有配置: LB({len(existing_configs.get('lb_rules', {}))}) ACL({len(existing_configs.get('acl_rules', {}))})")
        print(f"  - 生成配置: LB({len(generated_configs.get('lb_rules', {}))}) ACL({len(generated_configs.get('acl_rules', {}))})")

    return samples


def parse_model_output_configs(model_output):
    """解析模型输出的配置信息"""
    try:
        configs = json.loads(model_output)
        parsed_configs = {'lb_rules': {}, 'acl_rules': {}}
        
        for config in configs:
            device = config.get('device', '')
            block_type = config.get('block_type', '')
            
            if block_type == 'slb' and config.get('slb_config'):
                # 解析SLB配置
                slb_config = config['slb_config']
                if device not in parsed_configs['lb_rules']:
                    parsed_configs['lb_rules'][device] = {}
                
                # 构造流量标识符（简化处理）
                flow_key = f"[generated_flow]"
                parsed_configs['lb_rules'][device][flow_key] = {
                    'next_hops': [rs['ip'] for rs in slb_config.get('real_servers', [])],
                    'weights': [float(rs['weight']) / 100.0 for rs in slb_config.get('real_servers', [])]
                }
            
            elif block_type == 'acl' and config.get('acl_rules'):
                # 解析ACL配置
                acl_rules = config['acl_rules']
                if device not in parsed_configs['acl_rules']:
                    parsed_configs['acl_rules'][device] = {}
                
                for rule in acl_rules.get('rules', []):
                    permit = rule.get('permit', '')
                    rate_limit_str = rule.get('rate_limit', '0 Mbps')
                    rate_limit = float(re.findall(r'(\d+)', rate_limit_str)[0]) if re.findall(r'(\d+)', rate_limit_str) else 0
                    
                    # 提取源和目的IP
                    if '->' in permit:
                        src_ip, dst_ip = permit.split(' -> ')
                        flow_key = f"[{src_ip},{dst_ip}]"
                        parsed_configs['acl_rules'][device][flow_key] = {
                            'rate_limit': rate_limit
                        }
        
        return parsed_configs
    except Exception as e:
        print(f"解析模型输出配置失败: {e}")
        return {'lb_rules': {}, 'acl_rules': {}}


def convert_tensors_to_lists(obj):
    """递归地将PyTorch tensor转换为Python list"""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


def custom_serializer(obj):
    """自定义序列化函数，确保特定字段紧凑存储"""
    if isinstance(obj, dict):
        # 对特定字段进行紧凑处理
        compact_fields = {
            "graph": ["adj", "in_degree", "out_degree", 
                     "betweenness", "ec_counts", "ec_networks"],
            "non_graph": ["interfaces", "traffic_flows", "generated_config"],
            "labels": ["isolated", "loop", "blackhole", "reachability", 
                      "high_utilization", "load_balancing", "link_overload_pairs", "sla_labels"],
            "meta": ["original_nodes", "devices", "topology_links", 
                    "existing_configs", "generated_configs"],
        }
        
        for section, fields in compact_fields.items():
            if section in obj:
                for field in fields:
                    if field in obj[section]:
                        # 确保这些字段以紧凑形式存储
                        obj[section][field] = json.dumps(
                            obj[section][field], 
                            separators=(',', ':'),
                            cls=NumpyEncoder
                        )
        
        # 单独处理 mask 字段
        if "mask" in obj:
            obj["mask"] = json.dumps(obj["mask"], separators=(',', ':'))
    
    return obj


class NumpyEncoder(json.JSONEncoder):
    """NumPy数组编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 示例调用
if __name__ == "__main__":
    inference_path = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/inference_5_samples.jsonl"
    
    print("=== 开始转换推理结果为验证器输入格式 ===")
    
    # 检查文件是否存在
    if not os.path.exists(inference_path):
        print(f"错误: 推理结果文件不存在: {inference_path}")
        sys.exit(1)
    
    try:
        samples = convert_to_samples(
            inference_path=inference_path,
            max_samples=5
        )
        
        print(f"成功转换 {len(samples)} 个样本")
        
        # 保存转换后的样本
        output_path = "/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train/output/validator_input_samples.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            # 将tensor转换为list以便JSON序列化
            serializable_samples = []
            for sample in samples:
                serializable_sample = convert_tensors_to_lists(sample)
                serializable_samples.append(serializable_sample)
            
            # 使用紧凑序列化格式保存，类似data_augmentation.py
            f.write("[\n")  # 添加文件开头的中括号
            for i, sample in enumerate(serializable_samples):
                compact_item = json.loads(
                    json.dumps(sample, cls=NumpyEncoder, default=custom_serializer),
                    object_hook=custom_serializer
                )
                json.dump(compact_item, f, indent=2, separators=(',', ': '))
                if i < len(serializable_samples) - 1:
                    f.write(",\n")
            f.write("\n]")
        
        print(f"转换后的验证器输入已保存到: {output_path}")
        
        # 尝试加载验证器并计算奖励
        try:
            sys.path.append('/data/public/6g/USER_YCJ/MyCode/GRPO/GRPO-Intent-Train')
            from reward_fn import compute_reward
            
            print("\n=== 计算验证器奖励 ===")
            for i, s in enumerate(samples):
                try:
                    r = compute_reward(s)
                    print(f"[Sample {i+1}] Reward = {r:.4f}")
                except Exception as e:
                    print(f"[Sample {i+1}] 计算奖励时出错: {e}")
                    
        except ImportError:
            print("警告: 无法导入 reward_fn，跳过奖励计算")
        except Exception as e:
            print(f"计算奖励时出错: {e}")
            
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
