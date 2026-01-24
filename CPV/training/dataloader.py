import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gcn_module import GCNModule
from models.fcn_module import FCNModule
from models.multimodal_model import MultimodalMultitaskModel
import ast

class NetworkDataset(Dataset):
    def __init__(self, json_path, N_max=None):
        with open(json_path, "r") as f:
            self.scenarios = json.load(f)
        
        for s in self.scenarios:
            if isinstance(s["meta"]["original_nodes"], str):
                try:
                    # 尝试解析为真正的列表
                    s["meta"]["original_nodes"] = ast.literal_eval(s["meta"]["original_nodes"])
                except Exception as e:
                    print(f"[Error] original_nodes 格式错误：{s['meta']['original_nodes']}")
                    raise e

        node_counts = [len(s["meta"]["original_nodes"]) for s in self.scenarios]
        self.N_max = N_max if N_max is not None else max(node_counts)

        print(f"Loaded {len(self.scenarios)} scenarios with N_max={self.N_max}")

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]

        # 图结构特征
        adj = torch.tensor(parse_if_string(scenario["graph_features"]["adjacency"]), dtype=torch.float)
        in_degree = torch.tensor(parse_if_string(scenario["graph_features"]["in_degree"]), dtype=torch.float)
        out_degree = torch.tensor(parse_if_string(scenario["graph_features"]["out_degree"]), dtype=torch.float)
        betweenness = torch.tensor(parse_if_string(scenario["graph_features"]["betweenness"]), dtype=torch.float)
        ec_counts = torch.tensor(parse_if_string(scenario["graph_features"]["ec_counts"]), dtype=torch.float)
        ec_networks = encode_ec_networks(scenario["graph_features"]["ec_networks"], self.N_max)

        # 非图结构
        # interfaces = torch.tensor(parse_if_string(scenario["non_graph_features"]["interfaces"]), dtype=torch.float)
        flow_features, traffic_pairs = encode_traffic_flows(scenario["non_graph_features"]["traffic_flows"], num_links=100)

        # 标签
        isolated = torch.tensor(parse_if_string(scenario["labels"]["isolated"]), dtype=torch.float)
        loop = torch.tensor(parse_if_string(scenario["labels"]["loop"]), dtype=torch.float)
        blackhole = torch.tensor(parse_if_string(scenario["labels"]["blackhole"]), dtype=torch.float)
        reachability = torch.tensor(parse_if_string(scenario["labels"]["reachability"]), dtype=torch.float)
        # high_util = torch.tensor(parse_if_string(scenario["labels"]["high_utilization"]), dtype=torch.float)
        # load_balancing = torch.tensor(parse_if_string(scenario["labels"]["load_balancing"]), dtype=torch.float)
        link_overload_pairs = torch.tensor(parse_if_string(scenario["labels"]["link_overload_pairs"]), dtype=torch.float)  # [N_max, N_max]
        sla_labels = torch.tensor(parse_if_string(scenario["labels"]["sla_labels"]), dtype=torch.float)  # [num_flows]

        # 掩码
        mask = torch.tensor(parse_if_string(scenario["mask"]), dtype=torch.bool)

        # Padding 到 N_max
        isolated = F.pad(isolated, (0, self.N_max - isolated.shape[0]), value=0)
        loop = F.pad(loop, (0, self.N_max - loop.shape[0]), value=0)
        blackhole = F.pad(blackhole, (0, self.N_max - blackhole.shape[0]), value=0)
        reachability = F.pad(reachability, (0, self.N_max - reachability.shape[0], 0, self.N_max - reachability.shape[1]), value=0)
        link_overload_pairs = F.pad(link_overload_pairs, (0, self.N_max - link_overload_pairs.shape[0],
                                                  0, self.N_max - link_overload_pairs.shape[1]), value=0)
        # high_util = F.pad(high_util, (0, self.N_max - high_util.shape[0]), value=0)
        # load_balancing = F.pad(load_balancing, (0, self.N_max - load_balancing.shape[0], 0, self.N_max - load_balancing.shape[1]), value=0)

        return {
            "graph": {
                "adj": adj,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "betweenness": betweenness,
                "ec_counts": ec_counts,
                "ec_networks": ec_networks
            },
            "non_graph": {
                "traffic_pairs": traffic_pairs,
                "traffic_flows": flow_features
            },
            "labels": {
                "isolated": isolated,
                "loop": loop,
                "blackhole": blackhole,
                "reachability": reachability,
                # "high_util": high_util,
                # "load_balancing": load_balancing,
                "link_overload_pairs": link_overload_pairs,
                "sla_labels": sla_labels
            },
            "mask": mask
        }

def parse_if_string(x):
    return json.loads(x) if isinstance(x, str) else x

def collate_fn(batch):
    """
    自定义批处理函数
    由于每个场景的流量对数量不同，直接返回列表
    """
    return batch

def get_dataloader(json_path, batch_size=32, shuffle=True):
    dataset = NetworkDataset(json_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return dataloader

def get_test_dataloader(json_path, N_max, batch_size=32, shuffle=False):
    dataset = NetworkDataset(json_path, N_max=N_max)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return dataloader

import hashlib

def hash_prefix(s, vocab_size=10000):
    """将IP前缀或fake_id哈希为整数索引"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % vocab_size

def encode_ec_networks(ec_networks_raw, N_max, K_max=10, vocab_size=10000):
    """将每个节点的前缀集合编码为长度为 K_max 的整数列表"""
    ec_networks_list = json.loads(ec_networks_raw)  # 转成list[list[str]]
    encoded = []

    for i in range(N_max):
        if i < len(ec_networks_list):
            ec_set = ec_networks_list[i]
            h = [hash_prefix(p, vocab_size) for p in ec_set[:K_max]]  # 截断为最多 K_max 个
            h += [0] * (K_max - len(h))  # 不足补零
        else:
            h = [0] * K_max  # 节点不足 N_max 补零
        encoded.append(h)

    return torch.tensor(encoded, dtype=torch.long)  # shape [N_max, K_max]

def encode_acl_rules(acl_rules_raw, num_links=100):
    acl_tensor = []
    for rule_list in acl_rules_raw:
        vec = [0.0] * num_links
        for mid_id, bw in rule_list:
            if 0 <= mid_id < num_links:
                vec[mid_id] = bw
        acl_tensor.append(vec)
    return torch.tensor(acl_tensor, dtype=torch.float)

def encode_traffic_flows(traffic_flows_raw, num_links=100):
    flows = json.loads(traffic_flows_raw) if isinstance(traffic_flows_raw, str) else traffic_flows_raw
    src_dst_sla = []
    lb_vecs = []
    acl_rules = []

    for f in flows:
        src_dst_sla.append([f[0], f[1], f[2]])
        lb_vecs.append(f[3])
        acl_rules.append(f[4])

    src_dst_sla = torch.tensor(src_dst_sla, dtype=torch.float)
    lb_vecs = torch.tensor(lb_vecs, dtype=torch.float)
    acl_tensor = encode_acl_rules(acl_rules, num_links)

    flow_features = torch.cat([src_dst_sla, lb_vecs, acl_tensor], dim=1)  # [num_flows, D]
    traffic_pairs = src_dst_sla[:, :2].long()
    return flow_features, traffic_pairs

# 测试数据加载
if __name__ == "__main__":
    dataloader = get_dataloader("data/augmented_dataset_all.json", batch_size=32)
    test_dataloader = get_test_dataloader("data/test_dataset_all.json", N_max=dataloader.dataset.N_max, batch_size=32)
    for batch in dataloader:
        print("Batch size:", len(batch))  # 应该为 batch_size
        print("Graph features:", batch[0]["graph"].keys())
        print("Non-graph features:", batch[0]["non_graph"].keys())
        print("Labels:", batch[0]["labels"].keys())
        break

    # batch = next(iter(dataloader))

    # # 测试 FCN 模块
    # #N_max 和interface_dim从数据集中获取
    # N_max = dataloader.dataset.N_max
    # interface_dim = len(dataloader.dataset.scenarios[0]["non_graph_features"]["interfaces"][0])
    # print("N_max:", N_max)
    # print("interface_dim:", interface_dim)
    # hidden_dim = 16
    
    # fcn = FCNModule(interface_dim=interface_dim, hidden_dim=hidden_dim, N_max=N_max)
    # non_graph_data = batch[0]["non_graph"]
    # interfaces = non_graph_data["interfaces"]
    # traffic_pairs = non_graph_data["traffic_pairs"]
    # fcn_out = fcn(interfaces, traffic_pairs)
    # print("FCN output shape:", fcn_out.shape)  # 应该为 [hidden_dim]

    
    # # 测试 GCN 模块
    # gcn = GCNModule(ec_vocab_size=10000, ec_embed_dim=8, hidden_dim=16, output_dim=35)
    # graph_data = batch[0]["graph"]

    # node_features = gcn.build_node_features(graph_data)
    # G = gcn(graph_data["adj"], node_features)

    # print("GCN output shape:", G.shape)

    # 测试 Attention 模块
    # from models.attention_module import AttentionFCNModule

    # N_max = dataloader.dataset.N_max

    # non_graph_data = batch[0]["non_graph"]
    # interfaces = non_graph_data["interfaces"]                  # [N_max, interface_dim]
    # flow_features = non_graph_data["traffic_flows"]            # [num_flows, D]

    # flow_input_dim = flow_features.shape[1]  # 自动获取 D

    # attention_fcn = AttentionFCNModule(
    #     interface_dim=interfaces.shape[1],
    #     flow_input_dim=flow_input_dim,
    #     hidden_dim=16,
    #     N_max=N_max,
    #     num_heads=4
    # )

    # attention_out = attention_fcn(interfaces, flow_features)
    # print("Attention FCN output shape:", attention_out.shape)


    # 测试多模态模型

    # 获取 N_max 和 interface_dim
    N_max = dataloader.dataset.N_max

    flow_input_dim = batch[0]["non_graph"]["traffic_flows"].shape[1]
    print("flow_input_dim:", flow_input_dim)
    model = MultimodalMultitaskModel(
        N_max=N_max,
        flow_input_dim=flow_input_dim,
        ec_vocab_size=10000,
        ec_embed_dim=8,
        hidden_dim=16
    )

    output = model(batch[0]["graph"], batch[0]["non_graph"])
    # 打印输出形状时，忽略 batch 维度
    for task, tensor in output.items():
        print(f"{task} shape:", tensor.squeeze(0).shape)  # 移除 batch 维度
