import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

class NetworkDataset(Dataset):
    def __init__(self, scenarios):
        self.scenarios = scenarios
        self.N_max = max([len(s["meta"]["original_nodes"]) for s in scenarios])

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        # 图特征
        adj = torch.tensor(scenario["graph_features"]["adjacency"], dtype=torch.float)      # [N_max, N_max]
        in_degree = torch.tensor(scenario["graph_features"]["in_degree"], dtype=torch.float) # [N_max]
        out_degree = torch.tensor(scenario["graph_features"]["out_degree"], dtype=torch.float)
        betweenness = torch.tensor(scenario["graph_features"]["betweenness"], dtype=torch.float)
        ec_counts = torch.tensor(scenario["graph_features"]["ec_counts"], dtype=torch.float)
        
        # 非图特征
        interfaces = torch.tensor(scenario["non_graph_features"]["interfaces"], dtype=torch.float) # [N_max, interface_dim]
        traffic_pairs = torch.tensor(scenario["non_graph_features"]["traffic_pairs"], dtype=torch.long) # [num_pairs, 2]
        
        # 标签
        isolated = torch.tensor(scenario["labels"]["isolated"], dtype=torch.float) # [N_max]
        reachability = torch.tensor(scenario["labels"]["reachability"], dtype=torch.float) # [N_max, N_max]
        high_util = torch.tensor(scenario["labels"]["high_utilization"], dtype=torch.float) # [N_max*N_max]
        load_balancing = torch.tensor(scenario["labels"]["load_balancing"], dtype=torch.float) # [num_pairs]
        
        # 掩码
        mask = torch.tensor(scenario["mask"], dtype=torch.bool) # [N_max]
        
        return {
            "graph": {
                "adj": adj,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "betweenness": betweenness,
                "ec_counts": ec_counts
            },
            "non_graph": {
                "interfaces": interfaces,
                "traffic_pairs": traffic_pairs
            },
            "labels": {
                "isolated": isolated,
                "reachability": reachability,
                "high_util": high_util,
                "load_balancing": load_balancing
            },
            "mask": mask
        }

def collate_fn(batch):
    # 动态填充处理
    return batch  # 由于已提前填充到N_max，直接返回列表

# 示例用法
with open("augmented_dataset_all.json", "r") as f:
    all_scenarios = json.load(f)

dataset = NetworkDataset(all_scenarios)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)