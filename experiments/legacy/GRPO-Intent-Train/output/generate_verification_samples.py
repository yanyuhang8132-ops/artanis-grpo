
# generate_verification_samples.py

import json
import numpy as np
import torch
import networkx as nx
from extract_matrices_v2 import parse_topology, parse_device_configurations, parse_model_output, standardize_model_middleboxes, compare_rules, simulate_acl_effect, simulate_slb_effect
from utils.Equivalence_Class import generate_equivalence_classes
from utils.data_augmentation import generate_labels

def generate_traffic_flows(G, name_to_idx, acl_map, slb_map, num_flows=10, max_sla=100):
    nodes = list(name_to_idx.keys())
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    traffic_flows = []
    traffic_pairs = []

    for _ in range(num_flows):
        src, dst = np.random.choice(nodes, 2, replace=False)
        src_id = name_to_idx[src]
        dst_id = name_to_idx[dst]
        sla = np.random.randint(20, max_sla)

        lb_vector = [0.0] * len(slb_map)
        acl_rules = []

        try:
            path = nx.shortest_path(G, source=src_id, target=dst_id)
        except:
            continue

        for node_id in path:
            dev = idx_to_name[node_id]
            if dev in slb_map:
                lb_index = slb_map[dev]
                lb_vector[lb_index] = 1.0  # 假设默认经过该LB分流
            if dev in acl_map:
                rate = acl_map[dev]
                acl_rules.append([node_id, rate])

        traffic_flows.append([src_id, dst_id, sla, lb_vector, acl_rules])
        traffic_pairs.append([src_id, dst_id])

    return traffic_flows, traffic_pairs
