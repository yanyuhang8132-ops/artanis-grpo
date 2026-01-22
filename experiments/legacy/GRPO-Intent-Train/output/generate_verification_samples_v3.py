import json
import numpy as np
import pandas as pd
import torch
import networkx as nx
import re
from ast import literal_eval
from extract_matrices_v2 import (
    parse_topology,
    parse_device_configurations,
    parse_model_output,
    standardize_model_middleboxes,
    compare_rules,
    simulate_acl_effect,
    simulate_slb_effect,
)
from utils.Equivalence_Class import generate_equivalence_classes
from utils.data_augmentation import generate_labels

def parse_user_input_fields(user_input):
    device_match = re.search(r"## \[Device List\]\n(.+?)\n\n", user_input, re.DOTALL)
    topology_match = re.search(r"## \[Topology Info\]\n(.+?)\n##", user_input, re.DOTALL)
    config_match = re.search(r"## \[Device Configurations\]\n(.+?)\n##", user_input, re.DOTALL)
    devices = [d.strip() for d in device_match.group(1).split(",")] if device_match else []
    topology = topology_match.group(1).strip() if topology_match else ""
    configs = config_match.group(1).strip() if config_match else ""
    return devices, topology, configs

def generate_flows_from_policies(name_to_idx, acl_rules, slb_rules):
    flow_map = {}
    for rule in acl_rules:
        key = (rule['src_ip'], rule['dst_ip'])
        flow_map.setdefault(key, {})['acl'] = rule
    for rule in slb_rules:
        key = (rule['src_node'], rule['dst_node'])
        flow_map.setdefault(key, {})['slb'] = rule

    slb_devices = [r["device"] for r in slb_rules]
    slb_device_index = {dev: i for i, dev in enumerate(slb_devices)}
    lb_vector_len = len(slb_devices) * 2
    flows = []

    for (src_ip, dst_ip), content in flow_map.items():
        if src_ip not in name_to_idx or dst_ip not in name_to_idx:
            continue
        src_id = name_to_idx[src_ip]
        dst_id = name_to_idx[dst_ip]
        sla = np.random.randint(20, 100)

        lb_vector = [0.0] * lb_vector_len
        acl_entries = []
        if "acl" in content:
            acl_node = name_to_idx.get(content["acl"]["device"])
            if acl_node is not None:
                acl_entries.append([acl_node, sla])

        if "slb" in content:
            slb_rule = content["slb"]
            dev = slb_rule["device"]
            base = slb_device_index[dev] * 2
            weights = [s["weight"] for s in slb_rule["real_servers"]]
            total = sum(weights)
            if total > 0:
                lb_vector[base] = round(weights[0] / total, 2)
                if len(weights) > 1:
                    lb_vector[base+1] = round(weights[1] / total, 2)

        flows.append([src_id, dst_id, sla, lb_vector, acl_entries])
    return flows

def generate_sample(entry, sample_id, routes_df):
    user_input = entry["user_input"]
    model_output = literal_eval(entry["model_output"])
    device_names, topology_str, config_text = parse_user_input_fields(user_input)

    adj_matrix, name_to_idx, ip_to_node = parse_topology(topology_str, device_names)
    existing_acls, existing_slbs = parse_device_configurations(config_text)
    model_acls_raw, model_slbs_raw = standardize_model_middleboxes(model_output)
    new_acls, new_slbs = compare_rules(model_acls_raw, model_slbs_raw, existing_acls, existing_slbs)

    simulate_acl_effect(adj_matrix, name_to_idx, new_acls, ip_to_node)
    simulate_slb_effect(adj_matrix, name_to_idx, new_slbs, ip_to_node)

    G = nx.DiGraph()
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=1)

    N = len(device_names)
    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    in_degree = np.zeros(N)
    out_degree = np.zeros(N)
    betw = np.zeros(N)
    for i in range(N):
        in_degree[i] = out_degree[i] = degrees.get(i, 0)
        betw[i] = betweenness.get(i, 0)

    forwarding_graphs, ec_networks = generate_equivalence_classes(G, routes_df)
    ec_counts = [len(nets) for nets in ec_networks]

    traffic_flows = generate_flows_from_policies(name_to_idx, new_acls, new_slbs)
    labels = generate_labels(G, traffic_flows, forwarding_graphs_override=forwarding_graphs)

    sample = {
        "graph_features": {
            "adjacency": adj_matrix.tolist(),
            "in_degree": in_degree.tolist(),
            "out_degree": out_degree.tolist(),
            "betweenness": betw.tolist(),
            "ec_counts": ec_counts,
            "ec_networks": ec_networks
        },
        "non_graph_features": {
            "traffic_flows": traffic_flows
        },
        "labels": labels,
        "mask": [1] * N
    }
    return sample

def main():
    with open("inference_5_samples.jsonl", "r") as fin:
        lines = [json.loads(line) for line in fin]
    routes_df = pd.read_csv("routes.csv")
    results = []
    for i, entry in enumerate(lines):
        sample = generate_sample(entry, i, routes_df)
        results.append(sample)
    with open("verification_samples.jsonl", "w") as fout:
        for item in results:
            fout.write(json.dumps(item) + "\\n")

if __name__ == "__main__":
    main()
