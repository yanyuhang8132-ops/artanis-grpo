import csv
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from data_augmentation import generate_test_scenario, pad_and_mask, NumpyEncoder
from tqdm import tqdm  # 添加进度条库

def main():
    topologies = {
        "Arnes": {"adjacency": "./Arnes_abs_order_1_72/adjacency_matrix.csv", 
                  "interfaces": "./Arnes_abs_order_1_72/interface_onehot_encoding.json",
                  "traffic": "./Arnes_abs_order_1_72/traffic_pairs.json",
                  "equivalence_classes": "./Arnes_abs_order_1_72/equivalence_classes.csv"},
        # "Uninett": {"adjacency": "./Uninett2011_abs_order_1_225/adjacency_matrix.csv",
        #          "interfaces": "./Uninett2011_abs_order_1_225/interface_onehot_encoding.json",
        #          "traffic": "./Uninett2011_abs_order_1_225/traffic_pairs.json",
        #          "equivalence_classes": "./Uninett2011_abs_order_1_225/equivalence_classes.csv"}
    }

    # 确定全局最大节点数和最大接口数
    N_max = max([pd.read_csv(v["adjacency"], index_col=0).shape[0] for v in topologies.values()])
    M_max = max([len(next(iter(json.load(open(v["interfaces"])).values()))) for v in topologies.values()])

    test_scenarios = []
    for topology_name, paths in tqdm(topologies.items(), desc="生成测试场景"):  # 添加进度条
        # 加载数据
        adjacency = pd.read_csv(paths["adjacency"], index_col=0).rename(columns=str.lower, index=str.lower)
        G = nx.from_pandas_adjacency(adjacency, create_using=nx.DiGraph)
        
        with open(paths["interfaces"], "r") as f:
            interface_encoding = json.load(f)
        
        with open(paths["traffic"], "r") as f:
            traffic_pairs = json.load(f)
        
        ec_features = defaultdict(list)
        with open(paths["equivalence_classes"], "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                next_hop = row["Next_Hop"]
                networks = row["Networks"].split("|")
                if "_" in next_hop:  # 只处理节点_接口格式
                    node = next_hop.split("_")[0].lower()
                    ec_features[node].extend(networks)

        # 生成测试场景
        test_scenario = generate_test_scenario(G, interface_encoding, traffic_pairs, scenario_id=0, ec_features=ec_features)
        test_scenarios.append(test_scenario)

    # 填充和添加掩码
    padded_test_scenarios = pad_and_mask(test_scenarios, N_max, M_max)

    # 保存测试集
    with open("test_dataset_general.json", "w") as f:
        json.dump(padded_test_scenarios, f, indent=2, cls=NumpyEncoder)
    print(f"测试集已保存，共 {len(padded_test_scenarios)} 个场景")

if __name__ == "__main__":
    main()
