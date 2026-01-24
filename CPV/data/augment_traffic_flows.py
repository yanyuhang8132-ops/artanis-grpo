
import json
import random

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except:
            pass
        return super().default(obj)

def custom_serializer(obj):
    """自定义序列化函数，确保特定字段紧凑存储"""
    if isinstance(obj, dict):
        compact_fields = {
            "non_graph_features": ["traffic_flows"]
        }

        for section, fields in compact_fields.items():
            if section in obj:
                for field in fields:
                    if field in obj[section]:
                        obj[section][field] = json.dumps(
                            obj[section][field],
                            separators=(',', ':'),
                            cls=NumpyEncoder
                        )
    return obj

def augment_traffic_pairs(input_file, output_file, flow_summary_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    flow_summary = []

    for scenario in data:
        raw_pairs = scenario.get("non_graph_features", {}).get("traffic_pairs", "[]")
        if isinstance(raw_pairs, str):
            try:
                traffic_pairs = json.loads(raw_pairs)
            except:
                traffic_pairs = []
        else:
            traffic_pairs = raw_pairs

        new_flows = []
        scenario_flows = []
        scenario_id = scenario.get("meta", {}).get("scenario_id", -1)

        for i, (src, dst) in enumerate(traffic_pairs):
            dscp = random.randint(0, 7)
            sla_bw = random.randint(30, 100)
            new_flows.append([src, dst, dscp, sla_bw])
            flow_record = {
                "flow_id": f"F{i+1}",
                "src": f"Node{src}",
                "dst": f"Node{dst}",
                "dscp": dscp,
                "sla_bandwidth": sla_bw
            }
            scenario_flows.append(flow_record)

        scenario["non_graph_features"]["traffic_flows"] = new_flows
        scenario["non_graph_features"].pop("traffic_pairs", None)
        custom_serializer(scenario)

        flow_summary.append(["scenario_id:", scenario_id, scenario_flows])

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    with open(flow_summary_file, "w") as f:
        json.dump(flow_summary, f, indent=2)

if __name__ == "__main__":
    augment_traffic_pairs(
        "test_dataset_all.json", 
        "test_dataset_with_flows.json", 
        "flow_summary.json"
    )
