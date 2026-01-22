import torch
import numpy as np
from CPV.models.multimodal_model import MultimodalMultitaskModel

# 全局加载验证模型
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_path = "CPV/saved_models/multimodal_multitask_model.pth"
_model = None

def load_verifier_model(N_max, flow_input_dim):
    global _model
    if _model is not None:
        return _model
    _model = MultimodalMultitaskModel(
        N_max=N_max,
        flow_input_dim=flow_input_dim,
        ec_vocab_size=10000,
        ec_embed_dim=8,
        hidden_dim=16,
        num_heads=4
    )
    _model.load_state_dict(torch.load(_model_path, map_location=_device))
    _model.eval().to(_device)
    return _model

# 主函数：输入是场景字典（graph + non_graph + mask + labels），输出 reward
def compute_reward(sample):
    model = load_verifier_model(
        N_max=sample["graph"]["adj"].shape[0],
        flow_input_dim=sample["non_graph"]["traffic_flows"].shape[1]
    )

    with torch.no_grad():
        graph_data = {k: v.to(_device) for k, v in sample["graph"].items()}
        non_graph_data = {k: v.to(_device) for k, v in sample["non_graph"].items()}
        labels = {k: v.to(_device) for k, v in sample["labels"].items()}
        mask = sample["mask"].to(_device)

        outputs = model(graph_data, non_graph_data)

        acc_list = []
        for task in outputs:
            if task not in labels:
                continue

            if task in ["reachability", "link_overload_pairs"]:
                pred = (outputs[task][mask][:, mask] > 0.5).int()
                true = labels[task][mask][:, mask].int()
            elif task == "sla_labels":
                pred = (outputs[task] > 0.5).int()
                true = labels[task].int()
            else:
                pred = (outputs[task][mask] > 0.5).int()
                true = labels[task][mask].int()

            pred = pred.cpu().numpy()
            true = true.cpu().numpy()

            correct = (pred == true).sum()
            total = pred.size
            acc = correct / (total + 1e-10)
            acc_list.append(acc)

        return float(sum(acc_list))  # ← 不取平均，直接求和
