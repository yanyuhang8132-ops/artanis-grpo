import torch
import torch.nn.functional as F
import torch.nn as nn
from .gcn_module import GCNModule
from .attention_module import AttentionFCNModule

class MultimodalMultitaskModel(nn.Module):
    def __init__(self, N_max, flow_input_dim, ec_vocab_size, ec_embed_dim=8, hidden_dim=16, num_heads=4):
        super().__init__()
        self.N_max = N_max
        self.N2 = N_max * N_max

        # 图模态分支（输入维度 = 4 + ec_embed_dim）
        self.gcn = GCNModule(ec_vocab_size=ec_vocab_size, ec_embed_dim=ec_embed_dim,
                             hidden_dim=hidden_dim, output_dim=hidden_dim)

        # 非图模态分支（输入 flow_features）
        self.attention_fcn = AttentionFCNModule(flow_input_dim=flow_input_dim,
                                                hidden_dim=hidden_dim,
                                                num_heads=num_heads)

        # 融合 + 多任务预测
        self.fusion_fc = nn.Linear(2 * hidden_dim, hidden_dim)
        # 输出维度 = 6*N + 3*N^2（新增 blackhole + link_overload_pairs）
        # self.task_fc = nn.Linear(hidden_dim, 6 * N_max + 3 * N_max * N_max)
        self.task_fc = nn.Linear(hidden_dim, 3 * N_max + 2 * N_max * N_max + 30)  # +30 为最大 flow 数
        self.max_flows = 30

    def forward(self, graph_data, non_graph_data):
        # GCN分支
        node_features = self.gcn.build_node_features(graph_data)
        gcn_node_out = self.gcn(graph_data["adj"], node_features)  # [N, hidden]
        gcn_out = gcn_node_out.mean(dim=0)  # [hidden_dim]

        # Attention分支
        fcn_out = self.attention_fcn(non_graph_data["traffic_flows"]) 
        fcn_out = fcn_out.mean(dim=0)  # [N] → 可替换为 .mean(dim=0)
        fcn_out = fcn_out[:self.gcn.fc.out_features]  # 对齐 hidden_dim 长度

        # 融合
        fused = torch.cat([gcn_out, fcn_out], dim=0)  # [2 * hidden_dim]
        fused = F.relu(self.fusion_fc(fused))         # [hidden_dim]

        out = self.task_fc(fused)  # [3N + 2N^2 + 30]
        N, N2 = self.N_max, self.N2

        # 划分输出
        idx = 0
        isolated = out[idx:idx + N]; idx += N
        loop = out[idx:idx + N]; idx += N
        blackhole = out[idx:idx + N]; idx += N
        # high_util = out[idx:idx + N]; idx += N
        reachability = out[idx:idx + N2].view(N, N); idx += N2
        # load_balancing = out[idx:idx + N2].view(N, N); idx += N2
        link_overload_pairs = out[idx:idx + N2].view(N, N)
        sla_labels = out[idx:idx + self.max_flows]

        return {
            "isolated": torch.sigmoid(isolated),
            "loop": torch.sigmoid(loop),
            "blackhole": torch.sigmoid(blackhole),
            # "high_util": torch.sigmoid(high_util),
            "reachability": torch.sigmoid(reachability),
            # "load_balancing": torch.sigmoid(load_balancing),
            "link_overload_pairs": torch.sigmoid(link_overload_pairs),
            "sla_labels": torch.sigmoid(sla_labels)
        }
