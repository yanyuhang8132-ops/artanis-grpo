from torch import nn
from torch.nn import functional as F

# class InterfaceAttention(nn.Module):
#     def __init__(self, interface_dim, hidden_dim, num_heads):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
#         self.fc = nn.Linear(interface_dim, hidden_dim)
        
#     def forward(self, interfaces):
#         # 将接口编码映射到 hidden_dim 维度
#         interfaces = self.fc(interfaces)  # [N_max, hidden_dim]
        
#         # Self-Attention
#         interfaces = interfaces.unsqueeze(1)  # [N_max, 1, hidden_dim]
#         attn_output, _ = self.attention(interfaces, interfaces, interfaces)
#         attn_output = attn_output.squeeze(1)  # [N_max, hidden_dim]
        
#         return attn_output  # 返回所有节点的特征 [N_max, hidden_dim]

# class TrafficAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
#         self.fc = nn.Linear(input_dim, hidden_dim)

#     def forward(self, flow_features):  # flow_features: [num_flows, D]
#         traffic_feat = self.fc(flow_features)         # [num_flows, hidden_dim]
#         traffic_feat = traffic_feat.unsqueeze(1)      # [num_flows, 1, hidden_dim]
#         attn_output, _ = self.attention(traffic_feat, traffic_feat, traffic_feat)
#         attn_output = attn_output.squeeze(1)          # [num_flows, hidden_dim]
#         return attn_output


# class CrossAttention(nn.Module):
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
#     def forward(self, interface_feat, traffic_feat):
#         # interface_feat: [N_max, hidden_dim]
#         # traffic_feat: [num_pairs, hidden_dim]
#         interface_feat = interface_feat.unsqueeze(1)  # [N_max, 1, hidden_dim]
#         traffic_feat = traffic_feat.unsqueeze(1)  # [num_pairs, 1, hidden_dim]
        
#         # Cross-Attention
#         attn_output, _ = self.attention(interface_feat, traffic_feat, traffic_feat)
#         attn_output = attn_output.squeeze(1)  # [N_max, hidden_dim]
#         return attn_output

# class AttentionFCNModule(nn.Module):
#     def __init__(self, interface_dim, flow_input_dim, hidden_dim, N_max, num_heads):
#         super().__init__()
#         self.interface_attention = InterfaceAttention(interface_dim, hidden_dim, num_heads)
#         self.traffic_attention = TrafficAttention(flow_input_dim, hidden_dim, num_heads)
#         self.cross_attention = CrossAttention(hidden_dim, num_heads)
#         self.fcn_fc = nn.Linear(hidden_dim, N_max)

#     def forward(self, interfaces, traffic_flow_features):
#         # interfaces: [N_max, interface_dim]
#         # traffic_flow_features: [num_flows, flow_input_dim]

#         interface_feat = self.interface_attention(interfaces)              # [N_max, hidden_dim]
#         traffic_feat = self.traffic_attention(traffic_flow_features)       # [num_flows, hidden_dim]
#         fused = self.cross_attention(interface_feat, traffic_feat)         # [N_max, hidden_dim]
#         fcn_out = F.relu(self.fcn_fc(fused))                               # [N_max, N_max]
#         return fcn_out

class TrafficAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, flow_features):  # [num_flows, D]
        traffic_feat = self.fc(flow_features)        # [num_flows, hidden_dim]
        traffic_feat = traffic_feat.unsqueeze(1)     # [num_flows, 1, hidden_dim]
        attn_output, _ = self.attention(traffic_feat, traffic_feat, traffic_feat)
        return attn_output.mean(dim=0)               # [hidden_dim] → mean pool over flows


class AttentionFCNModule(nn.Module):
    def __init__(self, flow_input_dim, hidden_dim, num_heads):
        super().__init__()
        self.traffic_attention = TrafficAttention(flow_input_dim, hidden_dim, num_heads)
        self.fcn_fc = nn.Identity()  # 可替换为线性层 if needed

    def forward(self, traffic_flow_features):
        traffic_feat = self.traffic_attention(traffic_flow_features)  # [hidden_dim]
        return self.fcn_fc(traffic_feat)  # [hidden_dim]