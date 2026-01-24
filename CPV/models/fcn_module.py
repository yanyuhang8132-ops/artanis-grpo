import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNModule(nn.Module):
    def __init__(self, interface_dim, hidden_dim, N_max):
        super().__init__()
        self.N_max = N_max
        
        # 接口编码的嵌入层
        self.interface_embedding = nn.Embedding(interface_dim, hidden_dim)  # 将接口索引映射到 hidden_dim 维向量
        
        # 流量对编码的嵌入层
        self.traffic_encoder = nn.Embedding(N_max * N_max, hidden_dim)
        
        # 合并特征的全连接层
        self.fcn_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, interfaces, traffic_pairs):
        # 接口编码处理
        # 将独热编码转换为索引编码
        interface_indices = torch.argmax(interfaces, dim=1)  # [N_max]
        interface_feat = self.interface_embedding(interface_indices)  # [N_max, hidden_dim]
        interface_feat = interface_feat.mean(dim=0)  # 全局平均池化 [hidden_dim]
        
        # 流量对编码处理
        src = traffic_pairs[:, 0]  # 源节点索引 [num_pairs]
        dst = traffic_pairs[:, 1]  # 目标节点索引 [num_pairs]
        traffic_indices = src * self.N_max + dst  # 将 (src, dst) 映射到一维索引
        traffic_feat = self.traffic_encoder(traffic_indices)  # [num_pairs, hidden_dim]
        traffic_feat = traffic_feat.mean(dim=0)  # 全局平均池化 [hidden_dim]
        
        # 合并特征
        combined = torch.cat([interface_feat, traffic_feat], dim=0)  # [hidden_dim * 2]
        fcn_out = F.relu(self.fcn_fc(combined))  # [hidden_dim]
        
        return fcn_out