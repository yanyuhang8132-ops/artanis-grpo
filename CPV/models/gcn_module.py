import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
    
class GCNModule(nn.Module):
    def __init__(self, ec_vocab_size, ec_embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.ec_embed = nn.Embedding(ec_vocab_size, ec_embed_dim)
        self.conv1 = GCNConv(4 + ec_embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, adj, node_features):
        edge_index, _ = dense_to_sparse(adj)
        x = F.relu(self.conv1(node_features, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)

    def build_node_features(self, graph_data):
        # [N, 1]
        in_degree = graph_data["in_degree"].unsqueeze(-1)
        out_degree = graph_data["out_degree"].unsqueeze(-1)
        betweenness = graph_data["betweenness"].unsqueeze(-1)
        ec_counts = graph_data["ec_counts"].unsqueeze(-1)

        # ec_networks: [N, K]
        ec_ids = graph_data["ec_networks"]
        ec_emb = self.ec_embed(ec_ids)          # [N, K, D]
        ec_emb = ec_emb.mean(dim=1)             # [N, D]

        return torch.cat([in_degree, out_degree, betweenness, ec_counts, ec_emb], dim=-1)  # [N, D]
