import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm

# GATv2Conv를 사용한 Recurrent GNN 모델
# attention 메커니즘을 통해 이웃 노드의 중요도를 동적으로 학습
# GCNConv 기반 모델보다 더 유연하고 강력한 표현력 기대

class RecurrentGNN(nn.Module):
    def __init__(self, hidden_dim=96, num_steps=32, num_classes=9, heads=4):
        super(RecurrentGNN, self).__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        
        # 1. Embedding
        # 0(빈칸) ~ 9(숫자) -> 벡터
        self.embedding = nn.Embedding(10, hidden_dim)
        
        # 2. Processor (Attention Mechanism)
        # GATv2Conv: Dynamic Attention Weights 계산
        # heads=4: 4가지 다른 관점(subspace)에서 관계를 봄 (Row, Col, Box 등을 스스로 학습하길 기대)
        self.conv = GATv2Conv(
            hidden_dim, 
            hidden_dim // heads, 
            heads=heads, 
            concat=True, 
            dropout=0.1
        )
        
        # GRU Cell for Recurrence (Memory)
        # GAT의 출력은 hidden_dim 크기로 맞춰짐 (heads * (hidden // heads))
        self.norm = LayerNorm(hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # 3. Decoder
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, labels=None):
        if x.dim() > 1:
            x = x.squeeze()

        # Initial Hidden State
        h = self.embedding(x) # [N, hidden_dim]
        
        # Recurrent Message Passing with Attention
        for step in range(self.num_steps):
            # (1) Attention-based Message Passing
            # 여기서 edge_index를 통해 이웃 정보를 가져올 때, 
            # "얼마나 중요한 정보인가"를 attention score로 계산함
            m = self.conv(h, edge_index)
            m = F.elu(m) # GAT는 보통 ELU를 사용
            m = self.norm(m)
            
            # (2) State Update (Memory)
            h = self.rnn(m, h)

        # Final Prediction
        out = self.fc(h)
        return out