import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm

class RecurrentGNN(nn.Module):
    def __init__(self, hidden_dim=96, num_steps=32, num_classes=9):
        super(RecurrentGNN, self).__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim

        # 1. Embedding Layer
        # 입력 0~9 (0: 빈칸, 1~9: 숫자)를 고차원 벡터로 변환
        self.embedding = nn.Embedding(10, hidden_dim)
        
        # 2. Processor (Message Passing Mechanism)
        # GCN + GRU Cell을 결합하여 '기억' 능력을 가진 GNN 구현
        self.conv = GCNConv(hidden_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # 3. Decoder (Classifier)
        # Hidden State를 9개의 숫자 확률로 변환
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, labels=None):
        # x shape: [Batch * 81, 1] -> squeeze -> [Batch * 81]
        if x.dim() > 1:
            x = x.squeeze()

        # Initial Hidden State (H0)
        h = self.embedding(x) # [N, hidden_dim]
        
        # Recurrent Message Passing
        for step in range(self.num_steps):
            # (1) Message Passing: 이웃 노드의 정보 집계
            m = self.conv(h, edge_index)
            m = F.relu(m)
            m = self.norm(m)
            
            # (2) State Update: GRU를 통해 이전 기억(h)과 새로운 정보(m) 통합
            h = self.rnn(m, h)

        # Final Prediction
        out = self.fc(h) # [N, 9] (Logits for digits 1-9)
        
        return out

if __name__ == "__main__":
    # 모델 아키텍처 테스트
    model = RecurrentGNN()
    print(model)
    
    # 더미 데이터 테스트
    dummy_x = torch.randint(0, 10, (81, 1))
    dummy_edge = torch.randint(0, 81, (2, 1620))
    output = model(dummy_x, dummy_edge)
    
    print(f"\nInput shape: {dummy_x.shape}")
    print(f"Output shape: {output.shape}") # [81, 9]가 나와야 함
    assert output.shape == (81, 9), "Output shape mismatch!"
    print("✅ Model build successful.")