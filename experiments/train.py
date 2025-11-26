import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 프로젝트 루트 경로를 path에 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gnn_solver_v2 import RecurrentGNN
from data.load_dataset import SudokuDataset

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked_elements = 0
    
    # tqdm으로 진행 상황 시각화
    pbar = tqdm(loader, desc="Training", unit="batch")
    
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward Pass
        # x: [Batch * 81, 1], edge_index: [2, E]
        out = model(data.x, data.edge_index) # Output: [Batch * 81, 9]
        
        # Loss Calculation (Masked)
        # 힌트로 주어진 숫자(mask=False)는 loss 계산에서 제외하고,
        # 모델이 맞춰야 할 빈칸(mask=True)만 학습합니다.
        mask = data.train_mask
        
        if mask.sum() == 0:
            continue # 혹시라도 빈칸이 없는 데이터가 있다면 패스
            
        # data.y는 0~8 (정답 숫자 1~9에 대응)
        loss = F.cross_entropy(out[mask], data.y[mask])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy Calculation (Cell-level)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        total_correct += correct
        total_masked_elements += mask.sum().item()
        
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_masked_elements if total_masked_elements > 0 else 0
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_masked_elements = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        
        mask = data.train_mask
        pred = out.argmax(dim=1)
        
        correct = (pred[mask] == data.y[mask]).sum().item()
        total_correct += correct
        total_masked_elements += mask.sum().item()
        
    return total_correct / total_masked_elements if total_masked_elements > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Train NeuroSudoku GNN Solver")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--steps', type=int, default=32, help='Recurrent steps for GNN')
    parser.add_argument('--hidden', type=int, default=96, help='Hidden dimension')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    args = parser.parse_args()

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # Dataset Setup
    # csv가 data/raw/sudoku.csv에 있다고 가정
    dataset = SudokuDataset(root=args.data_dir, csv_path=os.path.join(args.data_dir, 'raw/sudoku.csv'))
    
    # 9:1 Train/Val Split
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"📊 Dataset: {len(dataset)} total | {len(train_dataset)} train | {len(val_dataset)} val")

    # Model Setup
    model = RecurrentGNN(hidden_dim=args.hidden, num_steps=args.steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("🔥 Starting Training...")
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Train Cell Acc: {train_acc*100:.2f}% | Val Cell Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 모델 저장
            save_path = os.path.join("models", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Best model saved to {save_path}")

if __name__ == "__main__":
    main()