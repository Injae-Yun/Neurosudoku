import sys
import os
import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gnn_solver_v2 import RecurrentGNN
from data.load_dataset import SudokuDataset
from solvers.optimized_solver import solve_sudoku_optimized as solve_sudoku

# -------------------------------------------------------------------------
# 1. Standard Backtracking Solver (Baseline)
# -------------------------------------------------------------------------
def is_valid(board, row, col, num):
    if num in board[row]: return False
    if num in [board[i][col] for i in range(9)]: return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num: return False
    return True

def solve_sudoku_backtracking(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku_backtracking(board): return True
                        board[row][col] = 0
                return False
    return True

# -------------------------------------------------------------------------
# 2. Visualization
# -------------------------------------------------------------------------
def save_performance_graph(gnn_time, bt_time, gnn_acc, bt_acc):
    labels = ['Neuro-Symbolic', 'Backtracking']
    times = [gnn_time, bt_time]
    accs = [gnn_acc, bt_acc]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Solver Type')
    ax1.set_ylabel('Avg Time (sec)', color=color)
    bars = ax1.bar(labels, times, color=color, alpha=0.6, label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 시간 값 표시
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color)
    line = ax2.plot(labels, accs, color=color, marker='o', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)

    # 정확도 값 표시
    for i, acc in enumerate(accs):
        ax2.text(i, acc + 2, f'{acc:.1f}%', ha='center', color='red', fontweight='bold')

    plt.title('Performance Benchmark: Neuro-Symbolic vs Backtracking')
    fig.tight_layout()
    
    save_path = 'benchmark_result.png'
    plt.savefig(save_path)
    print(f"\n")
    plt.close()

# -------------------------------------------------------------------------
# 3. Evaluation Logic
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate_benchmark(model, dataset, num_samples=1000, device='cpu'):
    model.eval()
    
    gnn_correct = 0
    gnn_total_time = 0
    bt_correct = 0
    bt_total_time = 0
    
    indices = np.random.choice(len(dataset), size=min(len(dataset), num_samples), replace=False)
    
    print(f"🔍 Benchmarking on {len(indices)} samples...")
    print(f"{'Method':<20} | {'Acc':<10} | {'Avg Time':<15}")
    print("-" * 55)

    for idx in tqdm(indices, desc="Running Benchmark"):
        data = dataset[idx]
        
        # Tensor -> String 변환 (Dynamic Solver 입력용)
        # data.x: [81, 1] -> flatten -> list -> string
        flat_list = data.x.squeeze().cpu().numpy().astype(int).tolist()
        input_str = "".join(map(str, flat_list))
        target_flat = data.y.cpu().numpy() + 1 # 정답 (1~9)

        # --- A. Neuro-Symbolic Solver ---
        start = time.time()
        success, result_board = solve_sudoku(input_str, model, device)
        gnn_time = time.time() - start
        gnn_total_time += gnn_time
        
        if success and np.array_equal(result_board.flatten(), target_flat):
            gnn_correct += 1

        # --- B. Backtracking Solver ---
        bt_input = np.array(flat_list).reshape(9, 9)
        start = time.time()
        solve_sudoku_backtracking(bt_input)
        bt_time = time.time() - start
        bt_total_time += bt_time
        
        if np.array_equal(bt_input.flatten(), target_flat):
            bt_correct += 1

    # Report
    avg_gnn_time = gnn_total_time / len(indices)
    avg_bt_time = bt_total_time / len(indices)
    gnn_acc = (gnn_correct / len(indices)) * 100
    bt_acc = (bt_correct / len(indices)) * 100
    
    print("\n" + "="*55)
    print("📊 Final Benchmark Results")
    print("="*55)
    print(f"Neuro-Symbolic (GNN): {gnn_acc:.2f}% Acc | {avg_gnn_time:.5f} sec")
    print(f"Backtracking (CPU):   {bt_acc:.2f}% Acc | {avg_bt_time:.5f} sec")
    print("="*55)

    save_performance_graph(avg_gnn_time, avg_bt_time, gnn_acc, bt_acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SudokuDataset(root='./data', csv_path='./data/raw/sudoku.csv')
    
    # 모델 로드 시 heads=4 필수 (GATv2)
    model = RecurrentGNN(hidden_dim=96, num_steps=32, heads=4).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"✅ Loaded model from {args.model_path}")
        evaluate_benchmark(model, dataset, num_samples=args.samples, device=device)
    else:
        print("❌ Model file not found.")

if __name__ == "__main__":
    main()