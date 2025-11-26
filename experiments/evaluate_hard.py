import sys
import os
import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gnn_solver_v2 import RecurrentGNN
from data.load_dataset import SudokuDataset
from solvers.optimized_solver import solve_sudoku_optimized as solve_sudoku
from solvers.simple_propagation import propagate_constraints

# 캐시 파일 경로
CACHE_FILE = 'data/hard_indices.json'

# --- Backtracking Logic ---
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

# --- Caching & Filtering Logic ---
def load_cached_indices():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            indices = json.load(f)
        print(f"📂 Loaded {len(indices)} hard indices from cache.")
        return indices
    return []

def save_cached_indices(indices):
    # 중복 제거 및 정렬
    indices = sorted(list(set(indices)))
    with open(CACHE_FILE, 'w') as f:
        json.dump(indices, f)
    print(f"💾 Saved {len(indices)} hard indices to {CACHE_FILE}.")

def get_hard_samples(dataset, target_count=200):
    # 1. 캐시 로드
    hard_indices = load_cached_indices()
    
    # 2. 이미 충분하면 바로 반환
    if len(hard_indices) >= target_count:
        print(f"✅ Cache has enough samples ({len(hard_indices)} >= {target_count}). Skipping scan.")
        return hard_indices[:target_count]

    needed = target_count - len(hard_indices)
    print(f"🔍 Scanning dataset for {needed} more hard puzzles...")
    
    # 이미 찾은 인덱스는 제외 (Set으로 변환하여 검색 속도 향상)
    existing_set = set(hard_indices)
    
    # 데이터셋 전체 인덱스 정의 (학습 데이터 제외하게끔 뒤 순서부터 스캔)
    total_len = len(dataset)
    all_indices = range(total_len - 1, -1, -1) #reversed
    
    scan_count = 0
    new_found = 0
    
    for idx in tqdm(all_indices, desc="Scanning Dataset"):
        # 이미 찾은 건 패스
        if int(idx) in existing_set:
            continue
            
        data = dataset[idx]
        flat_list = data.x.squeeze().cpu().numpy().astype(int).tolist()
        board = np.array(flat_list).reshape(9, 9)
        
        # 논리 전파 시도
        _, solved, contradiction = propagate_constraints(board)
        
        # 논리만으로 안 풀리고, 모순도 없는 경우 -> HARD
        if not solved and not contradiction:
            hard_indices.append(int(idx)) # JSON 저장을 위해 int 변환
            new_found += 1
            
            if len(hard_indices) >= target_count:
                break
        
        scan_count += 1
        
        # 너무 오래 걸리는 것을 방지하기 위해 1000개 스캔할 때마다 현황 출력 (옵션)
        # if scan_count % 10000 == 0:
        #     print(f"   Scanned {scan_count}, Found {new_found} new hard cases...")

    # 3. 결과 저장
    save_cached_indices(hard_indices)
    
    if len(hard_indices) < target_count:
        print(f"⚠️ Scanned entire dataset but only found {len(hard_indices)} hard cases.")
    else:
        print(f"✅ Successfully collected {len(hard_indices)} hard cases.")
        
    return hard_indices

# --- Benchmark & Plot ---
def save_hard_benchmark_graph(gnn_time, bt_time, gnn_acc, bt_acc, num_samples):
    labels = ['Neuro-Symbolic', 'Backtracking']
    times = [gnn_time, bt_time]
    accs = [gnn_acc, bt_acc]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar Plot (Time)
    color = 'tab:blue'
    ax1.set_xlabel('Solver Type')
    ax1.set_ylabel('Avg Time (sec)', color=color)
    bars = ax1.bar(labels, times, color=color, alpha=0.6, label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')

    # Line Plot (Accuracy)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.set_ylim(0, 115) 
    ax2.plot(labels, accs, color=color, marker='o', linewidth=2, markersize=8, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    for i, acc in enumerate(accs):
        ax2.text(i, acc + 3, f'{acc:.1f}%', ha='center', color='red', fontweight='bold')

    plt.title(f'Benchmark on {num_samples} HARD Puzzles: Time vs Accuracy')
    fig.tight_layout()
    
    save_path = 'benchmark_hard.png'
    plt.savefig(save_path)
    print(f"\n")
    plt.close()

def evaluate_hard(model, dataset, num_samples, device):
    # 1. 어려운 문제 수집 (캐시 활용)
    hard_indices = get_hard_samples(dataset, num_samples)
    
    if len(hard_indices) == 0:
        print("⚠️ No hard puzzles available.")
        return

    gnn_times = []
    bt_times = []
    gnn_correct = 0
    bt_correct = 0
    
    print(f"\n⚔️ Starting Duel on {len(hard_indices)} HARD puzzles...")

    for idx in tqdm(hard_indices, desc="Benchmarking"):
        data = dataset[idx]
        flat_list = data.x.squeeze().cpu().numpy().astype(int).tolist()
        input_str = "".join(map(str, flat_list))
        target_flat = data.y.cpu().numpy() + 1

        # A. Neuro-Symbolic
        start = time.time()
        success_gnn, result_board = solve_sudoku(input_str, model, device)
        gnn_times.append(time.time() - start)
        
        if success_gnn and np.array_equal(result_board.flatten(), target_flat):
            gnn_correct += 1

        # B. Backtracking
        bt_input = np.array(flat_list).reshape(9, 9)
        start = time.time()
        solve_sudoku_backtracking(bt_input)
        bt_times.append(time.time() - start)
        
        if np.array_equal(bt_input.flatten(), target_flat):
            bt_correct += 1

    avg_gnn = sum(gnn_times) / len(gnn_times)
    avg_bt = sum(bt_times) / len(bt_times)
    
    gnn_acc = (gnn_correct / len(hard_indices)) * 100
    bt_acc = (bt_correct / len(hard_indices)) * 100

    print("\n" + "="*50)
    print("📊 HARD Benchmark Results")
    print("="*50)
    print(f"Neuro-Symbolic: {gnn_acc:.2f}% Acc | {avg_gnn:.5f} sec")
    print(f"Backtracking:   {bt_acc:.2f}% Acc | {avg_bt:.5f} sec")
    
    save_hard_benchmark_graph(avg_gnn, avg_bt, gnn_acc, bt_acc, len(hard_indices))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    # 기본 목표 개수를 200개로 상향
    parser.add_argument('--samples', type=int, default=200, help='Target number of hard puzzles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SudokuDataset(root='./data', csv_path='./data/raw/sudoku.csv')
    
    model = RecurrentGNN(hidden_dim=96, num_steps=32, heads=4).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate_hard(model, dataset, args.samples, device)
    else:
        print("Model not found.")

if __name__ == "__main__":
    main()