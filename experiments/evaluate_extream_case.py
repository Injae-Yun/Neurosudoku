import sys
import os
import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gnn_solver_v2 import RecurrentGNN
from solvers.optimized_solver import solve_sudoku_dynamic as solve_sudoku

# 제한 시간 (초)
TIMEOUT_SEC = 60.0

EXTREME_PUZZLES = [
    "000006000059000008200008000045000000003000000006003054000325006000000000000000000", # Norvig #1
    "000005080000601043000000000010500000000106000300000005530000061000000004000000000", # Norvig #2
    "800000000003600000070090200050007000000045700000100030001000068008500010090000400", # AI Escargot
    "005300000800000020070010500400005300010070006003200080060500009004000030000009700", # Inkala 2010
    "000000000000003085001020000000507000004000100090000000500000073002010000000040009", # Platinum Blonde
    "000000039000001005003050800008090006070002000100400000009080050020000600400700000", # Golden Nugget
    "100000000002740000000500004030000000750000000000009600040006000000000071000001030", # Easter Monster
    "000000000050000801000002000020100003004000200500006090000500000608000070000000000", # Tarantula
    "020000000000600003074080000000003002080040010600500000000010780500009000000000040", # Red Dwarf
    "600000000004050000000001050090002000000030000000800070020300000000090200000000001"  # Unsolvable #28
]

PUZZLE_NAMES = [
    "Norvig #1", "Norvig #2", "AI Escargot", "Inkala 2010", "Platinum Blonde", "Golden Nugget", "Easter Monster",
    "Tarantula", "Red Dwarf",  "Unsolvable #28"
]

# -------------------------------------------------------------------------
# 1. Backtracking Logic with OPTIMIZED TIMEOUT
# -------------------------------------------------------------------------
class TimeoutException(Exception):
    pass

def is_valid(board, row, col, num):
    if num in board[row]: return False
    if num in [board[i][col] for i in range(9)]: return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num: return False
    return True

def solve_sudoku_backtracking(board):
    start_time = time.time()
    
    # ★ 핵심 최적화: 카운터를 클로저(Closure) 변수로 관리
    # steps[0]을 사용하여 참조에 의한 호출을 이용
    steps = [0] 
    
    def _recursive(board):
        # 1. 호출 횟수 증가
        steps[0] += 1
        
        # 2. 5000번마다 한 번만 시간 체크 (Overhead 제거)
        if steps[0] % 5000 == 0:
            if time.time() - start_time > TIMEOUT_SEC:
                raise TimeoutException()

        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if _recursive(board): return True
                            board[row][col] = 0
                    return False
        return True

    try:
        return _recursive(board)
    except TimeoutException:
        return False

# -------------------------------------------------------------------------
# 2. Benchmark Logic
# -------------------------------------------------------------------------
def run_extreme_benchmark(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running Extreme Benchmark (Timeout: {TIMEOUT_SEC}s)...")
    
    model = RecurrentGNN(hidden_dim=96, num_steps=32, heads=4).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("❌ Model not found.")
        return

    gnn_times = []
    bt_times = []
    
    print(f"\n{'Puzzle Name':<20} | {'Neuro (s)':<10} | {'Backtrack (s)':<15} | {'Speedup':<10}")
    print("-" * 70)

    for i, puzzle_str in enumerate(EXTREME_PUZZLES):
        name = PUZZLE_NAMES[i]
        
        # --- A. Neuro-Symbolic ---
        start = time.time()
        success, _ = solve_sudoku(puzzle_str, model, device, timeout=TIMEOUT_SEC)
        gnn_time = time.time() - start
        
        # GNN 타임아웃/실패 처리
        if not success or gnn_time > TIMEOUT_SEC:
            gnn_time = TIMEOUT_SEC
            gnn_str = f"> {TIMEOUT_SEC}s"
            gnn_failed = True
        else:
            gnn_str = f"{gnn_time:.4f}"
            gnn_failed = False
        
        # --- B. Backtracking ---
        bt_input = np.array([int(c) for c in puzzle_str]).reshape(9, 9)
        start = time.time()
        bt_success = solve_sudoku_backtracking(bt_input)
        bt_time = time.time() - start
        
        # Backtracking 타임아웃/실패 처리
        if not bt_success or bt_time > TIMEOUT_SEC:
            bt_time = TIMEOUT_SEC
            bt_str = f"> {TIMEOUT_SEC}s (TO)"
            bt_failed = True
        else:
            bt_str = f"{bt_time:.4f}"
            bt_failed = False
        
        # --- C. Report ---
        if gnn_failed and bt_failed:
            speedup_str = "DRAW"
        elif gnn_failed:
            speedup_str = "BT Wins"
        elif bt_failed:
            speedup_str = "Neuro Wins"
        else:
            ratio = bt_time / gnn_time
            speedup_str = f"{ratio:.2f}x" if ratio > 1 else f"0.{int(ratio*10)}x"
        
        print(f"{name:<20} | {gnn_str:<10} | {bt_str:<15} | {speedup_str}")
        
        gnn_times.append(gnn_time)
        bt_times.append(bt_time)

    visualize_results(gnn_times, bt_times)

def visualize_results(gnn_times, bt_times):
    x = np.arange(len(PUZZLE_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width/2, gnn_times, width, label='Neuro-Symbolic', color='#4e79a7')
    rects2 = ax.bar(x + width/2, bt_times, width, label='Backtracking', color='#e15759')

    ax.set_ylabel('Time (sec)')
    ax.set_title(f'Performance on World\'s Hardest Sudokus (Timeout: {TIMEOUT_SEC}s)')
    ax.set_xticks(x)
    ax.set_xticklabels(PUZZLE_NAMES, rotation=45, ha='right')
    ax.legend()
    
    plt.axhline(y=TIMEOUT_SEC, color='r', linestyle='--', alpha=0.5, label='Timeout Limit')

    plt.tight_layout()
    plt.savefig('benchmark_extreme.png')
    print("\n✅ Saved 'benchmark_extreme.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.pth')
    args = parser.parse_args()
    
    run_extreme_benchmark(args.model)