import sys
import os
import time
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.gnn_solver_v2 import RecurrentGNN
from solvers.optimized_solver import solve_sudoku_optimized as solve_sudoku

# --- Backtracking Logic (Baseline) ---
def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num: return False
        if board[i][col] == num: return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num: return False
    return True

def solve_backtracking(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_backtracking(board): return True
                        board[row][col] = 0
                return False
    return True

def check_solution(board):
    # 중복 검사
    for i in range(9):
        if len(set(board[i, :])) != 9: return False
        if len(set(board[:, i])) != 9: return False
    return True

# --- Main Comparison ---
def run_comparison(puzzle_str):
    print(f"🧩 Puzzle: {puzzle_str[:15]}... (Arto Inkala's Hardest)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 최신 모델 (GATv2) 로드
    model = RecurrentGNN(hidden_dim=96, num_steps=32, heads=4).to(device)
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 1. Neuro-Symbolic Solver Run
    start_gnn = time.time()
    # Dynamic Solver 호출
    success, result_board = solve_sudoku(puzzle_str, model, device)
    time_gnn = time.time() - start_gnn
    
    is_gnn_valid = success and check_solution(result_board)

    # 2. Backtracking Run
    bt_grid = np.array([int(c) for c in puzzle_str]).reshape(9, 9)
    start_bt = time.time()
    solve_backtracking(bt_grid)
    time_bt = time.time() - start_bt
    
    is_bt_valid = check_solution(bt_grid)

    # 3. Report
    print("\n" + "="*50)
    print(f"⚔️  Face-off Results")
    print("="*50)
    
    print(f"🤖 [Neuro-Symbolic (Dynamic GNN)]")
    print(f"   Time:   {time_gnn:.5f} sec")
    print(f"   Status: {'✅ Valid Solution' if is_gnn_valid else '❌ Failed'}")
    
    print(f"\n🧠 [Backtracking Algorithm]")
    print(f"   Time:   {time_bt:.5f} sec")
    print(f"   Status: {'✅ Valid Solution' if is_bt_valid else '❌ Failed'}")

    print("-" * 50)
    if is_gnn_valid and is_bt_valid:
        ratio = time_bt / time_gnn
        if ratio > 1:
            print(f"🚀 Winner: Neuro-Symbolic is {ratio:.2f}x FASTER than Backtracking!")
        else:
            print(f"🐌 Result: Neuro-Symbolic is {1/ratio:.2f}x slower than Backtracking.")
    else:
        print("⚠️ One of the solvers failed. Check logic.")

if __name__ == "__main__":
    # Arto Inkala's Puzzle
    puzzle = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
    run_comparison(puzzle)