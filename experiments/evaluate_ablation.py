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
from solvers.simple_propagation import propagate_constraints
from utils.graph_utils import get_sudoku_edges

# 악마의 문제들 (Value Ordering이 중요한 케이스)
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

TIMEOUT_SEC = 60.0
class TimeoutException(Exception):
    pass
# -------------------------------------------------------------------------
# 1. Pure Logic Solver (MRV Only, No GNN)
# -------------------------------------------------------------------------
def get_valid_candidates(board, row, col):
    candidates = []
    existing = set()
    existing.update(board[row, :])
    existing.update(board[:, col])
    br, bc = row // 3, col // 3
    existing.update(board[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten())
    for num in range(1, 10):
        if num not in existing:
            candidates.append(num)
    return candidates

def _recursive_mrv(board, start_time, steps):
    # 타임아웃 체크 (Mutable Steps 사용)
    steps[0] += 1
    if steps[0] % 1000 == 0:
        if time.time() - start_time > TIMEOUT_SEC:
            raise TimeoutException()
    # 1. MRV (Most Constrained Variable)
    best_r, best_c = -1, -1
    min_len = 10 
    best_candidates = []
    
    empty_cells = np.argwhere(board == 0)
    if len(empty_cells) == 0: return True # 성공
    
    for r, c in empty_cells:
        cands = get_valid_candidates(board, r, c)
        if len(cands) == 0: return False # 모순
        
        if len(cands) < min_len:
            min_len = len(cands)
            best_r, best_c = r, c
            best_candidates = cands
            if min_len == 1: break

    # 2. Sequential Value Ordering (No GNN)
    # GNN 없이 그냥 숫자 크기 순서대로(1,2,3...) 대입
    # best_candidates는 이미 오름차순 정렬되어 있음
    
    for num in best_candidates:
        board[best_r, best_c] = num
        
        if _recursive_mrv(board, start_time, steps):
            return True
            
        board[best_r, best_c] = 0 # Backtrack

    return False

def solve_mrv_pure(puzzle_str):
    start_time = time.time()
    board = np.array([int(c) for c in puzzle_str]).reshape(9, 9)
    
    # Propagation은 공정하게 적용
    board, solved, contradiction = propagate_constraints(board)
    if contradiction: return False, TIMEOUT_SEC
    if solved: return True, time.time() - start_time
    
    steps = [0]
    try:
        success = _recursive_mrv(board, start_time, steps)
        elapsed = time.time() - start_time
        return success, elapsed
    except TimeoutException:
        # ★ 예외를 잡아서 즉시 타임아웃 처리
        return False, TIMEOUT_SEC

# -------------------------------------------------------------------------
# 2. Run Ablation
# -------------------------------------------------------------------------
def run_ablation(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔬 Ablation Study: MRV Only vs MRV + GNN (Timeout: {TIMEOUT_SEC}s)")
    
    # Load Model
    model = RecurrentGNN(hidden_dim=96, num_steps=32, heads=4).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("❌ Model not found.")
        return
    
    print("-" * 80)
    print(f"{'Puzzle Name':<20} | {'MRV Only (s)':<12} | {'MRV + GNN (s)':<15} | {'GNN Impact'}")
    print("-" * 80)
    
    mrv_wins = 0
    gnn_wins = 0
    
    for i, p_str in enumerate(EXTREME_PUZZLES):
        # A. MRV Only (Baseline)
        success_mrv, time_mrv = solve_mrv_pure(p_str)
        
        if not success_mrv:
            str_mrv = "TIMEOUT"
            val_mrv = TIMEOUT_SEC
        else:
            str_mrv = f"{time_mrv:.4f}"
            val_mrv = time_mrv
            
        # B. Neuro-Symbolic (Ours)
        start = time.time()
        success_gnn, _ = solve_sudoku(p_str, model, device, timeout=TIMEOUT_SEC)
        time_gnn = time.time() - start
        
        if not success_gnn or time_gnn > TIMEOUT_SEC:
            str_gnn = "TIMEOUT"
            val_gnn = TIMEOUT_SEC
        else:
            str_gnn = f"{time_gnn:.4f}"
            val_gnn = time_gnn
            
        # C. Compare
        if str_mrv == "TIMEOUT" and str_gnn == "TIMEOUT":
            impact = "DRAW (Too Hard)"
        elif str_mrv == "TIMEOUT":
            impact = "🚀 GNN Enables Solve"
            gnn_wins += 1
        elif str_gnn == "TIMEOUT":
            impact = "❌ GNN Failed"
            mrv_wins += 1
        else:
            ratio = val_mrv / val_gnn
            if ratio > 1.1:
                impact = f"⚡ {ratio:.1f}x Speedup"
                gnn_wins += 1
            elif ratio < 0.9:
                impact = f"🔻 {1/ratio:.1f}x Slowdown"
                mrv_wins += 1
            else:
                impact = "Same (~)"
        
        print(f"{PUZZLE_NAMES[i]:<20} | {str_mrv:<12} | {str_gnn:<15} | {impact}")

    print("-" * 80)
    print(f"Summary: GNN Improved/Enabled solving in {gnn_wins}/{len(EXTREME_PUZZLES)} cases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_model.pth')
    args = parser.parse_args()
    
    run_ablation(args.model)