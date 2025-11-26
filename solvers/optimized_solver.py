import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solvers.simple_propagation import propagate_constraints
from utils.graph_utils import board_to_tensor, get_sudoku_edges

def get_valid_candidates(board, row, col):
    """
    해당 셀에 들어갈 수 있는 유효한 숫자 리스트 반환 (Logic)
    """
    candidates = []
    # 미리 1~9 Set에서 행/열/박스 숫자를 빼는 방식이 빠름
    existing = set()
    existing.update(board[row, :])
    existing.update(board[:, col])
    br, bc = row // 3, col // 3
    existing.update(board[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten())
    
    for num in range(1, 10):
        if num not in existing:
            candidates.append(num)
    return candidates

def recursive_dynamic_solve(board, cached_probs):
    """
    1. Variable Ordering: 후보가 가장 적은 칸을 동적으로 선택 (Logic)
    2. Value Ordering: 선택된 칸의 후보 중 GNN 확률이 높은 순서로 시도 (GNN)
    """
    
    # 1. 가장 '급한' 빈칸 찾기 (Minimum Remaining Values Heuristic)
    best_r, best_c = -1, -1
    min_len = 10 # 후보 개수는 최대 9개이므로 10으로 초기화
    best_candidates = []
    
    # 빈칸이 있는지 확인 + 동시에 최적의 빈칸 탐색
    # (이중 루프가 매번 돌지만, 빈칸 개수가 줄어들수록 빨라짐)
    empty_cells = np.argwhere(board == 0)
    
    if len(empty_cells) == 0:
        return True, board # 다 채움 (성공)

    for r, c in empty_cells:
        cands = get_valid_candidates(board, r, c)
        
        if len(cands) == 0:
            return False, None # 후보가 없는 빈칸 발견 -> 모순 (Backtrack)
        
        if len(cands) < min_len:
            min_len = len(cands)
            best_r, best_c = r, c
            best_candidates = cands
            
            if min_len == 1:
                break # 후보가 1개면 더 볼 것도 없이 얘부터 해야 함 (최적화)

    # 2. 값 정렬 (GNN Probability)
    # best_candidates 리스트의 숫자들을 GNN 확률이 높은 순서대로 정렬
    idx = best_r * 9 + best_c
    cell_probs = cached_probs[idx] # Tensor [9]
    
    # 후보 숫자(val)에 대해 cached_probs[val-1] 값을 기준으로 내림차순 정렬
    # (Python sort가 작은 리스트에서는 매우 빠름)
    best_candidates.sort(key=lambda val: cell_probs[val-1].item(), reverse=True)
    
    # 3. Recursion
    for num in best_candidates:
        board[best_r, best_c] = num
        
        success, final_board = recursive_dynamic_solve(board, cached_probs)
        if success:
            return True, final_board
        
        board[best_r, best_c] = 0 # Backtrack

    return False, None

def solve_sudoku_optimized(input_str, model, device):
    # 1. Init
    initial_board = np.array([int(c) for c in input_str]).reshape(9, 9)
    edge_index = get_sudoku_edges().to(device)
    
    # 2. Constraint Propagation (Pre-processing)
    board, solved, contradiction = propagate_constraints(initial_board)
    if contradiction: return False, None
    if solved: return True, board
    
    # 3. GNN Inference (One-Shot)
    x = torch.tensor(board.flatten(), dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad():
        out = model(x, edge_index)
        cached_probs = torch.softmax(out, dim=1) # [81, 9]

    # 4. Dynamic Recursion Start
    success, final_board = recursive_dynamic_solve(board, cached_probs)
    
    if final_board is None:
        return False, initial_board
        
    return True, final_board