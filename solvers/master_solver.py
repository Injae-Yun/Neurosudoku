import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from solvers.simple_propagation import propagate_constraints
from utils.graph_utils import board_to_tensor, get_sudoku_edges

def is_valid_move(board, row, col, num):
    if num in board[row, :]: return False
    if num in board[:, col]: return False
    br, bc = row // 3, col // 3
    if num in board[br*3:(br+1)*3, bc*3:(bc+1)*3]: return False
    return True

def recursive_hybrid_solve(board, cached_probs):
    """
    Args:
        board: 현재 보드 상태
        cached_probs: [81, 9] (맨 처음에 한 번 계산한 GNN 확률값)
    """
    # 1. 논리적 전파 (Logic Phase)
    # (박사님이 수정하신 True/False 로직 반영됨을 가정)
    board, solved, contradiction = propagate_constraints(board)
    
    if contradiction:
        return False, None
    if solved:
        return True, board

    # 2. 빈칸 선택 (Heuristic Selection with Cached Probs)
    # 여기서는 더 이상 model(x)를 호출하지 않습니다! (속도 향상의 핵심)
    
    best_r, best_c = -1, -1
    max_conf = -1.0
    
    empty_indices = np.argwhere(board == 0)
    
    if len(empty_indices) == 0:
        return True, board

    for r, c in empty_indices:
        idx = r * 9 + c
        # 미리 계산된 확률표(cached_probs)만 조회 (O(1))
        conf = torch.max(cached_probs[idx]).item()
        
        if conf > max_conf:
            max_conf = conf
            best_r, best_c = r, c

    # 3. 가지치기 탐색 (Branching)
    idx = best_r * 9 + best_c
    cell_probs = cached_probs[idx] # [9]
    
    # 확률 높은 순서대로 정렬
    candidates = torch.argsort(cell_probs, descending=True)
    
    for cand_idx in candidates:
        num = cand_idx.item() + 1
        
        if is_valid_move(board, best_r, best_c, num):
            board[best_r, best_c] = num
            
            # 재귀 호출 시에도 cached_probs를 그대로 넘김
            success, final_board = recursive_hybrid_solve(board, cached_probs)
            
            if success:
                return True, final_board
            
            # Backtrack
            board[best_r, best_c] = 0
            
    return False, None

def solve_sudoku_neuro_symbolic(input_str, model, device):
    initial_board = np.array([int(c) for c in input_str]).reshape(9, 9)
    edge_index = get_sudoku_edges().to(device)
    
    # [Step 1] GNN Inference (딱 한 번만 실행!) ⚡
    # 초기 상태에서의 "직관"을 계산하여 저장
    x = torch.tensor(initial_board.flatten(), dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad():
        out = model(x, edge_index)
        cached_probs = torch.softmax(out, dim=1) # [81, 9]
    
    # [Step 2] Recursion Start
    # 모델 대신 확률 테이블(cached_probs)을 넘김
    # 초기 보드는 복사해서 사용
    solve_board = initial_board.copy()
    success, final_board = recursive_hybrid_solve(solve_board, cached_probs)
    
    if final_board is None:
        final_board = initial_board
        
    return success, final_board