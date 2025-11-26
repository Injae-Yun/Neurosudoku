import torch
import numpy as np
import time
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
    existing = set()
    existing.update(board[row, :])
    existing.update(board[:, col])
    br, bc = row // 3, col // 3
    existing.update(board[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten())
    
    for num in range(1, 10):
        if num not in existing:
            candidates.append(num)
    return candidates

def recursive_dynamic_solve(board, cached_probs, start_time, timeout, steps):
    """
    Args:
        steps: [int] 리스트 (Call by Reference로 카운터 공유)
    """
    # ---------------------------------------------------------
    # 1. 타임아웃 체크
    # ---------------------------------------------------------
    steps[0] += 1
    # 1000번마다 체크 (너무 자주 체크하면 time.time 오버헤드 발생, 너무 적으면 반응 느림)
    if steps[0] % 2000 == 0:
        if timeout > 0 and (time.time() - start_time > timeout):
            return False, None # 타임아웃 발생 시그널 (None)

    # 2. MRV (Most Constrained Variable) 찾기
    best_r, best_c = -1, -1
    min_len = 10 
    best_candidates = []
    
    empty_cells = np.argwhere(board == 0)
    
    if len(empty_cells) == 0:
        return True, board # 성공

    for r, c in empty_cells:
        cands = get_valid_candidates(board, r, c)
        
        if len(cands) == 0:
            return False, board # 단순 실패 (Backtrack) - 보드는 그대로 리턴
        
        if len(cands) < min_len:
            min_len = len(cands)
            best_r, best_c = r, c
            best_candidates = cands
            
            if min_len == 1:
                break 

    # 3. GNN Value Ordering
    idx = best_r * 9 + best_c
    cell_probs = cached_probs[idx] 
    best_candidates.sort(key=lambda val: cell_probs[val-1].item(), reverse=True)
    
    # 4. Recursion
    for num in best_candidates:
        board[best_r, best_c] = num
        
        success, final_board = recursive_dynamic_solve(board, cached_probs, start_time, timeout, steps)
        
        if success:
            return True, final_board
        
        # 타임아웃 전파 (Signal Propagation) ★★★
        # 하위 재귀가 'None'을 리턴했다면, 이는 단순 실패가 아니라 '타임아웃'임.
        # 따라서 나도 즉시 'None'을 리턴해서 상위로 알려야 함.
        if final_board is None:
            return False, None

        # 타임아웃이 아니면(단순 실패면) 원상복구하고 다음 숫자 시도
        board[best_r, best_c] = 0 

    return False, board # 모든 숫자가 안 맞음 (단순 실패)

def solve_sudoku_dynamic(input_str, model, device, timeout=0.0):
    start_time = time.time()
    
    # 1. Init
    initial_board = np.array([int(c) for c in input_str]).reshape(9, 9)
    edge_index = get_sudoku_edges().to(device)
    
    # 2. Constraint Propagation
    board, solved, contradiction = propagate_constraints(initial_board)
    if contradiction: return False, None # 모순
    if solved: return True, board
    
    # 3. GNN Inference
    x = torch.tensor(board.flatten(), dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad():
        out = model(x, edge_index)
        cached_probs = torch.softmax(out, dim=1)

    # 4. Dynamic Recursion Start
    steps = [0]
    success, final_board = recursive_dynamic_solve(board, cached_probs, start_time, timeout, steps)
    
    # 타임아웃(None)이 반환되면 실패 처리
    if final_board is None:
        return False, initial_board
        
    return success, final_board