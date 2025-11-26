import torch
import numpy as np

def is_valid_move(board, row, col, num):
    # 충돌 체크 (기존과 동일)
    if num in board[row, :]: return False
    if num in board[:, col]: return False
    br, bc = row // 3, col // 3
    if num in board[br*3:(br+1)*3, bc*3:(bc+1)*3]: return False
    return True

def recursive_solve(board, empty_cells, probs, idx):
    """
    board: 현재 상태
    empty_cells: [(row, col, cell_index), ...] 로 미리 정렬된 빈칸 리스트
    probs: GNN 확률 테이블
    idx: 현재 처리할 empty_cells의 인덱스
    """
    # Base Case: 모든 빈칸을 다 채움
    if idx == len(empty_cells):
        return True

    row, col, cell_flat_idx = empty_cells[idx]

    # GNN이 추천하는 숫자 순서 가져오기
    # 매번 정렬하지 말고, 상위 k개만 가져오거나 미리 계산된 순서를 쓰면 더 빠름
    # 여기서는 직관성을 위해 해당 셀의 확률 분포를 가져와 정렬
    cell_probs = probs[cell_flat_idx] # [10]
    
    # 0번 인덱스 제외하고 1~9 확률 내림차순 정렬
    # (Tensor 연산이 무거우면 여기서 Numpy로 변환해서 처리해도 됨)
    candidates_probs = cell_probs[1:]
    candidates = torch.argsort(candidates_probs, descending=True) + 1
    
    for num_tensor in candidates:
        num = num_tensor.item()
        
        if is_valid_move(board, row, col, num):
            board[row, col] = num
            
            # 다음 빈칸(idx + 1)으로 이동
            if recursive_solve(board, empty_cells, probs, idx + 1):
                return True
            
            board[row, col] = 0 # Backtrack

    return False

def solve_with_gnn_guidance(board, probs):
    """
    Main Entry Point
    """
    # 1. 빈칸 정보 수집 (Pre-processing)
    empty_indices = np.argwhere(board == 0) # [[r, c], ...]
    if len(empty_indices) == 0:
        return True

    # 2. 빈칸 정렬 (Heuristic: Most Confident First)
    # GNN이 가장 강하게 확신하는 칸부터 채워야 트리가 덜 뻗어나감
    sorted_empty_cells = []
    
    for r, c in empty_indices:
        idx = r * 9 + c
        # 해당 셀의 최대 확률값 (Confidence)
        confidence = torch.max(probs[idx, 1:]).item()
        sorted_empty_cells.append((r, c, idx, confidence))
    
    # confidence(3번째 요소) 기준 내림차순 정렬
    # 즉, GNN이 "이건 확실해!" 하는 칸부터 리스트 앞단에 배치
    sorted_empty_cells.sort(key=lambda x: x[3], reverse=True)
    
    # 정렬된 리스트에서 (r, c, idx)만 추출
    final_empty_cells = [(x[0], x[1], x[2]) for x in sorted_empty_cells]

    # 3. 재귀 시작
    return recursive_solve(board, final_empty_cells, probs, 0)