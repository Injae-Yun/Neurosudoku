import numpy as np

def get_candidates(board, row, col):
    """특정 셀(row, col)에 들어갈 수 있는 후보 숫자 집합 반환"""
    if board[row, col] != 0:
        return set()
    
    candidates = set(range(1, 10))
    # Row, Col, Box Constraints
    candidates -= set(board[row, :])
    candidates -= set(board[:, col])
    
    br, bc = row // 3, col // 3
    candidates -= set(board[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten())
    
    return candidates

def propagate_constraints(board):
    """
    100% 확실한 칸들을 반복적으로 채워나감 (Naked Single 기법).
    변동 사항이 없을 때까지 반복 (Fixed Point Iteration).
    returns: (updated_board, is_stuck, is_contradiction)
    """
    board = board.copy()
    changed = True
    
    while changed:
        changed = False
        min_candidates = 10 # 후보가 가장 적은 칸을 찾기 위함
        best_cell = None
        
        # 모든 빈칸에 대해 후보군 조사
        for r in range(9):
            for c in range(9):
                if board[r, c] == 0:
                    cands = get_candidates(board, r, c)
                    
                    if len(cands) == 0:
                        return board, True, True # 모순 발생 (후보가 없음)
                    
                    if len(cands) == 1:
                        # 100% 확실한 경우 채워넣음
                        val = list(cands)[0]
                        board[r, c] = val
                        changed = True
                    else:
                        # 나중에 찍어야 할 때를 대비해 정보 수집 가능
                        pass
                        
    # 더 이상 논리적으로 채울 수 없음
    # 다 채웠는지 확인
    if np.all(board != 0):
        return board, True, False # 완료
        
    return board, False, False # Stuck (찍어야 함)