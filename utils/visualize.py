import torch
import numpy as np

def get_conflict_mask(grid):
    """
    9x9 그리드에서 규칙을 위반한 셀의 마스크(True=위반)를 반환
    """
    conflict_mask = np.zeros((9, 9), dtype=bool)
    
    for r in range(9):
        for c in range(9):
            val = grid[r, c]
            if val == 0: continue
            
            # 1. Row Check
            if np.sum(grid[r, :] == val) > 1:
                conflict_mask[r, c] = True
            # 2. Col Check
            if np.sum(grid[:, c] == val) > 1:
                conflict_mask[r, c] = True
            # 3. Box Check
            br, bc = r // 3, c // 3
            box = grid[br*3:(br+1)*3, bc*3:(bc+1)*3]
            if np.sum(box == val) > 1:
                conflict_mask[r, c] = True
                
    return conflict_mask

def print_sudoku(grid, original=None):
    """
    original: 초기 문제 (이건 절대 안 틀린다고 가정)
    grid: 모델이 푼 결과
    """
    # ANSI Colors
    RED = '\033[91m'   # 충돌 (에러)
    BLUE = '\033[94m'  # 모델이 채운 값 (정상)
    RESET = '\033[0m'
    
    conflicts = get_conflict_mask(grid)
    
    print("-" * 25)
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("-" * 25)
        row_str = ""
        for j in range(9):
            if j > 0 and j % 3 == 0:
                row_str += "| "
            
            val = grid[i, j]
            val_str = str(val) if val != 0 else "."
            
            is_error = conflicts[i, j]
            is_filled = (original is not None) and (original[i, j] == 0) and (val != 0)
            
            if is_error:
                # 틀린 건 무조건 빨간색
                row_str += f"{RED}{val_str}{RESET} "
            elif is_filled:
                # 모델이 채웠고 에러가 없으면 파란색
                row_str += f"{BLUE}{val_str}{RESET} "
            else:
                # 원래 있던 숫자
                row_str += f"{val_str} "
                
        print(row_str)
    print("-" * 25)

    if np.any(conflicts):
        print(f"{RED}⚠️  DETECTED ERRORS: The model violated Sudoku rules!{RESET}")
    else:
        print(f"{BLUE}✅ Perfect Solution!{RESET}")