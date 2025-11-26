import torch
import numpy as np

def get_sudoku_edges():
    """
    Generate the static edge index for a 9x9 Sudoku graph.
    Nodes are indexed 0 to 80 (row-major).
    Edges represent constraints: same row, same col, same 3x3 block.
    """
    edges = set()
    
    for i in range(81):
        row, col = i // 9, i % 9
        block_row, block_col = row // 3, col // 3
        
        # Constraints
        for j in range(81):
            if i == j: continue # No self-loops needed for now
            
            r, c = j // 9, j % 9
            br, bc = r // 3, c // 3
            
            # Check connection rules
            is_same_row = (row == r)
            is_same_col = (col == c)
            is_same_block = (block_row == br) and (block_col == bc)
            
            if is_same_row or is_same_col or is_same_block:
                edges.add((i, j))
                edges.add((j, i)) # Undirected

    # Convert to PyTorch Geometric format [2, num_edges]
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index

def board_to_tensor(board_str):
    """
    Convert a string of 81 digits (e.g., "53007...") to a torch tensor.
    0 represents an empty cell.
    """
    # Assuming input is a string of digits or a list
    if isinstance(board_str, str):
        data = [int(c) for c in board_str]
    else:
        data = board_str
        
    return torch.tensor(data, dtype=torch.long)