import argparse
import torch
import numpy as np
import sys
import os
import time

from models.gnn_solver_v2 import RecurrentGNN
from utils.visualize import print_sudoku
#from solvers.master_solver import solve_sudoku_neuro_symbolic as solver_sudoku
from solvers.optimized_solver import solve_sudoku_optimized as solver_sudoku

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
    default="800000000003600000070090200050007000000045700000100030001000068008500010090000400", 
    help='Sudoku string')
    parser.add_argument('--model', type=str, default='models/best_model.pth')
    args = parser.parse_args()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # 1. Load GATv2 Model
    # GATv2Conv를 썼으므로 heads 파라미터 등이 __init__ 기본값과 일치해야 함
    model = RecurrentGNN(hidden_dim=96, num_steps=32, heads=4).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print("✅ GATv2 Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Tip: Did you change the model architecture? Make sure hyperparameters match.")
        return

    # 2. Solve (Neuro-Symbolic)
    print(f"\n🧩 Puzzle: {args.input[:15]}...")
    start_time = time.time()
    
    success, result_grid = solver_sudoku(args.input, model, device)
    
    end_time = time.time()
    elapsed = end_time - start_time

    # 3. Visualization
    original_grid = np.array([int(c) for c in args.input]).reshape(9, 9)
    
    if success:
        print(f"\n🎉 Solved in {elapsed:.4f} sec!")
        print_sudoku(result_grid, original=original_grid)
    else:
        print(f"\n💀 Failed to solve after {elapsed:.4f} sec.")
        print_sudoku(result_grid, original=original_grid)

if __name__ == "__main__":
    main()