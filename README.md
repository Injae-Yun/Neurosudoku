# 🧠 Neuro-Symbolic Sudoku Solver

> **"Intuition meets Logic"**: A Hybrid AI System combining Graph Attention Networks (GATv2) with Dynamic Constraint Propagation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![PyG](https://img.shields.io/badge/PyG-Geometric-green)
![Status](https://img.shields.io/badge/Status-Solved_Extreme_Cases-success)

## 📌 Abstract

While Deep Learning excels at pattern recognition, it often struggles with strict logical reasoning, suffering from **"hallucinations"** (invalid moves) in combinatorial problems. Conversely, symbolic algorithms like Backtracking guarantee correctness but suffer from **exponential time complexity ($O(b^d)$)** on NP-hard instances.

This project introduces a **Neuro-Symbolic Architecture** that bridges this gap. By utilizing a **Recurrent Graph Attention Network (GATv2)** as a heuristic oracle for a dynamic backtracking search, we achieve state-of-the-art efficiency on extreme corner-case puzzles where traditional solvers struggle.

---

## 🚀 Key Results: Performance Analysis

We benchmarked the solver against an optimized Backtracking algorithm. The results demonstrate a clear trade-off between **fixed neural overhead** and **search efficiency**.

| Difficulty | Dataset / Source | Backtracking (CPU) | **Neuro-Symbolic (Ours)** | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | Kaggle (90k samples) | **< 0.004s** | ~0.003s | Solved mostly by **Constraint Propagation**. almost not use GNN module. |
| **Hard** | Kaggle Filtered (Top 1%) | **~0.007s** | ~0.037s | Logic depth is not deep enough to offset the GNN inference cost (Host-to-Device transfer). |
| **Extreme** | **Arto Inkala's Hardest** | 1.16s | **0.80s** | **~1.45x Faster.** The GNN heuristic significantly prunes the massive search tree, overcoming the initial overhead. |

### 📊 Interpretation
1.  **Easy & Hard Cases:** For standard puzzles, optimized Backtracking is extremely efficient due to CPU branch prediction. The Neuro-Symbolic approach incurs a fixed cost (GPU inference time), making it slightly slower.
2.  **Extreme Cases (OOD):** When the puzzle requires deep recursive search (like Arto Inkala's "AI Escargot"), Backtracking slows down linearly with the search space. Here, our **Neuro-Symbolic architecture shines** by guiding the search to the correct branch early, demonstrating robustness on NP-hard instances.

![Benchmark Result](benchmark_hard.png)
*(Generated via `experiments/evaluate_hard.py`)*

---

## 🏗 System Architecture

The solver is designed as a three-stage pipeline to ensure both **speed** and **correctness**.



### 1. Logic Layer: Constraint Propagation
Before invoking any heavy computation, the system applies **Naked Single** propagation. This clears 100% deterministic cells instantly (`O(1)`), handling simple puzzles without neural overhead.

### 2. Neural Layer: Recurrent GATv2
If the puzzle remains unsolved, the board is converted into a **Heterogeneous Graph** (Nodes: Cells, Edges: Constraints).
* **Model:** Recurrent Graph Attention Network (GATv2).
* **Output:** A probability heatmap (`cached_probs`) representing the likelihood of each digit (1-9) for every empty cell.
* **Role:** Acts as a **Static Value Ordering Heuristic**.

### 3. Search Layer: Dynamic Variable Ordering
The system enters a recursive search phase with a smart strategy:
* **Variable Ordering (Logic):** We dynamically select the cell with the **Minimum Remaining Values (MRV)** (i.e., the "most constrained" cell).
* **Value Ordering (Intuition):** For the selected cell, we try candidate numbers in the order of **highest GNN confidence**.

---

## 🧬 Evolutionary Process (Research Log)

This project evolved through rigorous failure analysis:

1.  **Pure GCN (Static):**
    * *Attempt:* Treat Sudoku as a simple node classification task.
    * *Failure:* Achieved 99.9% cell accuracy but **0% board accuracy** on hard puzzles due to lack of global consistency check.
2.  **Naive Hybrid (Static Ordering):**
    * *Attempt:* Use GNN confidence to determine fill order (Static Variable Ordering).
    * *Failure:* On dynamic puzzles, the "hardest cell" shifts after every move. Static ordering led to suboptimal paths, taking **> 70 seconds** on extreme cases.
3.  **Dynamic Neuro-Symbolic (Final):**
    * *Solution:* Decoupled "Which cell to solve" (Logic/MRV) from "Which number to try" (Neural/GNN).
    * *Result:* Inference time dropped to **0.8s**, beating the symbolic baseline.

---

## 💻 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch & PyTorch Geometric

### Installation
```bash
git clone [https://github.com/YourUsername/NeuroSudoku.git](https://github.com/YourUsername/NeuroSudoku.git)
cd NeuroSudoku
pip install -r requirements.txt


### Train 
python experiments/train.py --epochs 10 --batch_size 64

### solve (extream case)
python main.py --input "800000000003600000070090200050007000000045700000100030001000068008500010090000400"

### run benchmarks : Compare Neuro-Symbolic vs Backtracking on hard cases.
python experiments/evaluate_hard.py --samples 200

### Directory Structure
```text
NeuroSudoku/
├── data/               # Dataset processing
├── models/             # GATv2 + GRU Architecture
├── solvers/            # Logic & Hybrid Solvers
│   ├── simple_propagation.py  # Constraint Propagation
│   └── optimized_solver.py # Final Master Solver
├── experiments/        # Training & Benchmarking Scripts
│   ├── train.py        # training model
│   └── evaluate .. .py # model evaluating 
├── utils/              # Graph conversion & Visualization
│   ├── graph_utils.py  # Graph conversion
│   └── visualize.py    # Visualization
└── main.py             # CLI Entry point
