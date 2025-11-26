# 📐 Theoretical Approach: Evolution of the Neuro-Symbolic Solver

## 1. Problem Statement: The Limits of Pure Deep Learning

Sudoku is formally defined as a **Constraint Satisfaction Problem (CSP)**. While it is often used as a toy example for backtracking algorithms, solving "Extreme" (NP-Hard) instances requires navigating a massive search space.

### 1.1. The Hypothesis
Our initial hypothesis was that a **Recurrent Graph Neural Network (GNN)** could learn the underlying logical rules of Sudoku purely from data distribution, effectively acting as a differentiable solver.

### 1.2. The Failure of Pure Connectionism (Phase 1)
We treated Sudoku as a node classification task.
* **Model:** GCN + GRU (32 recurrence steps).
* **Observation:** The model achieved **99.9% cell-level accuracy** on the test set.
* **Critical Flaw:** Despite high accuracy, the **Board-level Accuracy on hard puzzles was near 0%**.
* **Analysis:** The model suffered from **"Hallucination"**. It learned local correlations (e.g., "if neighbors are 1 and 2, I am likely 3") but failed to enforce strict global constraints. A single error in an 81-cell grid renders the solution invalid. Deep Learning provides *approximations*, but logic puzzles require *exactness*.

---

## 2. The Hybrid Approach: Integrating Logic and Intuition

To address the limitations of pure DL, we adopted a **Neuro-Symbolic** approach, treating the GNN not as a solver, but as a **Heuristic Oracle**.

### 2.1. The Bottleneck of Static Ordering (Phase 2)
We initially implemented a hybrid solver where the GNN determined the search order before the search began (**Static Variable Ordering**).
* **Mechanism:** Sort all empty cells by GNN confidence and fill them sequentially.
* **Result:** On Arto Inkala's "AI Escargot", the solver took **> 70 seconds**, performing worse than standard backtracking.
* **Root Cause Analysis:** In CSPs, the "most constrained variable" (the hardest cell to fill) changes dynamically as the board is updated. A static plan derived from the initial state becomes obsolete after a few moves, leading the solver down deep, incorrect search branches.

### 2.2. The Solution: Dynamic Variable Ordering (Phase 3 - Final)
We decoupled the roles of "Logic" and "Intuition":

| Component | Role | Method |
| :--- | :--- | :--- |
| **Logic (Symbolic)** | Decides **WHERE** to look. | **Minimum Remaining Values (MRV)** Heuristic. |
| **Intuition (Neural)** | Decides **WHAT** to try. | **GNN Probability Distribution**. |

This architecture mimics human expert reasoning: *Look for the spot with the fewest options (Logic), and if there's ambiguity, trust your gut feeling (Intuition).*



---

## 3. System Architecture

### 3.1. Graph Representation
We model the Sudoku grid as a graph $G = (V, E)$.
* **Nodes ($V$):** 81 cells. Input features are one-hot encoded digits (or zero vector for empty).
* **Edges ($E$):** Undirected edges connecting cells that share a constraint (Row, Column, $3 \times 3$ Box). This results in a dense graph where each node has 20 neighbors.

### 3.2. Neural Model: Recurrent GATv2
Standard GCNs treat all neighbors equally (Isotropic). However, in Sudoku, a specific neighbor (e.g., a "5" in the same row) provides a stronger constraint than an empty neighbor.
* **Attention Mechanism:** We employ **GATv2 (Graph Attention Network v2)** with 4 heads to allow the model to dynamically weigh the importance of neighbor constraints.
* **Recurrence:** A GRU cell updates the node embeddings iteratively ($t=32$), simulating the "diffusion" of logical constraints across the board.

### 3.3. Optimization: One-Shot Inference
To eliminate the overhead of running a neural network inside a recursive loop, we perform **One-Shot Inference**.
1.  **Pre-computation:** Run the GNN **once** on the initial board to generate a $81 \times 9$ probability map.
2.  **Cached Heuristic:** This map is cached and used as a static lookup table for **Value Ordering** during the search.
3.  **Result:** This reduced the inference overhead from $O(N)$ (linear to search depth) to $O(1)$, achieving **0.8s** runtime on extreme cases.

---

## 4. Performance Summary

The final Dynamic Neuro-Symbolic solver demonstrates robust performance across distributions.

* **Robustness:** Unlike Backtracking, which exhibits high variance (solving some hard puzzles instantly and others in minutes), our solver maintains a consistent runtime.
* **Efficiency:** By using GNN probabilities to sort candidate values, we significantly reduce the **Branching Factor** of the search tree.

> **Conclusion:** This project proves that Deep Learning can effectively augment symbolic algorithms for NP-hard problems, provided it is used to **guide** the search (Heuristic) rather than **replace** it.