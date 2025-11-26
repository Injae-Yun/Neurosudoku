import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.graph_utils import get_sudoku_edges, board_to_tensor

class SudokuDataset(Dataset):
    def __init__(self, root, csv_path=None, transform=None, pre_transform=None):
        self.csv_path = csv_path
        # 원본 CSV를 메모리에 로드 (Pandas DataFrame은 100만 개도 수백 MB 수준이라 괜찮음)
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        # 엣지 인덱스는 정적이므로 미리 한 번만 계산
        self.edge_index = get_sudoku_edges()
        
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['sudoku.csv']

    @property
    def processed_file_names(self):
        # 처리된 파일을 저장하지 않으므로 빈 리스트 반환
        return []

    def download(self):
        pass

    def process(self):
        # 사전 처리 과정 없음 (On-the-fly 변환)
        pass

    def len(self):
        return len(self.df)

    def get(self, idx):
        # 요청받은 idx의 데이터만 그때그때 변환
        row = self.df.iloc[idx]
        quiz_str = row['quizzes']
        sol_str = row['solutions']
        
        x = board_to_tensor(quiz_str).unsqueeze(1)
        y = board_to_tensor(sol_str) - 1
        mask = (x.squeeze() == 0)
        
        # edge_index는 공유 (메모리 절약)
        data = Data(x=x, y=y, edge_index=self.edge_index, train_mask=mask)
        
        return data