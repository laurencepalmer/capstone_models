import torch 
import pandas as pd
import os
from torch.utils.data import Dataset
from typing import *

cd = os.getcwd()

class SequenceData(Dataset):
    """
    Class for easily loading sequential data
    """

    def __init__(self, y_col: str, x_col: List[str], window: int, data_path: str = None, df_arg: pd.DataFrame = None):
        """
        params
        ------
        y_col:: target column(s)
        x_col:: predictors
        window:: how long each subsequence is
        data_path:: where data lives

        """
        if data_path:
            df = pd.read_csv(data_path)
        else:
            df = df_arg

        self.features = x_col
        self.target = y_col
        self.window = window
        self.y = torch.tensor(df[y_col].values).float()
        self.X = torch.tensor(df[x_col].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.window - 1:
            i_start = i - self.window + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.window - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

class IdahoData(Dataset):
    """
    Class for loading the Idaho Dataset
    """
    
    def __init__(self, y_col: str, x_col: str, window: int, data_path: str):
        """
        Same as Sequence Data
        """
        self.y_col = y_col
        self.x_col = x_col
        self.window = window
        self.X = []
        self.y = []
        df = pd.read_csv(data_path)

        fund_names = df["Investment Name"].unique()

        for fund in fund_names:
            fund_spec_data = df[df["Investment Name"] == fund].copy()
            name = f"shifted_{y_col}"
            shifted_y = fund_spec_data[y_col].shift(1)
            fund_spec_data[name] = shifted_y
            fund_sequence = SequenceData(y_col, x_col + [name], window, data_path = None, df_arg = fund_spec_data[1:])
            for j in range(len(fund_sequence)):
                X, y = fund_sequence.__getitem__(j)
                self.X.append(X)
                self.y.append(y)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def prepare_data(data, forecast: int = 1, step: int = 1, window_len: int = 4):
    """
    Creates training data

    this will give a list of subseqences of the data 
    params
    ------
    forecast:: number of time steps out to forecast
    step:: number of time steps to take from the init point of the previous window
    window_len:: length of the sequence of values fed into the model
    values:: the data

    ret
    ------
    seq:: List[Tuple] contains the src and the tgt prediction

    """
    seq = []
    right_ind = window_len
    left_ind = 0
    while right_ind + forecast < len(data):
        src = data[left_ind:right_ind]
        tgt = data[left_ind:right_ind+forecast]

        left_ind+=step
        right_ind+= step
        seq.append((src, tgt))
    return seq
        