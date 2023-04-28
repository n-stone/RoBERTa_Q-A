import pandas as pd
import numpy as np
import torch
import config
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class QADataSet(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(config.ROBERTA_PATH)
        self.Q = [self.tokenizer(t, padding=config.PADDING, truncation=config.TRUNCATION, max_length=config.MAX_LEN).input_ids for t in self.df['Question'].to_list()]
        self.A = [self.tokenizer(t, padding=config.PADDING, truncation=config.TRUNCATION, max_length=config.MAX_LEN).input_ids for t in self.df['Answer'].to_list()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.Q[idx]
        answer = self.A[idx]
        return question, answer


if __name__ == "__main__":
  dataset = QADataSet()