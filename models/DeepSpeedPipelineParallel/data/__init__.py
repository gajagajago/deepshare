from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ClassificationDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer
    ):

        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data[index]
        tokenizer_result = self.tokenizer(
            str(data_row[0]) + self.tokenizer.eos_token + str(data_row[1]), padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        return dict(
            input_ids=tokenizer_result["input_ids"].flatten(),
            attention_masks=tokenizer_result["attention_mask"].flatten(),
            premise=data_row[0],
            hypothesis=data_row[1],
            labels=data_row[2]
        )

def collate_fn(batch):
    input_ids = torch.tensor([list(b['input_ids']) for b in batch])
    attention_masks = torch.tensor(
        [list(b['attention_masks']) for b in batch])
    labels = torch.tensor([b['labels'].as_py() for b in batch])
    return (tuple([input_ids, attention_masks]), labels)