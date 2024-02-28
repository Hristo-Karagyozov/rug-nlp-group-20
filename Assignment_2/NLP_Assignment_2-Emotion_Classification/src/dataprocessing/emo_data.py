import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EmoData(Dataset):
    def __init__(self, csv_path, tokenizer, context_window):
        """
        Parameters
        ----------
        csv_path : str
        tokenizer :
        context_window : int
        """
        self.data = pd.read_csv(csv_path, delimiter=';')
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.article_ids = self.data.article_id
        self.essay = self.data.essay
        self.labels = self.data.emotion

    def __len__(self):
        return len(self.essay)

    def __getitem__(self, idx):
        phrase = str(self.essay[idx])
        ws_sep_phrase = " ".join(phrase.split())

        inputs = self.tokenizer.encode_plus(
            phrase,
            None,
            add_special_tokens=True,
            max_length=self.context_window,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = ['token_type_ids']
        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def labels(self):
        return self.labels().unique()
