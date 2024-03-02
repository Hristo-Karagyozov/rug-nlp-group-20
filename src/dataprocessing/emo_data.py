from collections import namedtuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DataRet = namedtuple('DataRet', ('ids', 'attention_mask', 'token_type_ids', 'labels'))


class EmoData(Dataset):
    def __init__(self, csv_path, tokenizer, context_window, remove_neutral=False):
        """
        Parameters
        ----------
        csv_path : str
        tokenizer :
        context_window : int
        """
        data = pd.read_csv(csv_path, delimiter=';')
        if remove_neutral:
            data = data[data['emotion'] != 'neutral']
            data.reset_index(drop=True, inplace=True)
        self.data = data
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.article_ids = self.data.article_id
        self.essay = self.data.essay
        self.labels = self.data.emotion
        self.emotion_to_int = {emotion : idx for idx, emotion in enumerate(sorted(self.labels.unique()))}

    def __len__(self):
        return len(self.essay)

    def __getitem__(self, idx):
        phrase = str(self.essay[idx])
        ws_sep_phrase = " ".join(phrase.split())

        inputs = self.tokenizer.encode_plus(
            ws_sep_phrase,
            None,
            add_special_tokens=True,
            max_length=self.context_window,
            padding='max_length',
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return DataRet(
            torch.tensor(ids, dtype=torch.long,),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
            torch.tensor(self.emotion_to_int[self.labels[idx]], dtype=torch.long)
        )