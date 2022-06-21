import re
import time
from torch.utils.data import Dataset
import torch

class DialogueDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, max_length=128):
        input_ids = [item[:max_length] for item in src_texts]
        labels = [item[:max_length] for item in tgt_texts]
        self.batch = {
            'input_ids': input_ids,
            'labels': labels
        }
        # if save:
        #     to_save = {
        #         'input_ids': input_ids[:10000],
        #         'labels': labels[:10000]
        #     }
        #     torch.save(to_save,'data/temp_pt_data.pt')

    def __len__(self):
        return len(self.batch['input_ids'])


    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = self.batch['labels'][index]
        labels = torch.tensor(labels,dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
#
class TempDataset(Dataset):
    def __init__(self, path, max_length=128):
        self.batch = torch.load(path)


    def __len__(self):
        return len(self.batch['input_ids'])


    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        labels = self.batch['labels'][index]
        labels = torch.tensor(labels,dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }
