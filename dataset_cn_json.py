from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch


class DialogueDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, max_length=128):
        input_ids = [item[:max_length] for item in src_texts]
        labels = [item[:max_length] for item in tgt_texts]
        print("##" * 50)
        _size = len(input_ids)
        print(f"may take a long time with {_size} items")

        input_ids = [torch.tensor(item, dtype=torch.long) for item in input_ids]
        attention_mask = [torch.ones_like(item, dtype=torch.long) for item in input_ids]
        labels = [torch.tensor(item, dtype=torch.long) for item in labels]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)

        self.batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        return len(self.batch['input_ids'])

    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        attention_mask = self.batch['attention_mask'][index]
        labels = self.batch['labels'][index]

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
