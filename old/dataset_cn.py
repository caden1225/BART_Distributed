import re
import time
from torch.utils.data import Dataset
import torch


class DialogueDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length, save=False):
        started = time.time()
        self.batch = tokenizer(
            src_texts, add_special_tokens=True, return_tensors='pt',
            max_length=max_length, padding='longest', truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts, add_special_tokens=True, return_tensors='pt',
                max_length=max_length, padding='longest', truncation=True
            )
        print("processing the tokenize, it may take a long time when data is large ")
        self.batch['labels'] = labels['input_ids']
        cost = time.time() - started
        print(f"tokenized the dataset cost: {cost:.6f}")


    def __len__(self):
        return self.batch['input_ids'].size(0)

    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        attention_mask = self.batch['attention_mask'][index]
        labels = self.batch['labels'][index]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def clean_anything(line):
    if isinstance(line,list):
        return [clean_anything(item) for item in line]
    trans = str.maketrans("！？＂〝〞“”‟＃＄％＆＇‘’‛（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～",
                          "!?\"\"\"\"\"\"#$%&''''()*+,-/:;<=>@[\]^_`{|}~")
    line = line.translate(trans)
    line = re.sub("""\\t2\\t\d+""", '', line)
    line = re.sub("""(\\t\d+)+""", '', line)
    return line
