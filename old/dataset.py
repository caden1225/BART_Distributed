import os
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
class DialogueDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        src_texts = []
        tgt_texts = []
        with open(os.path.join(data_dir, split + '.tsv')) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            # for row in reader:
            for _, row in tqdm(enumerate(reader)):
                src_texts.append(row[0])
                tgt_texts.append(row[1])


        self.batch = tokenizer(
            src_texts, add_special_tokens=True, return_tensors='pt',
            max_length=max_length, padding='longest', truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts, add_special_tokens=True, return_tensors='pt',
                max_length=max_length, padding='longest', truncation=True
            )
        self.batch['labels'] = labels['input_ids']

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

if __name__ == "__main__":
    data_path = '/data/data_hub/BART_trainset/ED'
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_dataset = DialogueDataset(
        data_dir=data_path,
        split='train',
        tokenizer=tokenizer,
        max_length=128
    )