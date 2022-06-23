import logging
import os
import time
import json
from torch.nn.utils.rnn import pad_sequence
from dataset_cn_json import DialogueDataset



def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_date = time.strftime('%Y%m%d', time.localtime(time.time()))
    file_handler = logging.FileHandler(
        filename=os.path.join(args.log_path, 'train_log_' + file_date + '.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def collate_fn(batch):

    input_ids = []
    labels = []
    for item in batch:
        input_ids.append(item['input_ids'])
        labels.append(item['labels'])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return (input_ids, labels)


def load_dataset(logger, args):

    logger.info("loading training dataset and validating dataset")
    input_ids = []
    labels = []
    file_num = 0
    started = time.time()
    if os.path.isdir(args.data_path):
        for file in os.listdir(args.data_path)[:1]:
            if file.split('.')[-1] != 'json':
                continue
            with open(os.path.join(args.data_path, file), 'r', errors='ignore') as f:
                json_list = json.load(f)
                input_ids.extend(json_list['input_ids'][:10000])
                labels.extend(json_list['labels'][:10000])
            file_num += 1
            logger.info(f"loaded {file_num} file in process~~")
        logger.info(f"total load {file_num} files")
    elif os.path.isfile(args.data_path):
        with open(args.data_path) as f:
            json_list = json.load(f)
            input_ids.extend(json_list['input_ids'])
            labels.extend(json_list['labels'])
    else:
        raise Exception('no data in data_path')

    total_num = len(input_ids)
    logger.info(f"completed loading the data to list with {total_num} items")

    if args.val_rate is not None:
        val_rate = args.val_rate
        val_num = round(total_num * val_rate)
    else:
        val_num = args.val_num
    cost = time.time() - started
    logger.info(f"load data file cost: {cost:.4f}")
    validate_dataset = DialogueDataset(input_ids[:val_num], labels[:val_num], args.max_length)
    train_dataset = DialogueDataset(input_ids[val_num:], labels[val_num:], args.max_length)

    return validate_dataset, train_dataset

