# -*- coding: utf-8 -*-
# @Time    : 2022/6/19 上午11:32
# @Author  : caden1225
# @File    : multi_proc_100w.py
# @Description : 说明一下
import argparse
import pickle
import re
from tqdm import tqdm
import logging
import numpy as np
from transformers import CpmTokenizer
import argparse
import os
import re
from tqdm import tqdm
import logging
import time
import multiprocessing as mp
from transformers  import BertTokenizer
import json


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='/data/data_hub/HF_model/HF_BART_base', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='log/data_process.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--train_path', default='/data/data_hub/100w_闲聊数据/train_100w.txt', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--data_path', default='/zhengdong3/data/Pchatchit_splited', type=str, required=False, help='')
    parser.add_argument('--save_path', default='/data/data_hub/100w_tokens.json', type=str, required=False, help='')
    args = parser.parse_args()
    return args


def preprocess():
    args = set_args()
    # 初始化日志对象
    logger = create_logger(args.log_path)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    start = time.time()

    # sep_id = tokenizer.sep_token_id
    # start_id = tokenizer.bos_token_id
    logger.info("preprocessing data,data path:{}, save path:{}".format(args.train_path, args.save_path))

    # 读取训练数据集
    with open(args.train_path, 'rb') as f:
        data = f.read().decode("utf-8")
    # 需要区分linux和windows环境下的换行符
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")
    else:
        train_data = data.split("\n\n")
    logger.info("there are {} dialogue in dataset".format(len(train_data)))

    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    src_ids = []
    tgt_ids = []
    with open(args.save_path, "w", encoding="utf-8") as f:
        for index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            context = re.sub('<s>','',utterances[0])
            for utterance in utterances[1:]:
                utterance = re.sub('<s>','',utterance)
                src_ids.append(tokenizer.encode(context, truncation=True, max_length=128))
                tgt_ids.append(tokenizer.encode(utterance, truncation=True, max_length=128))
                context = utterance
    train_tensors = {
        'input_ids': src_ids,
        'labels': tgt_ids
    }
    with open(args.save_path, 'w') as result_file:
        json.dump(train_tensors, result_file)
    logger.info("finish preprocessing data,the result is stored in {}".format(args.save_path))
    cost = time.time()-start
    # print(f"one process cost {cost} with file {i} DONE########################")

if __name__ == '__main__':
    preprocess()

