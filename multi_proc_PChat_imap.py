import argparse
import os
import re
from tqdm import tqdm
import logging
import time
import multiprocessing as mp
from transformers  import BertTokenizer
import json

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='/data/data_hub/HF_model/HF_BART_large', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='log/preprocess.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--data_path', default='/data/data_hub/BART_trainset/PChatchit/PCbatchit_splited', type=str, required=False, help='')
    parser.add_argument('--save_path', default='/data/data_hub/BART_trainset/PChatchit/data_D_json_splited', type=str, required=False, help='')
    args = parser.parse_args()
    return args


def clean_anything(line):
    if isinstance(line,list):
        return [clean_anything(item) for item in line]
    trans = str.maketrans("！？＂〝〞“”‟＃＄％＆＇‘’‛（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～",
                          "!?\"\"\"\"\"\"#$%&''''()*+,-/:;<=>@[\]^_`{|}~")
    line = line.translate(trans)
    line = re.sub("""\\t2\\t\d+""", '', line)
    line = re.sub("""(\\t\d+)+""", '', line)
    return line


def process_one_file(params):
    return preprocess(params[0], params[1], params[2])


def preprocess(i, chunk_param, args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    start = time.time()

    src_ids = []
    tgt_ids = []

    with open(args.file_path) as f:
        f.seek(chunk_param[0])
        lines = f.read(chunk_param[1]).splitlines()
    for line in tqdm(lines):
        line = clean_anything(line)
        line = line.replace('\n', '').split('\t')
        if (len(line) != 2) or len(line[0]) < 10:
            continue
        src_ids.append(tokenizer.encode(line[0], truncation=True, max_length=128))
        tgt_ids.append(tokenizer.encode(line[1], truncation=True, max_length=128))
    train_tensors = {
        'input_ids': src_ids,
        'labels': tgt_ids
    }
    with open(os.path.join(args.save_path, ('data_split_' + str(i) + '.json')), 'w') as result_file:
        json.dump(train_tensors, result_file)

    cost = time.time()-start
    print(f"one process cost {cost} with file {i} DONE########################")
    del train_tensors
    return "DONE"


def chunkify(filename, size=1024):
    fileEnd = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        chunkEnd = f.tell()
        result = []
        while True:
            chunkStart = chunkEnd
            f.seek(size, 1)
            f.readline()
            chunkEnd = f.tell()
            result.append([chunkStart, chunkEnd - chunkStart])
            if chunkEnd > fileEnd:
                break
    return result


def m_proc_file(args):


def main_run():
    args = set_args()
    start_time = time.time()
    args.file_num = 0
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if os.path.isdir(args.data_path):
            m_proc_file(args)
    else:
        raise Exception('data_path is not dir')


    print("##" * 50)
    total_cost = time.time() - start_time
    print(f"done all..... total cost {total_cost}")
    print("##" * 50)


if __name__ == '__main__':
    main_run()

