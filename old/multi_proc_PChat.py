import argparse
import os
import re
from tqdm import tqdm
import logging
from datetime import datetime
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
    parser.add_argument('--data_path', default='/data/data_hub/BART_trainset/PChatchit/raw_samples_20w.txt', type=str,
                        required=False, help='')
    parser.add_argument('--save_path', default='/data/data_hub/BART_trainset/PChatchit/data_D_json_old', type=str,
                        required=False, help='')
    parser.add_argument('--chunk_size', default=1024 * 1024 * 16 , type=int, required=False, help='chunk大小')
    parser.add_argument('--win_size', default=200, type=int, required=False, help='滑动窗口的大小，相当于每条数据的最大长度')
    parser.add_argument('--step', default=100, type=int, required=False, help='滑动窗口的滑动步幅')
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
    return preprocess(params[0], params[1])


def preprocess(chunk_param, args):
    # 20220510修改为CpmTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)

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

    return {
        'input_ids': src_ids,
        'labels': tgt_ids,
    }


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
    cores = mp.cpu_count() - 4
    pool = mp.Pool(cores)
    print(args.file_path)
    chunk_num = chunkify(args.file_path, args.chunk_size)
    logger.info(f'the file @{args.file_num}@ {args.file_path} need {len(chunk_num)} chunks, start processing......')
    start = datetime.now()
    params = []
    for _, item in enumerate(chunk_num):
        params.append((item, args))
    res = pool.map(process_one_file, params)
    pool.close()
    pool.join()

    cost = datetime.now() - start
    logger.info(f"translate the data used {cost}")
    logger.info("##" * 50)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_tensors = {
        'input_ids': [],
        'labels': []
    }
    for i, data_piece in tqdm(enumerate(res)):
        train_tensors['input_ids'].extend(data_piece['input_ids'])
        train_tensors['labels'].extend(data_piece['labels'])
        if i % 10 == 0:
            with open(os.path.join(args.save_path, ('data_split_' + str(i) + '.json')), 'w') as result_file:
                json.dump(train_tensors,result_file)
            train_tensors = {
                'input_ids': [],
                'labels': []
            }
            logger.info(f"done write the file with num {i}")
    with open(os.path.join(args.save_path, ('data_split_final.json')), 'w') as result_file:
        json.dump(train_tensors, result_file)
    del train_tensors
    logger.info(f"done write the file with num {args.file_num}")

def main_run():
    args = set_args()
    start_time = datetime.now()
    args.file_num = 0
    if os.path.isdir(args.data_path):
        for file in os.listdir(args.data_path):
            if file.split('.')[-1] != 'txt':
                continue
            args.file_num += 1
            args.file_path = os.path.join(args.data_path, file)
            m_proc_file(args)
    elif os.path.isfile(args.data_path):
        args.file_path = args.data_path
        m_proc_file(args)
    else:
        raise Exception('no data in data_path')

    print("##" * 50)
    total_cost = datetime.now() - start_time
    print(f"done all..... total cost {total_cost}")
    print("##" * 50)


if __name__ == '__main__':
    main_run()
