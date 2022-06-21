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
    parser.add_argument('--vocab_path', default='/data/data_hub/HF_model/HF_BART_base', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='log/data_process.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--data_path', default='/zhengdong3/data/Pchatchit_splited', type=str, required=False, help='')
    parser.add_argument('--save_path', default='/zhengdong3/data/data_D_json', type=str, required=False, help='')
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


def preprocess(i, file_name, args):
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)
    start = time.time()

    src_ids = []
    tgt_ids = []
    file_path = os.path.join(args.data_path,file_name)
    with open(file_path,'r',errors='ignore') as f:
        lines = f.readlines()
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


def main_run():
    args = set_args()
    start_time = time.time()
    args.file_num = 0
    params = []

    print(f"from {args.data_path} to {args.save_path} ******************")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if os.path.isdir(args.data_path):
        for i, file in enumerate(os.listdir(args.data_path)):
            if file.split('.')[-1] != 'txt':
                continue
            args.file_num += 1
            params.append((i, file, args))
    else:
        raise Exception('data_path is not a dir')

    cores = mp.cpu_count() - 4
    pool = mp.Pool(cores)
    logger.info(
        f'the file @{args.file_num} files @ {args.data_path} process with {cores} processes, start processing......'
    )
    res = pool.imap(process_one_file, params)
    pool.close()
    # pool.join()
    for done_sig in enumerate(res):
        pass

    print("##" * 50)
    total_cost = time.time() - start_time
    print(f"done all..... total cost {total_cost}")
    print("##" * 50)


if __name__ == '__main__':
    main_run()

