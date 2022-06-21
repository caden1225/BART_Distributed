#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 下午1:37
# @Author  : caden1225
# @File    : train_single.py
# @Description : BART_分布式训练
import argparse
import logging
import os
import random
import torch
import torch.multiprocessing as mp
import time
import numpy as np
import csv
import torch.distributed as dist
import torch.utils.data.distributed
from torch.backends import cudnn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import log_softmax
from dataset_cn import DialogueDataset, clean_anything
from tqdm import tqdm
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup
)
from pytorchtools import EarlyStopping

from transformers.models.bart.modeling_bart import shift_tokens_right


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--epochs', default=15, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='训练的batch size')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')

    parser.add_argument('--data_path', default='data/', type=str, required=False, help='训练集路径')
    parser.add_argument('--save_model_path', default='model_debug', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--vocab_path', default='vocab/cpm_vocab.model', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--model_config', default='config/raw_BART_config.json', type=str, required=False,
                        help='设置模型参数')
    parser.add_argument('--tb_log_dir', default='tb_debug/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--log_path', default='log', type=str, required=False, help='训练日志存放位置')

    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--adam_betas', default='(0.9,0.999)')
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=500, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int)
    parser.add_argument('--gradient_clip_val', default=0.1, type=float)
    parser.add_argument('--patience', default=0, type=int)

    parsed_args = parser.parse_args()
    return parsed_args


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


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''From fairseq'''
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def load_dataset(logger, tokenizer, args):
    logger.info("loading training dataset and validating dataset")
    src_list = []
    tgt_list = []
    file_num = 0
    if os.path.isdir(args.data_path):
        for file in os.listdir(args.data_path):
            if file.split('.')[-1] != 'txt':
                continue
            with open(os.path.join(args.data_path, file), 'r', errors='ignore') as f:
                csv.field_size_limit(500 * 1024 * 1024)
                reader = csv.reader(f)
                for _, row in tqdm(enumerate(reader)):
                    dialogue = clean_anything(row)
                    dialogue = dialogue[0].split('\t')
                    if not len(dialogue) == 2:
                        continue
                    src_list.append(dialogue[0])
                    tgt_list.append(dialogue[1])
            file_num += 1
        logger.info(f"total load {file_num} files")
    elif os.path.isfile(args.data_path):
        with open(args.data_path) as f:
            csv.field_size_limit(500 * 1024 * 1024)
            reader = csv.reader(f)
            for _, row in tqdm(enumerate(reader)):
                dialogue = clean_anything(row)
                dialogue = dialogue[0].split('\t')
                if not len(dialogue) == 2:
                    continue
                src_list.append(dialogue[0])
                tgt_list.append(dialogue[1])
    else:
        raise Exception('no data in data_path')

    total_num = len(src_list)
    logger.info(f"completed loading the data to list with {total_num} items")

    if args.val_rate is not None:
        val_rate = args.val_rate
        val_num = round(total_num * val_rate / 1000) * 1000
    else:
        val_num = args.val_num
    validate_dataset = DialogueDataset(src_list[:val_num], tgt_list[:val_num], tokenizer, args.max_length)
    train_dataset = DialogueDataset(src_list[val_num:], tgt_list[val_num:], tokenizer, args.max_length)
    val_sampler = torch.utils.data.distributed.DistributedSampler(validate_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    return validate_dataset, train_dataset, val_sampler, train_sampler


def train_epoch(model, train_loader, optimizer, criterion, scheduler, logger, epoch, args, local_rank):
    pad_token_id = args.pad_token_id
    tb_writer = SummaryWriter(log_dir=args.tb_log_dir)

    model.train()
    epoch_start = time.time()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].cuda(local_rank, non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(local_rank, non_blocking=True)
        labels = batch['labels'].cuda(local_rank, non_blocking=True)
        decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id=args.sep_token_id)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        logits = outputs.logits
        lprobs = log_softmax(logits, dim=-1)
        loss, nll_loss = criterion(
            lprobs=lprobs,
            target=labels,
            epsilon=args.label_smoothing,
            ignore_index=pad_token_id
        )

        # todo 平均多GPu的loss
        loss = reduce_mean(loss, args.nprocs)
        total_loss += loss.item()
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        del input_ids, attention_mask, labels, outputs
        if ((batch_idx + 1) % args.log_step == 0) & (local_rank == 0):
            global overstep
            tb_writer.add_scalar('train_loss', loss.item(), global_step=overstep)
            logger.info(f" current totalstep is {overstep} step")
            logger.info(f" current train_loss is {loss:.6f}")

        if local_rank == 0:
            overstep += 1

    epoch_loss = total_loss / len(train_loader)

    if local_rank == 0:
        epoch_cost = time.time() - epoch_start
        logger.info("############# epoch {}: loss {} #############".format(epoch, epoch_loss))
        model_path = os.path.join(args.save_model_path, 'epoch{}'.format(epoch))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch))
        logger.info('time for one epoch: {}'.format(epoch_cost))

    return epoch_loss


def valid_epoch(model, validate_loader, criterion, logger, epoch, args, local_rank):
    if local_rank == 0:
        logger.info("validating stage")
    pad_token_id = args.pad_token_id
    tb_writer = SummaryWriter(log_dir=args.tb_log_dir)

    model.eval()
    valid_start = time.time()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(validate_loader):
            input_ids = batch['input_ids'].cuda(local_rank, non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(local_rank, non_blocking=True)
            labels = batch['labels'].cuda(local_rank, non_blocking=True)

            decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id=args.sep_token_id)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            lprobs = log_softmax(logits, dim=-1)
            loss, nll_loss = criterion(
                lprobs=lprobs,
                target=labels,
                epsilon=args.label_smoothing,
                ignore_index=pad_token_id
            )

            # todo
            # loss = reduce_mean(loss, args.nprocs)
            total_loss += loss.item()
            del input_ids, attention_mask, labels, outputs

        if args.local_rank == 0:
            valid_loss = total_loss / len(validate_loader)
            valid_cost = time.time() - valid_start
            logger.info(
                "validate epoch {}: loss {}".format(epoch + 1, valid_loss))
            logger.info('time for validating one epoch: {}'.format(valid_cost))

        return valid_loss


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

        logging.warning('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:2345',
                            world_size=args.nprocs,
                            rank=local_rank)
    # 创建日志对象
    logger = create_logger(args)

    if args.pretrained_model:  # 加载预训练模型
        model = BartForConditionalGeneration.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model_config = BartConfig.from_json_file(args.model_config)
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large',
                                                             config=model_config)
        print(model_config)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    logger.info("use GPU {} training".format(args.device))

    tokenizer = BartTokenizer.from_pretrained(args.vocab_path)
    args.pad_token_id = tokenizer.pad_token_id
    args.sep_token_id = tokenizer.sep_token_id

    # todo
    # criterion = label_smoothed_nll_loss.cuda(local_rank)
    cudnn.benchmark = True

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    if local_rank == 0:
        # 计算模型参数数量
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info('number of model parameters: {}'.format(num_parameters))
        # 记录参数设置
        logger.info("args:{}".format(args))

    # ========= Loading Dataset ========= #
    validate_dataset, train_dataset, val_sampler, train_sampler = load_dataset(logger, tokenizer, args)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    criterion = label_smoothed_nll_loss
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_eps)
    t_total = len(train_loader) // args.accumulate_grad_batches * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    logger.info('starting training')

    train_losses, validate_losses = [], []
    best_val_loss = 1000
    overstep = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model, train_loader=train_loader,
            optimizer=optimizer, criterion=criterion, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args, local_rank=local_rank
        )
        train_losses.append(train_loss)

        validate_loss = valid_epoch(
            model=model, validate_loader=validate_loader, criterion=criterion,
            logger=logger, epoch=epoch, args=args, local_rank=local_rank
        )
        validate_losses.append(validate_loss)


        if (validate_loss < best_val_loss) & (args.local_rank == 0):
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch))
            ppl_model_path = os.path.join(args.save_model_path, 'min_ppl_model'.format(epoch))
            if not os.path.exists(ppl_model_path):
                os.mkdir(ppl_model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(ppl_model_path)

            if args.patience == 0:
                continue

        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        logger.info(f"training finished with total step {overstep}")
        logger.info("train_loss:{}".format(np.mean(train_losses)))
        logger.info("validate_loss:{}".format(np.mean(validate_loss)))


if __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICEDS"] = args.device
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
