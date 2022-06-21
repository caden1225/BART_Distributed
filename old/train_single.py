#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 下午1:37
# @Author  : caden1225
# @File    : train_single.py
# @Description : 单GPu训练脚本
import argparse
import logging
import os
import torch
import time
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import log_softmax
from old.dataset import DialogueDataset
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

    args = parser.parse_args()
    return args


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


def train_epoch(model, train_loader, optimizer, criterion, scheduler, logger, epoch, args):
    pad_token_id = args.pad_token_id
    tb_writer = SummaryWriter(log_dir=args.tb_log_dir)

    model.train()
    epoch_start = time.time()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id=2)

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

        total_loss += loss.item()
        # if args.gradient_accumulation_steps > 1:
        #     loss = loss / args.gradient_accumulation_steps
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        del input_ids, attention_mask, labels, outputs
        if (batch_idx + 1) % args.log_step == 0:
            global overstep
            overstep += 1
            tb_writer.add_scalar('train_loss', loss.item(), global_step=overstep)
            logger.info(f" current totalstep is {overstep} step")
            logger.info(f" current train_loss is {loss:.6f}")

    epoch_loss = total_loss / len(train_loader)
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


def valid_epoch(model, validate_loader, criterion, logger, epoch, args):
    pad_token_id = args.pad_token_id
    tb_writer = SummaryWriter(log_dir=args.tb_log_dir)

    model.eval()
    valid_start = time.time()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(validate_loader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            decoder_input_ids = shift_tokens_right(labels, pad_token_id, decoder_start_token_id=2)

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

            del input_ids, attention_mask, labels, outputs
            if (batch_idx + 1) % args.log_step == 0:
                global overstep
                tb_writer.add_scalar('val_loss', loss.item(), global_step=overstep)
                print(f"########## current totalstep is {overstep}")
                overstep += 1

        epoch_loss = total_loss / len(validate_loader)
        valid_cost = time.time() - valid_start
        logger.info(
            "validate epoch {}: loss {}".format(epoch + 1, epoch_loss))
        logger.info('time for validating one epoch: {}'.format(valid_cost))

        return epoch_loss


if __name__ == '__main__':
    args = set_args()
    logger = create_logger(args)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    if args.pretrained_model:  # 加载预训练模型
        model = BartForConditionalGeneration.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        # model_config = BartConfig.from_json_file(args.model_config)
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large',
            config=BartConfig())
        print(BartConfig())
    logger.info("use GPU {} training".format(args.device))
    model.cuda()

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))
    logger.info("args:{}".format(args))

    train_dataset = DialogueDataset(
        data_dir=args.data_path,
        split='train',
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    valid_dataset = DialogueDataset(
        data_dir=args.data_path,
        split='val',
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    validate_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    criterion = label_smoothed_nll_loss
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_eps)
    t_total = len(train_loader) // args.accumulate_grad_batches * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    train_losses, validate_losses = [], []
    best_val_loss = 1000
    overstep = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model=model, train_loader=train_loader,
            optimizer=optimizer, criterion=criterion, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args
        )
        train_losses.append(train_loss)

        validate_loss = valid_epoch(
            model=model, validate_loader= validate_loader,
            criterion=criterion, logger=logger, epoch=epoch, args=args
        )
        validate_losses.append(validate_loss)

        # if (validate_loss < best_val_loss) & (args.local_rank == 0):
        if (validate_loss < best_val_loss):
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch))
            model_path = os.path.join(args.save_model_path, 'min_ppl_model'.format(epoch))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
        logger.info(f"training finished with total step {overstep}")
        logger.info("train_loss:{}".format(np.mean(train_losses)))
        logger.info("validate_loss:{}".format(np.mean(validate_loss)))
