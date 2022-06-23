#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 下午1:37
# @Author  : caden1225
# @File    : train_single.py
# @Description : colossalai training in distributed_amp_mode
import colossalai
import os
import torch
import time
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import log_softmax
from transformers import (
    BartConfig,
    BartForConditionalGeneration)
from transformers.models.bart.modeling_bart import shift_tokens_right
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader
from colossalai.nn.lr_scheduler import LinearWarmupLR
from utils_dev import load_dataset, collate_fn

def get_time_stamp():
    torch.cuda.synchronize()
    return time.time()

def set_args():
    parser = colossalai.get_default_parser()
    # parser.add_argument('--use_trainer', action='store_true', help='whether use trainer to execute the training')
    # parser.add_argument('--local_rank', default=-1, type=int, required=False, help='分布式训练的GPU对应的进程号')

    # parser.add_argument('--model_config', default='config/raw_BART_config.json', type=str, required=False,
    #                     help='设置模型参数')
    parser.add_argument('--data_path', default='/zhengdong3/data/data_D_json_10files', type=str, required=False, help='训练集路径')
    parser.add_argument('--colossal_config', default='/zhengdong3/projects/BART_Distributed_multi/config/config_amp.py', type=str, required=False, help='训练集路径')
    parser.add_argument('--save_model_path', default='/zhengdong3/projects/BART_Distributed_multi/model_colossal_dist', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='/zhengdong3/pretrained_model/min_ppl_model', type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--log_path', default='/zhengdong3/projects/BART_Distributed_multi/log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--tb_log_dir', default='/zhengdong3/projects/BART_Distributed_multi/tb_log', type=str, required=False, help='tensorboard训练日志存放位置')
    parser.add_argument('--log_steps', default=50, type=int, required=False, help='日志输出的步长间隔')

    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--adam_eps', default=1e-8, type=float)
    parser.add_argument('--warmup_steps', default=100, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    parser.add_argument('--vocab_path', default='/zhengdong3/pretrained_model/HF_BART_large', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--val_rate', type=float, default=0.01, help='验证集比例')
    parser.add_argument('--num_workers', type=int, default=1, required=False, help="dataloader加载数据时使用的线程数量")

    args = parser.parse_args()
    return args


def scaled_all_reduce(tensors):
    gpus = dist.get_world_size()
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    for reduction in reductions:
        reduction.wait()
    for tensor in tensors:
        tensor.mul_(1.0 / gpus)
    return tensors


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=0):
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

    nll_loss = nll_loss.mean()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def main():
    args = set_args()
    colossalai.launch_from_torch(config=args.colossal_config)

    logger = get_dist_logger()
    logger.log_to_file(path=args.log_path)
    logger.info("initialized distributed environment", ranks=[0])
    tb_writer = SummaryWriter(log_dir=args.tb_log_dir)

    if args.pretrained_model:  # 加载预训练模型
        model = BartForConditionalGeneration.from_pretrained(args.pretrained_model)
        logger.info(f"using the pre-trained_model training", ranks=[0])
        logger.info("#"*50)
    else:  # 初始化模型
        model_config = BartConfig.from_json_file(args.model_config)
        model = BartForConditionalGeneration(config=model_config)
        logger.info(f"using the initial model training", ranks=[0])

    validate_dataset, train_dataset = load_dataset(logger, args)
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=gpc.config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )
    validate_loader = get_dataloader(
        dataset=validate_dataset,
        batch_size=gpc.config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    args.pad_token_id = 0
    args.sep_token_id = 102
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_eps)
    criterion = label_smoothed_nll_loss
    # total_steps = len(train_loader) // gpc.config.gradient_accumulation * gpc.config.NUM_EPOCHS
    total_steps = len(train_loader) * gpc.config.NUM_EPOCHS
    scheduler = LinearWarmupLR(optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps)

    engine, train_loader, validate_loader, _ = colossalai.initialize(
        model,
        optimizer,
        criterion,
        train_dataloader=train_loader,
        test_dataloader=validate_loader,
    )
    if gpc.get_global_rank() == 0:
        logger.info(f"start training with total step is {total_steps} with {gpc.config.NUM_EPOCHS} epochs")

    best_val_loss = 10000
    overstep = 0
    for epoch in range(gpc.config.NUM_EPOCHS):
        started = get_time_stamp()
        train_losses = []
        engine.train()
        step_loss = []
        for batch in train_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            decoder_input_ids = shift_tokens_right(labels, pad_token_id=args.pad_token_id,
                                                   decoder_start_token_id=args.sep_token_id)

            engine.zero_grad()
            outputs = engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )
            logits = outputs.logits
            lprobs = log_softmax(logits, dim=-1)
            train_loss, nll_loss = engine.criterion(
                lprobs=lprobs,
                target=labels,
                epsilon=args.label_smoothing,
                ignore_index=args.pad_token_id
            )
            reduced_loss,reducded_nll_loss = scaled_all_reduce([train_loss, nll_loss])
            train_losses.append(reduced_loss.item())
            step_loss.append(reduced_loss.item())
            engine.backward(train_loss)
            engine.step()
            scheduler.step()

            if gpc.get_global_rank() == 0:
                overstep += 1
                if overstep % args.log_steps == 0:
                    step_loss = np.mean(step_loss)
                    logger.info(
                        f"##### step {overstep} in epoch {epoch}: avg_train_loss is {step_loss:.5}, nnl_loss is {nll_loss:.5}, "
                        f"##### current the reduced_loss is {reduced_loss} and {reducded_nll_loss} as nll_loss, "
                        f"##### lr is {scheduler.get_last_lr()[0]:.10g} #############",
                        ranks=[0])
                    tb_writer.add_scalar('train_loss', train_loss.item(), global_step=overstep)
                    tb_writer.add_scalar('avg_train_loss', step_loss, global_step=overstep)
                    tb_writer.add_scalar('nll_loss', nll_loss.item(), global_step=overstep)
                    step_loss = []
        avg_loss = np.mean(train_losses)
        if gpc.get_global_rank() == 0:
            avg_step_cost = (get_time_stamp() - started) / len(train_loader)
            logger.info(f"@@@@@ epoch {epoch}: train average loss is {avg_loss:.5f}, avg step time: {avg_step_cost:.4f} s",
                        ranks=[0])
            model_path = os.path.join(args.save_model_path, 'epoch{}'.format(epoch))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        engine.eval()
        val_losses = []
        for batch in validate_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            decoder_input_ids = shift_tokens_right(labels, pad_token_id=args.pad_token_id,
                                                   decoder_start_token_id=args.sep_token_id)

            with torch.no_grad():
                engine.zero_grad()
                outputs = engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                logits = outputs.logits
                lprobs = log_softmax(logits, dim=-1)
                val_loss, nll_loss = engine.criterion(
                    lprobs=lprobs,
                    target=labels,
                    epsilon=args.label_smoothing,
                    ignore_index=args.pad_token_id
                )
                reduced_loss, reducded_nll_loss = scaled_all_reduce([val_loss, nll_loss])
                val_losses.append(reduced_loss.item())
        epoch_cost = get_time_stamp() - started
        avg_val_loss = np.mean(val_losses)
        tb_writer.add_scalar('avg_val_loss', avg_val_loss, global_step=epoch)
        logger.info(
            f"@@@@@ Epoch {epoch} - step {overstep} val loss: {avg_val_loss:.5}, "
            f"lr current is {scheduler.get_last_lr()[0]:.5g}, "
            f"total epoch cost: {epoch_cost:.2g} s",
            ranks=[0]
        )

        if gpc.get_global_rank() == 0:
            if avg_val_loss < best_val_loss:
                logger.info(f"@@@@@@ saving the bset_val_loss_model from {epoch} @@@@@@",
                            ranks=[0])
                model_path = os.path.join(args.save_model_path, 'min_ppl_model')
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(model_path)
    gpc.destroy()

if __name__ == "__main__":
    main()
# colossalai run --nproc_per_node 8 train_colossal_amp.py
# python -m torch.distributed.launch --nproc_per_node 2 train_colossal_amp.py