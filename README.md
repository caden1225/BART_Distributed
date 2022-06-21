# 单击单卡及分布式环境BART模型微调

## Overview

对话任务，包含PChat数据预处理，单机单卡训练脚本，单机多卡训练脚本，基于ColissalAI的单机多卡训练脚本，自定义的Dataset及其他一些utils脚本。 用于进行对话任务的微调训练。

### 1、数据预处理

- split_PChat_file将单个大文本数据文件且分为多个文件，方便后面并行及tokenize,当有需要切分的单个大数据文件时可以使用;
- muliti_proc_*文件为多进程的数据预处理脚本，使用multi_proc_PChat_files.py;因数据量过大，采用先进行tokenzie的方式处理，具体存储格式及special_token添加等操作可在此处完成;

### 2、单机单卡训练脚本测试

 首先尝试在单机单卡环境下完成基础训练逻辑开发，dataset、dataloader等加载与epoch循环等，train_zh_single_json.py为本次使用的脚本;

### 3、修改单击单卡环境至单机多卡

**train_dist.py为使用pytorch.distributed及spawm实现的原生分布式训练脚本，需要修改的all_reduce地方较多，暂时弃用;**
- 本次使用基于ColossalAI实现的分布式训练，脚本文件为train_colossal_*;
- train_colossal_amp.py为开启amp及混合精度的脚本，需要安装Nvidia的Apex;

## 踩坑方法

1、通过数据预处理脚本完成数据准备;
2、尝试单机单卡训练脚本调通
3、修改ColossalAI的多卡训练脚本：
   - 添加config.py的配置文件设置任务及框架相关参数，包括BATCH_SIZE,NUM_EPOCHS,fp16,gradient_accumulation等
   - 修改train_colossal_*的训练脚本
   - 修改train_colossal_*的相关参数
   - 通过命令行启动训练任务（坑爹框架现在只能通过命令行启动，又增加了使用AIStation时候的坑数） ：（

### 涉及参数
- --device        '设置使用哪些显卡'
- --epochs        '训练的最大轮次'
- --batch_size        '训练的batch size'
- --log_step      '多少步汇报一次loss'
- --local_rank        '分布式训练的GPU对应的进程号',这个参数默认-1,必须得有
- --model_config      'init模型的初始化配置文件，从头训练时使用'
- --data_path      '训练集路径'
- --save_model_path
- --pretrained_model
- --log_path      '日志保存路径'
- --tb_log_dir        'tensorboard训练日志路径'
- --max_length         '单个样本最大长度'
- --lr         '学习率'
- --adam_eps      '学习率更新参数，有学习率的scheduler时候使用'
- --warmup_steps       ‘scheduler更新步长’
- --label_smoothing        '平滑损失函数的参数，fairseq用的BART的损失函数'
- --patience      ‘earlystoping的参数'
- --vocab_path        '词表路径'
- --val_rate      'float验证集比例'
- --num_workers        'dataloader加载数据时使用的线程数量"

### 训练启动命令
```bash
colossalai run --nproc_per_node 8 train_colossal_engine.py
# nproc_per_node为但机卡数
或者
python -m torch.distributed.launch --nproc_per_node 2 train_colossal_amp.py
```

## TODO
- 添加多机多卡的训练脚本
- 修改数据加载的方式为bin-idx方式以释放cpu内存对数据加载量的限制