#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/22 2:43 PM
# @Author  : caden1225
# @File    : config.py
# @Description :    colossalai training config file
from colossalai.amp import AMP_TYPE

BATCH_SIZE = 8
DROP_RATE = 0.1
NUM_EPOCHS = 10

fp16=dict(
    mode=AMP_TYPE.TORCH
)

# gradient_accumulation = 8
clip_grad_norm = 1.0