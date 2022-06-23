from colossalai.amp import AMP_TYPE

BATCH_SIZE = 32
DROP_RATE = 0.1
NUM_EPOCHS = 500
TENSOR_PARALLEL = 4

fp16=dict(
    mode=AMP_TYPE.NAIVE,
)

# gradient_accumulation = 4
# clip_grad_norm = 2.0

parallel = dict(
    pipeline=1,
    tensor=dict(size=TENSOR_PARALLEL, mode='2d')
)