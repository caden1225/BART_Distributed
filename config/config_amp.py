from colossalai.amp import AMP_TYPE

BATCH_SIZE = 32
DROP_RATE = 0.1
NUM_EPOCHS = 500

fp16=dict(
    mode=AMP_TYPE.APEX,
)

# gradient_accumulation = 4
# clip_grad_norm = 2.0