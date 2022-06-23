from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.nn.optimizer import HybridAdam

# gradient_accumulation = 4
# clip_grad_norm = 2.0
BATCH_SIZE = 32
NUM_EPOCHS = 60
SEQ_LEN = 128

optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

# model = dict(
#     type=gpt2_small,
#     checkpoint=True,
# )

parallel = dict(
    pipeline=1,
    tensor=dict(size=1, mode=None),
)


zero = dict(
    model_config=dict(
        shard_strategy=TensorShardStrategy(),
        reduce_scatter_bucket_size_mb=25,
        fp32_reduce_scatter=False,
        tensor_placement_policy="cuda",
        gradient_predivide_factor=1.0,
        reuse_fp16_shard=False
    ),
    optimizer_config=dict(
        gpu_margin_mem_ratio=0.8,
        initial_scale=2**5,
        min_scale=1,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=1000,
        hysteresis=2,
        max_scale=2**32
    )
)