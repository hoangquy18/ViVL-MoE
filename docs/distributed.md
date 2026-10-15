# Distributed training

`vivlmoe/training/distributed.py` builds a `DistContext(rank, local_rank, world_size, device,
backend)` from torchrun's `RANK / LOCAL_RANK / WORLD_SIZE` (`dist.backend: auto` → nccl on CUDA,
gloo on CPU; MPS is never used). Collectives used by the trainer and evaluators: `barrier`,
`all_reduce_sum_`, `all_reduce_mean`, `all_gather_varlen` (zero-length shards allowed),
`all_gather_object`, `broadcast_object`. No `nn.Module` ever calls a collective.

## DDP settings (`vivlmoe/training/ddp.py`)

`broadcast_buffers=False` (router biases stay identical through all-reduced updates),
`find_unused_parameters=False` (idle experts are always called on an empty tensor),
`gradient_as_bucket_view=True`, `bucket_cap_mb`, optional `static_graph`, optional bf16
gradient-compression hook (`dist.grad_compression_bf16`, nccl only). `sync_context(model, sync)`
wraps accumulation micro-batches in `no_sync`. `check_all_trainable_params_received_grad` is the
startup hard check.

## Seeding

`seed_everything(run.seed)` before model construction (identical init on all ranks; DDP then
broadcasts rank 0's weights), `seed_rank_streams` afterwards (`seed + 1_000_003 · rank`) so dropout
and sampling differ per rank. DataLoader workers are seeded by `vivlmoe.data.loaders.seed_worker`
(the single worker-seeding helper; `move_to_device` lives in the same module). The Trainer creates
the train iterator under `torch.random.fork_rng` so a mid-epoch resume reproduces the dropout stream.

## Samplers and evaluation

Train samplers implement `set_epoch(epoch)`, `skip_first` (samples skipped on the next `__iter__`,
reset to 0 afterwards) and equal per-rank lengths (drop_last). Eval loaders shard `rank::world`
without padding; evaluators call `gather_ordered` / `gather_ordered_tensors` on every rank (even with
an empty shard), which reassemble by uid and assert the ids are exactly `0..N-1`; rank 0 computes
the metrics and the dict is broadcast. `gather_ordered_tensors` moves its inputs to `ctx.device`
for the collective (NCCL cannot gather CPU tensors) and returns the result on the input's device;
`gather_ordered` pickles Python objects and is meant for small payloads (strings, ids), not
embeddings.

## Tests

`pytest -m gloo` spawns 2 gloo processes (`torch.multiprocessing.spawn`, `file://` init) to check the
collectives, ordered gather with an empty shard, and that DDP + `no_sync` accumulation produces
gradients identical across ranks and equal to a single-process reference.
