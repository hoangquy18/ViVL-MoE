"""The SyntheticTask conforms to the Task / Evaluator / sampler contracts used by the Trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from tests.unit.training.synthetic_task import ShardSampler, SyntheticTask, make_cfg
from vivlmoe.tasks.base import LoaderBundle, Task
from vivlmoe.tasks.evaluators.base import Evaluator, gather_ordered
from vivlmoe.training.distributed import DistContext
from vivlmoe.training.state import EvalContext, StepContext, StepOutput

CTX = DistContext.single_process("cpu")


def test_task_protocol(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, data={"batch_size": 4})
    task = SyntheticTask(cfg, CTX, n_train=12, n_dev=5)
    assert isinstance(task, Task)
    model = task.build_model()
    assert isinstance(model, nn.Module) and task.always_frozen() == ["frozen"]
    bundle = task.build_loaders()
    assert isinstance(bundle, LoaderBundle) and set(bundle.evals) == {"dev"}
    sampler = bundle.train.sampler
    assert hasattr(sampler, "set_epoch") and sampler.skip_first == 0 and len(sampler) == 12
    batch = next(iter(bundle.train))
    assert batch["x"].shape == (4, 8) and isinstance(batch["uids"], list)
    out = task.train_step(model, batch, StepContext(step=0, micro=0, accum=1, device=CTX.device))
    assert isinstance(out, StepOutput) and torch.isfinite(out.loss) and out.batch_size == 4
    ev = task.evaluators()["dev"]
    assert isinstance(ev, Evaluator)
    model.eval()
    with torch.no_grad():
        res = ev.run(model, bundle.evals["dev"], EvalContext(CTX, 0, tmp_path, "dev"))
    assert set(res) == {"mse", "score"} and res["score"] == -res["mse"]
    assert task.selection_metric() == ("dev/score", "max")
    task.load_state_dict({"micro_counter": 7})
    assert task.state_dict() == {"micro_counter": 7}


def test_shard_sampler_protocol() -> None:
    s0, s1 = ShardSampler(10, 2, 0, seed=3), ShardSampler(10, 2, 1, seed=3)
    for s in (s0, s1):
        s.set_epoch(1)
    a, b = list(s0), list(s1)
    assert len(a) == len(b) == 5 and not set(a) & set(b) and sorted(a + b) == list(range(10))
    s0.set_epoch(2)
    assert list(s0) != a
    s0.set_epoch(1)
    s0.skip_first = 2
    assert list(s0) == a[2:] and s0.skip_first == 0
    assert list(s0) == a  # skip consumed


def test_gather_ordered_single_process() -> None:
    assert gather_ordered([2, 0, 1], ["c", "a", "b"], CTX) == ["a", "b", "c"]
    assert gather_ordered([], [], CTX) == []
