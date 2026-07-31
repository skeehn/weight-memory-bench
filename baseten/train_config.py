"""Baseten training job: the weight-memory scaling curve at 1B, 3B, and 8B.

This is not a training job in the usual sense -- nothing is being fit for deployment. It is
a measurement that happens to need gradients, which is exactly what Baseten Training
provides: an arbitrary container on a GPU, billed per minute, with the source tree shipped
along.

**Why remote at all.** 1B and 3B run on a 16GB laptop. 8B does not: 16GB of bf16 weights
fits in neither the RAM nor the 13GB of free disk. And the three rungs must run on the same
hardware in one process, because comparing Apple Silicon against an H100 confounds scale
with dtype behaviour and kernel nondeterminism -- and the effect being measured is smaller
than that confound. So all three run here.

**Cost.** H100 at $0.10833/min against a $4.62 budget is 42 minutes total, for everything,
with no second account. `run.sh` carries a 25-minute hard timeout; this file requests one
GPU and nothing else. Checkpointing is off because there are no weights worth keeping --
the artifact is a JSON file of measurements.

    uvx truss train push baseten/train_config.py
    uvx truss train logs --job-id <id> --tail
"""

from truss.base.truss_config import AcceleratorSpec
from truss_train import (
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    TrainingJob,
    TrainingProject,
)

# CUDA 12.8 runtime with torch preinstalled, so the only pip work at start-up is
# transformers and peft. Every minute spent installing is a billed minute.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

runtime = Runtime(
    start_commands=["chmod +x ./baseten/run.sh && ./baseten/run.sh"],
    # Caches model weights between jobs. The three rungs pull ~25GB from the hub; on a
    # retry that download would otherwise be paid for twice.
    cache_config=CacheConfig(enabled=True),
    # No checkpoints: this job produces measurements, not weights. Syncing would add
    # billed minutes for artifacts nobody wants.
    checkpointing_config=CheckpointingConfig(enabled=False),
)

compute = Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1))

training_project = TrainingProject(
    name="weight-memory-scaling-curve",
    job=TrainingJob(image=Image(base_image=BASE_IMAGE), compute=compute, runtime=runtime),
)
