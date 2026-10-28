"""Pointwise distillation sampler and trainer for Experimaestro-IR."""

from typing import List, TypedDict, Optional
from typing_extensions import ReadOnly
from pathlib import Path
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from experimaestro import Param, field, Config
from xpm_torch.trainers import LossTrainer, TrainerContext
from xpm_torch.losses import Loss
from xpm_torch.datasets import ShardedIterableDataset, InfiniteDataset
from xpm_torch.base import Sampler
from datamaestro_ir.data.distillation import (
    PointwiseDistillationSample,
    PointwiseDistillationSamples,
)
from xpmir.letor.records import PointwiseItems, PointwiseItem
from xpmir.rankers import AbstractModuleScorer
from xpmir.letor.trainers.utils import wrap_collate_with_tokenizer

logger = logging.getLogger(__name__)


class PointwiseDistillationInputs(TypedDict):
    records: ReadOnly[PointwiseItems]
    teacher_scores: ReadOnly[torch.Tensor]
    tokenized_records: ReadOnly[dict]


def pointwise_distillation_collate(
    samples: List[PointwiseDistillationSample],
) -> PointwiseDistillationInputs:
    records = PointwiseItems()
    teacher_scores = []
    for sample in samples:
        records.add(
            PointwiseItem(sample.query, sample.document.document, sample.document.score)
        )
        teacher_scores.append(sample.document.score)
    return PointwiseDistillationInputs(
        records=records,
        teacher_scores=torch.tensor(teacher_scores, dtype=torch.float32),
        tokenized_records=None,
    )


class PreShuffledPointwiseDataset(ShardedIterableDataset):
    """Yields pointwise distillation samples directly from a pre-shuffled Arrow dataset.

    Inherits elastic, O(1) multi-GPU sharding, state tracking, and logging directly from
    ShardedIterableDataset.
    """

    def __init__(
        self,
        hf_data,
        sample_builder,
        global_batch_size: int = 1,
        context: Optional[TrainerContext] = None,
    ):
        self.hf_data = hf_data
        self.sample_builder = sample_builder
        builder_fn = (
            sample_builder._build_sample if sample_builder is not None else None
        )
        super().__init__(
            source=hf_data,
            sample_builder=builder_fn,
            global_batch_size=global_batch_size,
        )
        if context is not None and getattr(context, "state", None) is not None:
            self.set_resume_step(getattr(context.state, "steps", 0))


class PointwiseDistillationSampler(Sampler):
    """Sampler wrapper for pointwise distillation datasets"""

    dataset: Param[PointwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def as_dataset(
        self,
        global_batch_size: Optional[int] = None,
        context: Optional[TrainerContext] = None,
    ) -> ShardedIterableDataset:
        from datasets import load_from_disk

        # Experimaestro might pass local_path as a string or Path
        local_path = getattr(self.dataset, "local_path", None)
        if local_path and (Path(local_path) / "dataset_info.json").exists():
            # The dataset was saved by our task using save_to_disk()
            hf_dataset = load_from_disk(str(local_path))
        else:
            # Fall back to default load_dataset logic
            hf_dataset = self.dataset.data

        return InfiniteDataset(
            PreShuffledPointwiseDataset(
                hf_dataset,
                self.dataset,
                global_batch_size=global_batch_size or 1,
                context=context,
            )
        )


class DistillationPointwiseLoss(Config, nn.Module):
    """The abstract loss for pointwise distillation"""

    weight: Param[float] = field(default=1.0, ignore_default=True)
    NAME = "?"

    def initialize(self, ranker: AbstractModuleScorer):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"pointwise-{self.NAME}", loss, self.weight))

    def compute(
        self,
        student_scores: Tensor,
        teacher_scores: Tensor,
        context: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        """Compute the loss

        Arguments:
            student_scores: A batch tensor of student predictions
            teacher_scores: A batch tensor of teacher scores
        """
        raise NotImplementedError()

    def forward(
        self,
        student_scores: Tensor,
        teacher_scores: Tensor,
        context: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        return self.compute(student_scores, teacher_scores, context)


class PointwiseMSELoss(DistillationPointwiseLoss):
    """Plain pointwise MSE loss between student and teacher scores"""

    NAME = "MSE"

    teacher_scale: Param[Optional[float]] = field(default=None, ignore_default=True)
    """Optional multiplicative scale applied to teacher scores before MSE computation"""

    teacher_bias: Param[Optional[float]] = field(default=None, ignore_default=True)
    """Optional additive bias applied to teacher scores before MSE computation"""

    def initialize(self, ranker: AbstractModuleScorer):
        super().initialize(ranker)
        self.loss = nn.MSELoss()

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> Tensor:
        if self.teacher_scale is not None or self.teacher_bias is not None:
            scale = self.teacher_scale if self.teacher_scale is not None else 1.0
            bias = self.teacher_bias if self.teacher_bias is not None else 0.0
            teacher_scores = teacher_scores * scale + bias
        return self.loss(student_scores.view(-1), teacher_scores.view(-1))


class PointwiseDistillationTrainer(LossTrainer):
    """Trainer for pointwise distillation"""

    lossfn: Param[DistillationPointwiseLoss]
    """The distillation pointwise batch function"""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.lossfn.initialize(self.model)
        for loss in context.hooks(DistillationPointwiseLoss):
            loss.initialize(self.model)

        self.sampler.initialize(random)

        dataset = self.sampler.as_dataset(
            global_batch_size=self.batch_size, context=context
        )

        collate_fn = wrap_collate_with_tokenizer(
            self.model, pointwise_distillation_collate, log=logger
        )
        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: PointwiseDistillationInputs):
        records = inputs["records"]
        teacher_scores = inputs["teacher_scores"]
        tokenized_records = inputs.get("tokenized_records")

        if tokenized_records is not None:
            student_scores = self.model(records, tokenized=tokenized_records)
        else:
            student_scores = self.model(records)

        if torch.isnan(student_scores).any() or torch.isinf(student_scores).any():
            self.logger.error(
                "nan or inf relevance score detected. Aborting (pointwise distillation)."
            )
            sys.exit(1)

        teacher_scores = teacher_scores.to(student_scores.device)
        self.lossfn.process(student_scores, teacher_scores, self.context)
