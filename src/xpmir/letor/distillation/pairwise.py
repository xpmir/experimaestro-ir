import sys
from typing import List, TypedDict
from typing_extensions import ReadOnly
import numpy as np
import torch
from torch import nn
from torch.functional import Tensor
from experimaestro import Config, Param, field
from xpmir.letor.records import (
    PairwiseItem,
    PairwiseItems,
)
from xpm_torch.trainers import TrainerContext, LossTrainer
from xpm_torch.losses import Loss

from .samplers import PairwiseDistillationSample

from xpmir.text import TokenizedTexts
from xpmir.rankers import AbstractModuleScorer
from xpmir.letor.trainers.utils import wrap_collate_with_tokenizer


import logging

logger = logging.getLogger(__name__)


class DistillationPairwiseLoss(Config, nn.Module):
    """The abstract loss for pairwise distillation"""

    weight: Param[float] = field(default=1.0, ignore_default=True)
    NAME = "?"

    def initialize(self, ranker: AbstractModuleScorer):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"pairwise-{self.NAME}", loss, self.weight))

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

            student_scores: A (batch x 2) tensor
            teacher_scores: A (batch x 2) tensor
        """
        raise NotImplementedError()


class MSEDifferenceLoss(DistillationPairwiseLoss):
    """Computes the MSE between the score differences

    Compute ((student 1 - student 2) - (teacher 1 - teacher 2))**2
    """

    NAME = "delta-MSE"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.MSELoss()

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        return self.loss(
            student_scores[:, 1] - student_scores[:, 0],
            teacher_scores[:, 1] - teacher_scores[:, 0],
        )


class DistillationKLLoss(DistillationPairwiseLoss):
    """
    Distillation loss from: Distilling Dense Representations for Ranking using
    Tightly-Coupled Teachers https://arxiv.org/abs/2010.11386
    """

    NAME = "Distil-KL"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.KLDivLoss(reduction="none")

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        pos_student = student_scores[:, 0].unsqueeze(0)
        neg_student = student_scores[:, 1].unsqueeze(0)
        pos_teacher = teacher_scores[:, 0].unsqueeze(0)
        neg_teacher = teacher_scores[:, 1].unsqueeze(0)

        scores = torch.cat([pos_student, neg_student], dim=1)
        local_scores = torch.log_softmax(scores, dim=1)
        teacher_scores = torch.cat(
            [pos_teacher.unsqueeze(-1), neg_teacher.unsqueeze(-1)], dim=1
        )
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        return self.loss(local_scores, teacher_scores).sum(dim=1).mean(dim=0)


class DistillationPairwiseInputs(TypedDict):
    """A record with just a text item"""

    records: ReadOnly[PairwiseItems]
    tokenized_records: ReadOnly[TokenizedTexts]
    teacher_scores: ReadOnly[Tensor]


def distillation_pairwise_collate(
    samples: List[PairwiseDistillationSample],
) -> DistillationPairwiseInputs:
    """Collate function for Distillation Pairwise trainer
    Args:
        samples: List of pairwise distillation samples
        transform_records: A function to transform the records before feeding them to the model.
    """
    teacher_scores = torch.empty(len(samples), 2)
    records = PairwiseItems()
    for ix, sample in enumerate(samples):
        records.add(
            PairwiseItem(
                sample.query,
                sample.documents[0].document,  # positive
                sample.documents[1].document,  # negative
            )
        )
        teacher_scores[ix, 0] = sample.documents[0].score
        teacher_scores[ix, 1] = sample.documents[1].score

    return DistillationPairwiseInputs(
        records=records, tokenized_records=None, teacher_scores=teacher_scores
    )


class DistillationPairwiseTrainer(LossTrainer):
    """Pairwise trainer for distillation"""

    lossfn: Param[DistillationPairwiseLoss]
    """The distillation pairwise batch function"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.model)
        for loss in context.hooks(DistillationPairwiseLoss):
            loss.initialize(self.model)

        self.sampler.initialize(random)

        dataset = self.sampler.as_dataset()

        collate_fn = wrap_collate_with_tokenizer(
            self.model, distillation_pairwise_collate, log=logger
        )
        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: DistillationPairwiseInputs):
        # Builds records and teacher score matrix
        records, teacher_scores = inputs["records"], inputs["teacher_scores"]
        tokenized_records = inputs.get("tokenized_records")

        # Get the next batch and compute the scores for each query/document pair
        if tokenized_records is not None:
            scores = self.model(tokenized=tokenized_records).reshape(2, len(records)).T
        else:
            scores = self.model(records).reshape(2, len(records)).T

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            self.logger.error(
                "nan or inf relevance score detected. Aborting (pairwise distillation)."
            )
            sys.exit(1)

        # Call the losses (distillation, pairwise and pointwise)
        teacher_scores = teacher_scores.to(
            scores.device
        )  # no op with fabric but ensures that the teacher scores are on the same device as the student scores
        self.lossfn.process(scores, teacher_scores, self.context)
