import sys
from typing import List
from typing_extensions import ReadOnly, TypedDict
import torch
from torch import nn, Tensor
from experimaestro import Config, Param, field
import torch.nn.functional as F
from xpmir.text import TokenizedTexts
from xpmir.letor.records import (
    PointwiseItem,
    PointwiseItems,
)
from xpm_torch.trainers import TrainerContext, LossTrainer
from xpm_torch.losses import Loss, ModuleOutputType, bce_with_logits_loss

from .samplers import ListwiseDistillationSample
import numpy as np
from xpmir.rankers import AbstractModuleScorer
from xpmir.letor.trainers.utils import wrap_collate_with_tokenizer

import logging

logger = logging.getLogger(__name__)

### Losses


class DistillationListwiseLoss(Config, nn.Module):
    """The abstract loss for listwise distillation"""

    weight: Param[float] = field(default=1.0, ignore_default=True)
    NAME = "?"

    def initialize(self, ranker: AbstractModuleScorer):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"listwise-{self.NAME}", loss, self.weight))

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


class DistillRankNetLoss(DistillationListwiseLoss):
    """Adaptation of the pairwise RankNET loss to lists of passages
    ranked by a LLM. Follows Rank-DistiLLM: Closing the Effectiveness Gap
    Between Cross-Encoders and LLMs for Passage Re-Ranking, 2025
    """

    NAME = "DistillRankNET"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits

    @staticmethod
    def get_pairwise_idcs(targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Get pairwise indices for positive and negative samples based on targets.

        Function copied from the official implementation of Rank-DistiLLM:
        https://github.com/webis-de/lightning-ir/blob/main/lightning_ir/loss/base.py#L131
        """
        # positive items are items where label is greater than other label in sample
        return torch.nonzero(targets[..., None] > targets[:, None], as_tuple=True)

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the DistillRankNet loss

        Arguments:

            student_scores: A (batch x num_docs) tensor
            teacher_scores: A (batch x num_docs) tensor
        """
        query_idcs, pos_idcs, neg_idcs = self.get_pairwise_idcs(teacher_scores)
        pos = student_scores[query_idcs, pos_idcs]
        neg = student_scores[query_idcs, neg_idcs]
        margin = pos - neg
        loss = self.loss(margin, torch.ones_like(margin))
        return loss


class ADR_MSE(DistillationListwiseLoss):
    """New loss to distill from lists of passages ranked by LLM, proposed
    by Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and
    LLMs for Passage Re-Ranking, 2025
    """

    NAME = "ADR-MSE"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.MSELoss(reduction="none")
        self.discount = "log2"
        self.temperature = 1

    @staticmethod
    def get_approx_ranks(scores: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute approximate ranks from scores.
        Function copied from the official implementation of Rank-DistiLLM:
        https://github.com/webis-de/lightning-ir/blob/main/lightning_ir/loss/approximate.py#L34

        """
        score_diff = scores[:, None] - scores[..., None]
        normalized_score_diff = torch.sigmoid(score_diff / temperature)
        # set diagonal to 0
        normalized_score_diff = normalized_score_diff * (
            1 - torch.eye(scores.shape[1], device=scores.device)
        )
        approx_ranks = normalized_score_diff.sum(-1) + 1
        return approx_ranks

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the ADR-MSE loss

        Arguments:

            student_scores: A (batch x num_docs) tensor
            teacher_scores: A (batch x num_docs) tensor
        """
        student_ranks = self.get_approx_ranks(student_scores, self.temperature)
        # teacher ranks are integer (Long) after argsort; cast to student's dtype/device
        teacher_ranks = (
            torch.argsort(torch.argsort(teacher_scores, descending=True)) + 1
        )
        teacher_ranks = teacher_ranks.to(
            dtype=student_ranks.dtype, device=student_ranks.device
        )

        loss = self.loss(student_ranks, teacher_ranks)
        if self.discount == "log2":
            weight = 1 / torch.log2(teacher_ranks + 1)
        else:
            weight = 1
        loss = loss * weight
        loss = loss.mean()
        return loss


class ListwiseSoftmaxCrossEntropy(DistillationListwiseLoss):
    """Reproduces the original `SoftmaxCrossEntropy` behavior used in
    batchwise losses, adapted to listwise distillation.

    The original formula is:
      `-logsumexp(normalize(scores) + (1 - 1.0 / relevances), dim=-1).mean()`

    where `normalize` depends on the model output type.
    """

    NAME = "infonce"

    temperature: Param[float] = field(default=1.0, ignore_default=True)
    """Temperature scaling factor applied to logits before softmax"""

    def __validate__(self):
        if self.temperature <= 0:
            raise ValueError(
                f"ListwiseSoftmaxCrossEntropy temperature must be > 0, got {self.temperature}"
            )

    def initialize(self, ranker: AbstractModuleScorer):
        super().initialize(ranker)
        if self.temperature <= 0:
            raise ValueError(
                f"ListwiseSoftmaxCrossEntropy temperature must be > 0, got {self.temperature}"
            )
        self.normalize = {
            ModuleOutputType.REAL: lambda x: F.log_softmax(x / self.temperature, -1),
            ModuleOutputType.LOG_PROBABILITY: lambda x: x,
            ModuleOutputType.PROBABILITY: lambda x: x.log(),
        }[ranker.outputType]

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        # teacher_scores used as "relevances" in the original formula.
        # Guard against zeros to avoid division-by-zero.
        eps = 1e-8
        rel = teacher_scores.clone()
        rel = torch.where(
            rel == 0, torch.tensor(eps, device=rel.device, dtype=rel.dtype), rel
        )

        term = self.normalize(student_scores) + (1.0 - 1.0 / rel)
        # sum over documents, mean over queries
        loss = -torch.logsumexp(term, dim=-1).sum() / student_scores.shape[0]
        return loss


class ListwiseInfoNCE(DistillationListwiseLoss):
    """Standard InfoNCE loss for listwise supervised training.

    This loss expects binary relevance labels (1 for positive, 0 for negative).
    If multiple positives are present, it averages the cross-entropy loss over them.
    TODO CHECK
    """

    NAME = "listwise-infonce"

    def initialize(self, ranker: AbstractModuleScorer):
        super().initialize(ranker)
        self.normalize = {
            ModuleOutputType.REAL: lambda x: F.log_softmax(x, -1),
            ModuleOutputType.LOG_PROBABILITY: lambda x: x,
            ModuleOutputType.PROBABILITY: lambda x: x.log(),
        }[ranker.outputType]

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        # teacher_scores are binary (1 for positive, 0 for negative)
        log_probs = self.normalize(student_scores)

        # Binary mask for positives
        is_positive = (teacher_scores > 0).float()

        # Number of positives per query
        num_positives = is_positive.sum(dim=-1, keepdim=True)

        # Target distribution: uniform over positives
        targets = is_positive / torch.clamp(num_positives, min=1.0)

        # Cross entropy: -sum(targets * log_probs)
        # This computes the average log-probability of positive documents
        loss = -(targets * log_probs).sum(dim=-1)

        # Mask out queries with no positives to avoid contributing to the mean
        mask = (num_positives > 0).float().squeeze(-1)
        loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        return loss


class ListwiseBCE(DistillationListwiseLoss):
    """Point-wise cross-entropy loss for listwise samples.
    Computes BCE for each document in the list, adapting to the student's output type.
    """

    NAME = "bce"

    def initialize(self, ranker: AbstractModuleScorer):
        super().initialize(ranker)
        self.output_type = ranker.outputType

        if self.output_type == ModuleOutputType.REAL:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.output_type == ModuleOutputType.PROBABILITY:
            self.loss_fn = nn.BCELoss()
        elif self.output_type == ModuleOutputType.LOG_PROBABILITY:
            # Using your custom autograd function for log-space stability
            self.loss_fn = bce_with_logits_loss
        else:
            raise NotImplementedError(f"Output type {self.output_type} not supported.")

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        # student_scores: (Batch, NumPassages)
        # teacher_scores: (Batch, NumPassages) - binary 1s and 0s

        # Ensure targets match the student's precision and device
        targets = teacher_scores.to(student_scores.dtype)

        if self.output_type == ModuleOutputType.LOG_PROBABILITY:
            # Your custom BCEWithLogLoss expects vectors (1D tensors)
            # Flattening ensures compatibility regardless of batch size/list length
            return self.loss_fn(student_scores.flatten(), targets.flatten())

        # For standard nn.Modules, (Batch, NumPassages) is handled automatically
        return self.loss_fn(student_scores, targets)


class ListwiseHingeLoss(DistillationListwiseLoss):
    """Pairwise Hinge loss for listwise samples.
    Computes max(0, margin - (s_pos - s_neg)) for each negative.

    This implementation assumes a fixed number of positives per query
    (typically 1) as provided by the DistillationNegativesSampler.
    """

    NAME = "hinge"
    margin: Param[float] = field(default=1.0, ignore_default=True)

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        # teacher_scores are binary (1 for positive, 0 for negative)
        is_positive = teacher_scores > 0

        # We assume that each query has at least one positive and one negative
        # DistillationNegativesSampler gives exactly 1 pos and passages_per_query - 1 negs.

        # Flattening and reshaping is risky unless shapes are guaranteed.
        # If using fixed 1-pos, many-neg:
        batch_size = student_scores.shape[0]
        pos_scores = student_scores[is_positive].reshape(batch_size, -1)  # (B, P)
        neg_scores = student_scores[~is_positive].reshape(batch_size, -1)  # (B, N)

        # (B, num_pos, 1) - (B, 1, num_neg) -> (B, num_pos, num_neg)
        loss = F.relu(self.margin - pos_scores.unsqueeze(2) + neg_scores.unsqueeze(1))
        return loss.mean()


### Trainer


class DistillationListwiseInputs(TypedDict):
    records: ReadOnly[PointwiseItems]
    tokenized_records: ReadOnly[TokenizedTexts]
    teacher_scores: ReadOnly[Tensor]


def distillation_listwise_collate(
    samples: List[ListwiseDistillationSample],
) -> DistillationListwiseInputs:
    """Collate function for Distillation Listwise trainer"""
    teacher_scores = torch.empty(len(samples), len(samples[0].documents))
    records = PointwiseItems()
    for ix, sample in enumerate(samples):
        for doc in sample.documents:
            records.add(PointwiseItem(sample.query, doc.document, doc.score))
        teacher_scores[ix] = torch.tensor([doc.score for doc in sample.documents])

    return DistillationListwiseInputs(
        records=records, tokenized_records=None, teacher_scores=teacher_scores
    )


class DistillationListwiseTrainer(LossTrainer):
    """Listwise trainer for distillation"""

    lossfn: Param[DistillationListwiseLoss]
    """The distillation pairwise batch function"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.model)
        for loss in context.hooks(DistillationListwiseLoss):
            loss.initialize(self.model)

        self.sampler.initialize(random)

        dataset = self.sampler.as_dataset()

        collate_fn = wrap_collate_with_tokenizer(
            self.model, distillation_listwise_collate, log=logger
        )
        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: DistillationListwiseInputs):
        # Builds records and teacher score matrix
        records, teacher_scores = inputs["records"], inputs["teacher_scores"]
        tokenized_records = inputs.get("tokenized_records", None)

        # Get the next batch and compute the scores for each query/document
        if tokenized_records is not None:
            scores = self.model(records, tokenized=tokenized_records)
        else:
            scores = self.model(records)

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            self.logger.error(
                "nan or inf relevance score detected. Aborting (listwise distillation)."
            )
            sys.exit(1)

        # Call the losses (distillation, pairwise and pointwise)
        teacher_scores = teacher_scores.to(scores.device)
        self.lossfn.process(
            scores.reshape_as(teacher_scores), teacher_scores, self.context
        )
