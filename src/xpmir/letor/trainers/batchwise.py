import sys
import torch
import numpy as np
from typing import List, Union
from typing_extensions import TypedDict, ReadOnly
from experimaestro import Param, initializer

from datamaestro_ir.data.distillation import ListwiseDistillationSample
from xpm_torch.losses.batchwise import BatchwiseLoss
from xpm_torch.trainers import TrainerContext, LossTrainer

from xpmir.text import TokenizedTexts
from xpmir.letor.records import (
    PairwiseItem,
    ProductItems,
)
from .utils import wrap_collate_with_tokenizer

import logging

logger = logging.getLogger(__name__)


class BatchwiseInputs(TypedDict):
    records: ReadOnly[ProductItems]
    tokenized_records: ReadOnly[TokenizedTexts]


def batchwise_collate(
    records: List[Union[PairwiseItem, ListwiseDistillationSample]],
) -> BatchwiseInputs:
    """Collate PairwiseItem or ListwiseDistillationSample records into a ProductItems batch with in-batch negatives.

    Builds a relevance matrix where the first `batch_size` columns correspond to each query's
    positive document (diagonal = 1, off-diagonal = 0) and the remaining `batch_size * K` columns
    correspond to all mined hard negatives in the batch (relevance = 0).
    """
    if not records:
        raise ValueError("batchwise_collate received an empty records list.")

    batch_size = len(records)
    batch = ProductItems()
    positives = []
    negatives = []

    first = records[0]
    if isinstance(first, PairwiseItem):
        for record in records:
            batch.add_topics(record.query)
            positives.append(record.positive)
            negatives.append(record.negative)
        relevances = torch.cat(
            (torch.eye(batch_size), torch.zeros(batch_size, batch_size)), 1
        )
    elif isinstance(first, ListwiseDistillationSample):
        docs_per_query = len(first.documents)
        if docs_per_query < 2:
            raise ValueError(
                f"ListwiseDistillationSample in batchwise_collate must contain at least 2 documents "
                f"(1 positive + >=1 negative), got {docs_per_query}."
            )
        num_negatives = docs_per_query - 1
        for record in records:
            if len(record.documents) != docs_per_query:
                raise ValueError(
                    f"Inconsistent document count in batchwise_collate: expected {docs_per_query}, "
                    f"got {len(record.documents)}."
                )
            batch.add_topics(record.query)
            positives.append(record.documents[0].document)
            for neg_doc in record.documents[1:]:
                negatives.append(neg_doc.document)
        relevances = torch.cat(
            (
                torch.eye(batch_size),
                torch.zeros(batch_size, batch_size * num_negatives),
            ),
            1,
        )
    else:
        raise TypeError(
            f"Unsupported record type in batchwise_collate: {type(first).__name__}. "
            "Expected PairwiseItem or ListwiseDistillationSample."
        )

    batch.add_documents(*positives)
    batch.add_documents(*negatives)
    batch.set_relevances(relevances)
    return BatchwiseInputs(records=batch, tokenized_records=None)


class BatchwiseTrainer(LossTrainer):
    """Batchwise trainer

    Arguments:

    lossfn: The loss function to use
    sampler: A batchwise sampler
    """

    lossfn: Param[BatchwiseLoss]
    """A batchwise loss function"""

    @initializer
    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(context)

        dataset = self.sampler.as_dataset()

        collate_fn = wrap_collate_with_tokenizer(
            self.model, batchwise_collate, log=logger
        )
        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: BatchwiseInputs):
        batch = inputs["records"]
        tokenized_records = inputs.get("tokenized_records")

        # Get the next batch and compute the scores for each query/document
        if tokenized_records is not None:
            rel_scores = self.model(batch, tokenized=tokenized_records)
        else:
            rel_scores = self.model(batch)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        batch_scores = rel_scores.reshape(*batch.relevances.shape)
        self.lossfn.process(
            batch_scores, batch.relevances.to(batch_scores.device), self.context
        )
