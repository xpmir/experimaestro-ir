from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Any
from attrs import evolve
import torch
from experimaestro import DataPath, OptionalDataPath, field, Param
from datamaestro_ir.data import IDTextRecord
from xpmir.neural import DualRepresentationScorer, QueriesRep, DocsRep

from xpmir.text.encoders import TextEncoderBase
from xpm_torch.learner import TrainerContext
from xpm_torch.losses import Loss
from xpm_torch.module import ModuleLoader
from xpm_torch.trainers import TrainingHook
from xpm_torch.metrics import ScalarMetric

import logging

logger = logging.getLogger(__name__)


class DualVectorListener(TrainingHook):
    """Listener called with the (vectorial) representation of queries and
    documents

    The hook is called just after the computation of documents and queries
    representations.

    This can be used for logging purposes, but more importantly, to add
    regularization losses such as the :class:`FlopsRegularizer` regularizer.
    """

    def __call__(
        self, context: TrainerContext, queries: torch.Tensor, documents: torch.Tensor
    ):
        """Hook handler

        Args:
            context (TrainerContext): The training context
            queries (torch.Tensor): The query vectors
            documents (torch.Tensor): The document vectors

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualVectorScorerListener(TrainingHook, ABC):
    """Listener called with the (vectorial) representation of queries and
    documents

    The hook is called just after the computation of documents and queries
    representations.

    This can be used for logging purposes, but more importantly, to add
    regularization losses such as the :class:`FlopsRegularizer` regularizer.
    """

    @abstractmethod
    def __call__(
        self,
        context: TrainerContext,
        queries: torch.Tensor,
        documents: torch.Tensor,
        scorer: torch.Tensor,
    ):
        """Hook handler

        Args:
            context (TrainerContext): The training context queries
            (torch.Tensor): The query vectors documents (torch.Tensor): The
            document vectors scores (torch.Tensor): A vector or matrix of scores
            (depending on how the scores were computed)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualModuleLoader(ModuleLoader):
    """ModuleLoader for dual encoder models.

    Has distinct ``encoder_path`` and ``query_encoder_path`` DataPaths so each
    encoder is serialized independently. This enables proper
    sentence-transformers format on HF Hub export (symmetric vs
    router/asymmetric).
    """

    encoder_path: Param[DataPath]
    """Path to the document encoder checkpoint"""

    query_encoder_path: OptionalDataPath = None
    """Path to the query encoder checkpoint (if separate from doc encoder)"""

    def execute(self):
        self.value.initialize()
        self.value.encoder.load_model(Path(self.encoder_path))
        if self.query_encoder_path and self.value.query_encoder is not None:
            self.value._query_encoder.load_model(Path(self.query_encoder_path))


class DualVectorScorer(DualRepresentationScorer[QueriesRep, DocsRep]):
    """A scorer based on dual vectorial representations"""

    CONFIG_LOADER = DualModuleLoader.C

    encoder: Param[TextEncoderBase[Any, Any]]
    """The document (and potentially query) encoder"""

    query_encoder: Param[Optional[TextEncoderBase[Any, Any]]]
    """The query encoder (optional, if not defined uses the query_encoder)"""

    def __initialize__(self):
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()

    @property
    def _query_encoder(self):
        return self.query_encoder or self.encoder

    def __validate__(self):
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _has_separate_query_model(self) -> bool:
        """Check if query and doc encoders are separate models.
        Default: object identity check. Override for more specific logic.
        """
        return self.query_encoder is not None and self.query_encoder is not self.encoder

    def export_action(self, loader, **kwargs):
        from xpmir.models import XPMIRExportAction

        if self.doc:
            kwargs.setdefault("doc", self.doc)
        if self.bibtex:
            kwargs.setdefault("bibtex", self.bibtex)
        return XPMIRExportAction.C(loader=loader, **kwargs)

    def loader_config(self, path: Path, *, settings=None) -> DualModuleLoader:
        has_separate_query = self._has_separate_query_model()
        return self.CONFIG_LOADER(
            value=self,
            encoder_path=path / "encoder",
            query_encoder_path=(path / "query_encoder" if has_separate_query else None),
            settings=settings,
        )


class Dense(DualVectorScorer[QueriesRep, DocsRep]):
    """A scorer based on a pair of (query, document) dense vectors"""

    def score_product(self, queries, documents):
        info = TrainerContext.current()
        # Gather the representations across all processes
        if info is not None and info.fabric is not None:
            fabric = info.fabric
            q_all = fabric.all_gather(queries.value, sync_grads=True)
            d_all = fabric.all_gather(documents.value, sync_grads=True)
        else:
            # We should always have access to fabric (even not in trainer mode)
            q_all = queries.value
            d_all = documents.value

        scores = q_all @ d_all.T

        if info is not None:
            for hook in info.hooks(DualVectorListener):
                hook(info, queries, documents)
            for hook in info.hooks(DualVectorScorerListener):
                hook(info, queries, documents, scores)

        return scores

    def score_pairs(self, queries, documents):
        scores = (
            (queries.value.unsqueeze(1) @ documents.value.unsqueeze(2))
            .squeeze(-1)
            .squeeze(-1)
        )

        # Apply the dual vector hook
        info = TrainerContext.current()
        if info is not None:
            for hook in info.hooks(DualVectorListener):
                hook(info, queries, documents)
            for hook in info.hooks(DualVectorScorerListener):
                hook(info, queries, documents, scores)
        return scores

    @classmethod
    def from_pretrained_hf(cls, model_id: str, **kwargs):
        """Creates a Dense model from a HuggingFace encoder model

        :param model_id: The HuggingFace model ID
        :param kwargs: Additional keyword arguments passed to the constructor
        :returns: (model, init_tasks) tuple
        """
        from xpmir.text.huggingface.base import (
            HFConfigID,
            HFModel,
            HFModelInitFromID,
        )
        from xpmir.text.huggingface.encoders import HFCLSEncoder

        hf_model = HFModel.C(config=HFConfigID.C(hf_id=model_id))
        init_hf = HFModelInitFromID.C(model=hf_model)
        encoder = HFCLSEncoder.C(model=hf_model, **kwargs)
        return cls.C(encoder=encoder), [init_hf]

    @classmethod
    def from_sentence_transformers(cls, hf_id: str, **kwargs):
        """Creates a dense model from a Sentence transformer

        The list can be found on HuggingFace
        https://huggingface.co/models?library=sentence-transformers

        :param hf_id: The HuggingFace ID
        """
        from xpmir.text.huggingface.encoders import SentenceTransformerTextEncoder

        encoder = SentenceTransformerTextEncoder.C(model_id=hf_id)
        return cls.C(encoder=encoder, **kwargs)


class CosineDense(Dense):
    """Dual model based on cosine similarity."""

    def encode_queries(self, records: List[IDTextRecord]):
        queries = (self.query_encoder or self.encoder)(records)
        return evolve(
            queries, value=queries.value / queries.value.norm(dim=-1, keepdim=True)
        )

    def encode_documents(self, records: List[IDTextRecord]):
        documents = self.encoder(records)
        return evolve(
            documents,
            value=documents.value / documents.value.norm(dim=-1, keepdim=True),
        )


class DotDense(Dense):
    """Dual model based on inner product."""

    def __validate__(self):
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def encode_queries(self, records: List[IDTextRecord]):
        """Encode the different queries"""
        return self._query_encoder(records)

    def encode_documents(self, records: List[IDTextRecord]):
        """Encode the different documents"""
        return self.encoder(records)

    def save_model(self, path: Path):
        """Save sub-encoders independently to subdirectories."""
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_model(path / "encoder")
        if self.query_encoder is not None and self.query_encoder is not self.encoder:
            self._query_encoder.save_model(path / "query_encoder")

    def load_model(self, path: Path):
        """Load sub-encoders from subdirectories, with backward compat."""
        if (path / "encoder").exists():
            self.encoder.load_model(path / "encoder")
            if (path / "query_encoder").exists() and self.query_encoder is not None:
                self._query_encoder.load_model(path / "query_encoder")
        else:
            # Backward compat: flat state_dict
            super().load_model(path)


def dual_representation_metrics(
    info: TrainerContext, queries: torch.Tensor, documents: torch.Tensor
):
    """Compute and report sparsity and QD-FLOPS metrics for dual representations"""
    with torch.no_grad():
        qdflops_count = (
            ((queries > 0).sum(0) * (documents > 0).sum(0)).float().mean()
        ) / queries.shape[1]

        info.metrics.add(ScalarMetric("qdflops_count", qdflops_count.item(), 1))
        info.metrics.add(
            ScalarMetric(
                "sparsity_q",
                torch.count_nonzero(queries).item()
                / (queries.shape[0] * queries.shape[1]),
                len(queries),
            )
        )
        info.metrics.add(
            ScalarMetric(
                "sparsity_d",
                torch.count_nonzero(documents).item()
                / (documents.shape[0] * documents.shape[1]),
                len(documents),
            )
        )

        # Median activation rate of the top-k most frequent terms
        for prefix, x in [("q", queries), ("d", documents)]:
            freq = (x > 0).float().mean(0)
            top_freq = freq.topk(min(20, freq.shape[0])).values
            for k in (1, 5, 10, 20):
                if k <= top_freq.shape[0]:
                    info.metrics.add(
                        ScalarMetric(
                            f"saturation_{prefix}/top{k}",
                            top_freq[:k].median().item(),
                            1,
                        )
                    )


class FlopsRegularizer(DualVectorListener):
    r"""The FLOPS regularizer computes

    .. math::

        FLOPS(q,d) = \lambda_q FLOPS(q) + \lambda_d FLOPS(d)

    where

    .. math::
        FLOPS(x) = \left( \frac{1}{d} \sum_{i=1}^d |x_i| \right)^2
    """

    lambda_q: Param[float]
    """Lambda for queries"""

    lambda_d: Param[float]
    """Lambda for documents"""

    def compute(x: torch.Tensor):
        """
        :param x: term vectors (batch size x vocabulary dimension)
        :returns: A couple (vocabulary dimension / FLOPS regularizations)
        """
        # Computes the mean for each term (weights are positive)
        y = x.mean(0)

        # Returns the sum of squared means
        return y, (y * y).sum()

    def __call__(self, info: TrainerContext, queries, documents):
        # queries and documents are length x dimension
        # Assumes that all weights are positive
        assert info.metrics is not None

        queries = queries.value
        documents = documents.value

        # q of shape (dimension), flops_q of shape (1)
        q, flops_q = FlopsRegularizer.compute(queries)
        d, flops_d = FlopsRegularizer.compute(documents)

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.add_loss(Loss("flops", flops, 1.0))

        info.metrics.add(ScalarMetric("flops", flops.item(), 1))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), 1))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), 1))

        dual_representation_metrics(info, queries, documents)


class ScheduledFlopsRegularizer(FlopsRegularizer):
    """
    The FLOPS regularizer where the lamdba_q and lambda_d varie according to the
    steps. The lambda values goes quadratic before the
    ```lambda_warmup_steps```, and then remains constant
    """

    min_lambda_q: Param[float] = field(default=0, ignore_default=True)
    """Min value for the lambda_q before it increase"""

    min_lambda_d: Param[float] = field(default=0, ignore_default=True)
    """Min value for the lambda_d before it increase"""

    lambda_warmup_steps: Param[int] = field(default=0, ignore_default=True)
    """The warmup steps for the lambda"""

    def quadratic_ratio(self, step):
        if step > self.lambda_warmup_steps:
            return 1
        else:
            return (step / self.lambda_warmup_steps) ** 2

    def __post_init__(self):
        self.initial_lambda_q = self.lambda_q
        self.initial_lambda_d = self.lambda_d

    def __call__(self, info: TrainerContext, queries, documents):
        current_step = info.steps
        self.lambda_q = (
            self.initial_lambda_q - self.min_lambda_q
        ) * self.quadratic_ratio(current_step) + self.min_lambda_q
        self.lambda_d = (
            self.initial_lambda_d - self.min_lambda_d
        ) * self.quadratic_ratio(current_step) + self.min_lambda_d
        FlopsRegularizer.__call__(self, info, queries, documents)
