"""Interface to the fast-plaid library.

`fast-plaid <https://github.com/lightonai/fast-plaid>`_ is a Rust-based
implementation of PLAID / ColBERT late-interaction retrieval. This module
wraps it to build and query an index from a
:class:`~xpmir.neural.colbert.ColBERTEncoder`.

Three classes are exposed:

- :class:`PlaidIndex` — the index configuration (paths, metadata). Supports
  retrieving per-document token vectors via :meth:`PlaidIndex.get_document_tokens`
  using fast-plaid's compressed centroid+residual storage.
- :class:`PlaidIndexBuilder` — a :class:`~experimaestro.Task` that encodes a
  :class:`~datamaestro_ir.data.DocumentStore` and builds the fast-plaid index.
- :class:`PlaidRetriever` — a :class:`~xpmir.rankers.Retriever` that searches
  the index given a query.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Union

import torch
from experimaestro import (
    Config,
    Meta,
    Param,
    PathGenerator,
    Task,
    field,
    tqdm,
)
from datamaestro_ir.data import DocumentStore, IDTextRecord

from xpm_torch.configuration import FabricConfiguration
from xpm_torch.utils.fabric import fallback_fa2_if_incompatible_precision
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.rankers.scorer import AbstractModuleScorer
from xpmir.text.encoders import TextEncoderBase
from xpmir.utils.utils import batchiter

logger = logging.getLogger(__name__)


def _import_fast_plaid():
    try:
        from fast_plaid import search as fp_search  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover - guard clause
        raise ModuleNotFoundError(
            "fast-plaid is not installed. Install it with `pip install fast-plaid`"
            " (see https://github.com/lightonai/fast-plaid)."
        ) from exc
    return fp_search


# File layout (under ``index_path``):
#
#   plaid/           — fast-plaid index directory
#   metadata.json    — dim, num_docs, n_bits
#

_PLAID_SUBDIR = "plaid"
_METADATA_FILE = "metadata.json"
_EXT2INT_FILE = "docid_ext2int.json"


class PlaidIndex(Config):
    """A ColBERT / PLAID index backed by `fast-plaid`_.

    The index stores per-token document embeddings in fast-plaid's compressed
    centroid + residual format. Per-document token vectors can be
    reconstructed (approximately) via :meth:`get_document_tokens`, which
    delegates to fast-plaid's ``get_embeddings`` method. The reconstruction
    quality is controlled by :attr:`n_bits`.

    When :attr:`compress_only` is ``True`` the index only contains the
    compressed vectors (centroids + quantised residuals) without the IVF
    search structure. This is cheaper to build and sufficient when only
    :meth:`get_document_tokens` is needed. Attempting to search a
    compress-only index via :class:`PlaidRetriever` will raise an error.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    documents: Param[DocumentStore]
    """Set of documents to index."""

    compress_only: Param[bool] = False

    index_path: Meta[Path]
    """Directory containing the fast-plaid index and side-car files."""

    device: Meta[str] = field(default="", ignore_default=True)
    """Device used to load the index for :meth:`get_document_tokens`
    (``""`` = auto). Fixed at first use because the underlying
    ``FastPlaid`` instance is cached."""

    in_memory: Meta[bool] = field(default=False, ignore_default=True)
    """If ``True``, load the index fully into device memory (passes
    ``low_memory=False`` to fast-plaid). Use when the index fits in
    VRAM/RAM and you want faster decompression/search; otherwise the
    document codes and residuals stay memory-mapped from disk."""

    def _plaid_dir(self) -> Path:
        return self.index_path / _PLAID_SUBDIR

    def _ext2int_file(self) -> Path:
        return self.index_path / _EXT2INT_FILE

    def _load_ext2int(self) -> dict[str, int]:
        cached = getattr(self, "_ext2int", None)
        if cached is not None:
            return cached
        path = self._ext2int_file()
        if path.exists():
            with path.open("r") as fh:
                ext2int = json.load(fh)
        else:
            logger.info(
                "Building external-to-internal docid mapping for %s...", self.index_path
            )
            ext2int = {}
            for doc in self.documents.iter_documents():
                doc_id = (
                    doc.get("id") if isinstance(doc, dict) else getattr(doc, "id", None)
                )
                if doc_id is not None:
                    ext2int[str(doc_id)] = len(ext2int)
            with path.open("w") as fh:
                json.dump(ext2int, fh)
        self._ext2int = ext2int
        return ext2int

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_fast_plaid", None)
        state.pop("_fast_plaid_device", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fast_plaid = None
        self._fast_plaid_device = None

    def _get_fast_plaid(self, device: str = ""):
        """Return a cached ``FastPlaid`` instance for this index.

        The instance is constructed lazily on first access and reused
        afterwards so that subsequent calls (e.g. repeated
        :meth:`get_document_tokens`) avoid reloading the index.
        """
        cached = getattr(self, "_fast_plaid", None)
        cached_device = getattr(self, "_fast_plaid_device", None)
        target_device = device or self.device or None
        if cached is not None and (
            target_device is None or target_device == cached_device
        ):
            return cached

        fp_search = _import_fast_plaid()
        fp = fp_search.FastPlaid(
            index=str(self._plaid_dir()),
            device=target_device,
            low_memory=not self.in_memory,
        )
        self._fast_plaid = fp
        self._fast_plaid_device = target_device
        return fp

    def get_document_tokens(
        self,
        docids: list[int | str] | int | str,
        device: str = "",
    ) -> list[torch.Tensor] | torch.Tensor:
        """Return the (approximate) per-token embeddings for a document or batch of documents.

        The vectors are reconstructed from fast-plaid's compressed
        centroid + residual storage using ``FastPlaid.get_embeddings``.
        The reconstruction quality depends on :attr:`n_bits`.

        :param docids: The document identifier(s). Integers are interpreted as
            internal positions in the index (``0..num_docs-1``); strings are
            looked up in the external-to-internal map written at indexing
            time.
        :param device: Device for the fast-plaid instance used to decompress
            (``""`` = auto).
        :returns: A single ``(num_tokens, dim)`` tensor if docids is a single ID,
            or a list of float tensors if docids is a list of IDs.
        """
        is_single = isinstance(docids, (str, int))
        docids_list = [docids] if is_single else list(docids)

        if not docids_list:
            return torch.empty(0) if is_single else []

        if isinstance(docids_list[0], str):
            ext2int = self._load_ext2int()
            internal_docids: list[int] = []
            for docid in docids_list:
                if docid not in ext2int:
                    raise KeyError(
                        f"External document id {docid!r} is unknown to this index"
                    )
                internal_docids.append(int(ext2int[docid]))
        else:
            internal_docids = [int(docid) for docid in docids_list]

        fp = self._get_fast_plaid(device=device)
        results = fp.get_embeddings(subset=internal_docids)
        if is_single:
            return results[0]
        return results


class PlaidIndexBuilder(Task):
    """Builds a fast-plaid index from a document collection.

    The builder encodes every document using the given
    :class:`~xpmir.neural.colbert.ColBERTEncoder`, collects the valid (i.e.
    non-padding) token vectors, and feeds them to ``fast-plaid``.

    The fast-plaid index stores the embeddings in a compressed
    centroid + residual format, so no separate raw-token file is needed.
    Per-document token vectors can be reconstructed later via
    :meth:`PlaidIndex.get_document_tokens`.
    """

    documents: Param[DocumentStore]
    """Set of documents to index."""

    encoder: Param[Union[TextEncoderBase, AbstractModuleScorer]]
    """The ColBERT-style encoder used to produce per-token embeddings."""

    batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Encoder batch size. Warning, different from the batch size used internally by fast-plaid ('fast_plaid_batch_size')"""

    buffer_size: Param[int] = field(default=1000, ignore_default=True)
    """Number of documents to encode and accumulate in RAM before creating/updating the fast-plaid index and fitting the centroids.
    The token embeddings used to initialize the centroids will be sampled randomly from those documents by plaid
    (or they will all be used if n_samples_kmeans is 0)."""

    fast_plaid_batch_size: Meta[int] = field(default=32, ignore_default=True)
    """Fast plaid internal batch size. If the value is 0 means using the auto"""

    n_bits: Param[int] = field(default=2, ignore_default=True)
    """Number of bits used by fast-plaid for residual quantisation."""

    kmeans_niters: Param[int] = field(default=4, ignore_default=True)
    """Number of K-means iterations performed by fast-plaid when clustering
    the centroids."""

    n_samples_kmeans: Param[int] = field(default=0, ignore_default=True)
    """Number of token samples used to train the centroids (0 = fast-plaid
    default)."""

    max_points_per_centroid: Param[int] = field(default=256, ignore_default=True)
    """Maximum number of points (documents) per centroid. Controls the creation of new centroids."""

    seed: Param[int] = field(default=42, ignore_default=True)
    """Random seed for reproducibility (passed to fast-plaid's index creation)."""

    compress_only: Param[bool] = field(default=False, ignore_default=True)
    """When ``True``, skip IVF construction. The resulting index supports
    :meth:`PlaidIndex.get_document_tokens` but not search via
    :class:`PlaidRetriever`.

    Requires fast-plaid support for ``compress_only``
    (see `lightonai/fast-plaid#41 <https://github.com/lightonai/fast-plaid/pull/41>`_).
    Falls back to building the full index with a warning if unsupported."""

    low_memory: Param[bool] = field(default=True)
    """https://github.com/lightonai/fast-plaid#-search-speed-tip-low_memoryfalse
    If index fits on VRAM, set to False for faster search. Otherwise, keep True to avoid OOM errors."""

    force_cpu_indexing: Param[bool] = field(default=False)
    """When True, forces the use of CPU for indexing even if a GPU is available.
    This can be useful to avoid GPU OOM errors during indexing, especially for large corpora."""

    fabric_config: Meta[FabricConfiguration] = field(
        default_factory=FabricConfiguration.C
    )
    """Control the device for the model encoding and fast-plaid index."""

    index_path: Meta[Path] = field(default_factory=PathGenerator("plaid-index"))
    """Output directory for the index and its side-car files."""

    def task_outputs(self, dep) -> PlaidIndex:
        """Expose a :class:`PlaidIndex` for downstream tasks."""
        return dep(
            PlaidIndex.C(
                documents=self.documents,
                index_path=self.index_path,
                compress_only=self.compress_only,
            )
        )

    def execute(self):
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Fabric first
        fabric = self.fabric_config.get_fabric()
        fabric.launch()

        with fabric.init_module():
            self.encoder.initialize()
            # PlaidIndexBuilder calls fabric.setup() directly rather than a
            # setup_with_fabric() hook, so the FA2->SDPA downgrade normally
            # applied during training never runs here; do it explicitly.
            fallback_fa2_if_incompatible_precision(self.encoder.model, fabric)
            self.encoder = fabric.setup(self.encoder)
            self.encoder.eval()

        fp_search = _import_fast_plaid()

        plaid_dir = self.index_path / _PLAID_SUBDIR
        plaid_dir.mkdir(parents=True, exist_ok=True)

        device = fabric.device if not self.force_cpu_indexing else "cpu"
        fast_plaid = fp_search.FastPlaid(
            index=str(plaid_dir), device=device, low_memory=self.low_memory
        )

        total_docs = self.documents.documentcount or 0
        num_docs_seen = 0
        index_created = False
        doc_buffer: list = []

        ext2int: dict = {}
        with torch.no_grad():
            pbar = tqdm(
                total=total_docs or None,
                desc="Encoding documents for fast-plaid",
                unit="doc",
            )
            for batch in batchiter(self.batch_size, self.documents.iter_documents()):
                for doc in batch:
                    doc_id = (
                        doc.get("id")
                        if isinstance(doc, dict)
                        else getattr(doc, "id", None)
                    )
                    if doc_id is not None:
                        ext2int[str(doc_id)] = len(ext2int)
                per_doc = self.encoder.document_token_embeddings(batch)
                per_doc_cpu = [
                    t.detach().to("cpu", dtype=torch.float32) for t in per_doc
                ]
                if (
                    not num_docs_seen
                    and len(per_doc_cpu) > 0
                    and per_doc_cpu[0].shape[0] > 0
                ):
                    sample_norms = per_doc_cpu[0].norm(dim=-1)
                    mean_token_norm = sample_norms.mean().item()
                    if abs(mean_token_norm - 1.0) > 0.15:
                        logger.warning(
                            "⚠️ [FastPLAID Warning] Document token embeddings have mean norm %.3f "
                            "(expected ~1.0). FastPLAID centroids are unit-norm; unnormalized embeddings "
                            "cause residual quantization distortion at low bit-widths (e.g. 2-bit).",
                            mean_token_norm,
                        )
                doc_buffer.extend(per_doc_cpu)

                if len(doc_buffer) >= self.buffer_size:
                    logging.debug(
                        "Warmup buffer filled (%d documents, %d tokens). "
                        "Creating the fast-plaid index and fitting centroids...",
                        len(doc_buffer),
                        sum(t.shape[0] for t in doc_buffer),
                    )

                    if not index_created:
                        # Enough docs accumulated — fit centroids and create index
                        create_kwargs = {
                            "documents_embeddings": doc_buffer,
                            "nbits": self.n_bits,
                            "kmeans_niters": self.kmeans_niters,
                            "seed": self.seed,
                            "max_points_per_centroid": self.max_points_per_centroid,
                        }
                        if self.fast_plaid_batch_size:
                            create_kwargs["batch_size"] = self.fast_plaid_batch_size
                        if self.n_samples_kmeans:
                            create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
                        if self.compress_only:
                            create_kwargs["compress_only"] = True
                        try:
                            fast_plaid.create(**create_kwargs)
                        except TypeError:
                            if self.compress_only:
                                logger.warning(
                                    "compress_only is not supported by this "
                                    "version of fast-plaid; building the full "
                                    "index instead. See "
                                    "https://github.com/lightonai/fast-plaid/pull/41"
                                )
                                del create_kwargs["compress_only"]
                                fast_plaid.create(**create_kwargs)
                            else:
                                raise

                        doc_buffer.clear()  # free RAM immediately
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        index_created = True
                    else:
                        create_kwargs = {
                            "kmeans_niters": self.kmeans_niters,
                            "seed": self.seed,
                            "max_points_per_centroid": self.max_points_per_centroid,
                        }
                        if self.fast_plaid_batch_size:
                            create_kwargs["batch_size"] = self.fast_plaid_batch_size
                        if self.n_samples_kmeans:
                            create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        fast_plaid.update(
                            documents_embeddings=doc_buffer, **create_kwargs
                        )
                        doc_buffer.clear()  # free RAM immediately
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # ID mapping — same logic regardless of create vs update path
                num_docs_seen += len(per_doc_cpu)
                pbar.update(len(per_doc_cpu))
            pbar.close()

        # In case the whole corpus was smaller than the warmup buffer, we still want to create the index
        if not index_created and doc_buffer:
            create_kwargs = {
                "documents_embeddings": doc_buffer,
                "nbits": self.n_bits,
                "kmeans_niters": self.kmeans_niters,
                "seed": self.seed,
                "max_points_per_centroid": self.max_points_per_centroid,
            }
            if self.fast_plaid_batch_size:
                create_kwargs["batch_size"] = self.fast_plaid_batch_size
            if self.n_samples_kmeans:
                create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
            if self.compress_only:
                create_kwargs["compress_only"] = True
            try:
                fast_plaid.create(**create_kwargs)
            except TypeError:
                if self.compress_only:
                    logger.warning(
                        "compress_only is not supported by this "
                        "version of fast-plaid; building the full "
                        "index instead. See "
                        "https://github.com/lightonai/fast-plaid/pull/41"
                    )
                    del create_kwargs["compress_only"]
                    fast_plaid.create(**create_kwargs)
                else:
                    raise
            doc_buffer.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif index_created and doc_buffer:
            create_kwargs = {
                "kmeans_niters": self.kmeans_niters,
                "seed": self.seed,
                "max_points_per_centroid": self.max_points_per_centroid,
            }
            if self.fast_plaid_batch_size:
                create_kwargs["batch_size"] = self.fast_plaid_batch_size
            if self.n_samples_kmeans:
                create_kwargs["n_samples_kmeans"] = self.n_samples_kmeans
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fast_plaid.update(documents_embeddings=doc_buffer, **create_kwargs)
            doc_buffer.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("No documents left to encode.")

        with (self.index_path / _EXT2INT_FILE).open("w") as fh:
            json.dump(ext2int, fh)

        with (self.index_path / _METADATA_FILE).open("w") as fh:
            json.dump(
                {
                    "num_docs": num_docs_seen,
                    "dim": self.encoder.dimension,
                    "n_bits": self.n_bits,
                },
                fh,
            )

        logger.info(
            "fast-plaid index built: %d documents, dim=%d, n_bits=%d",
            num_docs_seen,
            self.encoder.dimension,
            self.n_bits,
        )


class PlaidRetriever(Retriever):
    """Retriever using a `fast-plaid`_ PLAID index.

    .. _fast-plaid: https://github.com/lightonai/fast-plaid
    """

    encoder: Param[AbstractModuleScorer]
    """The query encoder. Typically the same encoder that was used to build
    :attr:`index`."""

    index: Param[PlaidIndex]
    """The fast-plaid index to search."""

    topk: Param[int]
    """Number of documents to return per query."""

    n_ivf_probe: Meta[int] = field(default=8, ignore_default=True)
    """Number of inverted-list clusters explored by fast-plaid at search
    time."""

    n_full_scores: Meta[int] = field(default=0, ignore_default=True)
    """Number of candidates for which fast-plaid computes full scores
    (0 = fast-plaid default)."""

    batch_size: Meta[int] = field(default=64, ignore_default=True)
    """Query encoding and search batch size for retrieve_all."""

    fabric_config: Meta[FabricConfiguration] = field(
        default_factory=FabricConfiguration.C
    )
    """Control the device for the model encoding and fast-plaid index."""

    def initialize(self):
        super().initialize()
        if self.index.compress_only:
            raise RuntimeError(
                "Cannot search a compress-only PLAID index. "
                "Rebuild with compress_only=False to enable retrieval."
            )

        logger.info("PLAID retriever (1/2): initializing the encoder")
        # 1. Initialize Fabric first
        fabric = self.fabric_config.get_fabric()
        fabric.launch()
        self.fabric = fabric

        with fabric.init_module():
            self.encoder.initialize()
            self.encoder = fabric.setup(self.encoder)
            self.encoder.eval()
            # query_token_embeddings (called directly from retrieve() below,
            # not via forward()) needs to be explicitly whitelisted -- under
            # strategies that wrap the module (e.g. DDP), Fabric only
            # auto-instruments forward() by default.
            self.encoder.mark_forward_method("query_token_embeddings")

        logger.info("PLAID retriever (2/2): opening the fast-plaid index")
        fp_search = _import_fast_plaid()
        if fabric.device and fabric.device.type == "cuda":
            device = str(fabric.device)
        else:
            device = "cpu"
        self._fast_plaid = fp_search.FastPlaid(
            index=str(self.index._plaid_dir()), device=device
        )

    def _store(self):
        return self.index.documents

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        with torch.no_grad():
            queries_embeddings = self.encoder.query_token_embeddings([record])
            queries_embeddings = queries_embeddings.detach().to(
                "cpu", dtype=torch.float32
            )

            search_kwargs = {"top_k": self.topk, "n_ivf_probe": self.n_ivf_probe}
            if self.n_full_scores:
                search_kwargs["n_full_scores"] = self.n_full_scores

            results = self._fast_plaid.search(
                queries_embeddings=queries_embeddings,
                **search_kwargs,
            )

        single = results[0] if results else []
        documents = self.index.documents
        out: List[ScoredDocument] = []
        for doc_index, score in single:
            out.append(
                ScoredDocument(documents.document_int(int(doc_index)), float(score))
            )
        return out

    def retrieve_all(
        self, queries: Dict[str, IDTextRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        """Retrieves documents for a set of queries in batches.

        Processes queries in batches of size `self.batch_size` to optimize
        GPU utilization and query encoding speed. Supports multi-GPU distributed
        evaluation via Lightning Fabric / torch.distributed.
        """
        documents = self.index.documents
        results: Dict[str, List[ScoredDocument]] = {}
        items = list(queries.items())

        fabric = getattr(self, "fabric", None)
        is_distributed = fabric is not None and fabric.world_size > 1

        if is_distributed:
            world_size = fabric.world_size
            rank = fabric.global_rank
            local_items = items[rank::world_size]
            desc = f"(Rank {rank}): PLAID retrieval (batch_size={self.batch_size})"
            logger.info(
                f"{desc} ({len(local_items)}/{len(items)} queries on this rank)"
            )
            disable_tqdm = not fabric.is_global_zero
        else:
            local_items = items
            desc = f"PLAID retrieval (batch_size={self.batch_size})"
            disable_tqdm = False

        search_kwargs = {"top_k": self.topk, "n_ivf_probe": self.n_ivf_probe}
        if self.n_full_scores:
            search_kwargs["n_full_scores"] = self.n_full_scores

        pbar = (
            tqdm(
                total=len(local_items),
                desc=desc,
                unit="query",
            )
            if not disable_tqdm
            else None
        )

        for i in range(0, len(local_items), self.batch_size):
            batch_items = local_items[i : i + self.batch_size]
            batch_keys = [k for k, _ in batch_items]
            batch_records = [r for _, r in batch_items]

            with torch.no_grad():
                batch_embeddings = self.encoder.query_token_embeddings(batch_records)
                batch_embeddings = batch_embeddings.detach().to(
                    "cpu", dtype=torch.float32
                )
                batch_results = self._fast_plaid.search(
                    queries_embeddings=batch_embeddings,
                    show_progress=False,
                    **search_kwargs,
                )

            for key, query_results in zip(batch_keys, batch_results):
                results[key] = [
                    ScoredDocument(documents.document_int(int(doc_index)), float(score))
                    for doc_index, score in query_results
                ]

            if pbar is not None:
                pbar.update(len(batch_items))

        if pbar is not None:
            pbar.close()

        if is_distributed:
            import torch.distributed as dist

            # Gather results from all GPUs
            local_data = list(results.items())
            gathered_data = [None] * fabric.world_size
            dist.all_gather_object(gathered_data, local_data)

            if fabric.is_global_zero:
                # Merge into master dictionary
                final_results = {}
                for rank_data in gathered_data:
                    for key, scored_docs in rank_data:
                        final_results[key] = scored_docs
                # Preserve original query ordering
                results = {k: final_results.get(k, []) for k in queries}
            else:
                return {}

        return results
