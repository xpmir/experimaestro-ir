"""Bag-of-Words index with BM25 scoring using impact_index"""

import gc
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from experimaestro import (
    Config,
    Task,
    Param,
    Meta,
    Prepare,
    field,
    PathGenerator,
    tqdm,
    Constant,
)
from datamaestro_ir.data import DocumentStore, IDTextRecord

from xpmir.rankers import Retriever, ScoredDocument
from xpmir.rankers.standard import Model, BM25
import impact_index

logger = logging.getLogger(__name__)

# Default batch size for parallel text analysis
DEFAULT_BATCH_SIZE = 10000


class BOWSparseRetrieverIndex(Config):
    """A bag-of-words index with BM25 scoring

    Uses impact_index.BOWIndexBuilder for text-based tokenization
    and BM25 scoring at retrieval time.
    """

    documents: Param[DocumentStore]
    """The indexed document collection"""

    index_path: Meta[Path]
    """Path to the index directory"""

    def initialize(self, in_memory: bool, model: Model):
        """Initialize the index with scoring model

        :param in_memory: Whether to load the index fully in memory
        :param model: The scoring model (BM25)
        """
        index = impact_index.Index.load(str(self.index_path.absolute()), in_memory)

        if isinstance(model, BM25):
            scoring = impact_index.BM25Scoring(
                k1=model.k1, b=model.b, variant=model.variant
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        self.scored_index = index.with_scoring(scoring)

        # Auto-load analyzer from the index
        self._analyzer = index.analyzer()

    def analyze_query(self, text: str) -> Dict[int, float]:
        """Tokenize and stem a query string into term IDs"""
        return self._analyzer.analyze_query(text)

    def retrieve(self, query: Dict[int, float], top_k: int) -> List[ScoredDocument]:
        return self.retrieve_many([query], top_k)[0]

    def retrieve_many(
        self, queries: List[Dict[int, float]], top_k: int
    ) -> List[List[ScoredDocument]]:
        """Searches a batch of queries (in parallel, on impact-index's thread
        pool)"""
        return [
            [
                ScoredDocument(self.documents.document_int(sd.docid), sd.score)
                for sd in hits
            ]
            for hits in self.scored_index.search_maxscore_many(queries, top_k)
        ]

    def retrieve_structured_many(
        self, queries: List[str], top_k: int
    ) -> List[List[ScoredDocument]]:
        """Searches a batch of structured (matchop) queries, e.g.
        ``#combine:0=2:1=1(lazy dog)``, in parallel"""
        return [
            [
                ScoredDocument(self.documents.document_int(sd.docid), sd.score)
                for sd in hits
            ]
            for hits in self.scored_index.search_maxscore_query_many(queries, top_k)
        ]


class BOWRetriever(Retriever):
    """BM25 retriever using the impact_index BOW index

    This mirrors the AnseriniRetriever but uses the impact_index library
    for BM25 scoring instead of Lucene/pyserini.
    """

    index: Param[BOWSparseRetrieverIndex]
    """The BOW index"""

    model: Param[Model]
    """The scoring model (e.g. BM25)"""

    topk: Param[int]
    """Number of documents to return"""

    in_memory: Meta[bool] = field(default=False, ignore_default=True)
    """Whether the index should be fully loaded in memory"""

    structured: Param[bool] = field(default=False, ignore_default=True)
    """Queries are structured (matchop) queries, e.g. ``#combine:0=2(a b)``
    or ``#combine(a #syn(b c))`` (see impact-index), instead of text"""

    def initialize(self):
        super().initialize()
        self.index.initialize(self.in_memory, self.model)

    @staticmethod
    def _text(record: IDTextRecord | str) -> str:
        return record["text_item"].text if isinstance(record, dict) else record

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        if self.structured:
            return self.index.retrieve_structured_many([self._text(record)], self.topk)[
                0
            ]
        query = self.index.analyze_query(self._text(record))
        return self.index.retrieve(query, self.topk)

    def retrieve_all(
        self, queries: Dict[str, IDTextRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        keys = list(queries)
        if self.structured:
            texts = [self._text(queries[k]) for k in keys]
            return dict(
                zip(keys, self.index.retrieve_structured_many(texts, self.topk))
            )
        analyzed = [self.index.analyze_query(self._text(queries[k])) for k in keys]
        return dict(zip(keys, self.index.retrieve_many(analyzed, self.topk)))


class BOWSparseRetrieverIndexBuilder(Task, Prepare):
    """Builds a bag-of-words index from document text

    Uses impact_index.BOWIndexBuilder to tokenize documents and store
    term frequencies + document lengths for BM25 scoring.

    Defaults match Lucene/Pyserini's EnglishAnalyzer pipeline:
    - Porter stemmer (original, not Snowball/Porter2)
    - English stop words (33-word Lucene default)
    - UAX#29 tokenization with English possessive filter
    - Block size 128 for effective block-max pruning
    """

    documents: Param[DocumentStore]
    """Set of documents to index"""

    stemmer: Param[str] = field(default="porter", ignore_default=True)
    """Stemmer: 'porter' (Lucene-compatible), 'snowball' (Porter2), or 'none'"""

    language: Param[str | None] = field(default=None, ignore_default=True)
    """Language for stemming and stop words (name or ISO 639-1 code); by
    default, the single language of ``documents.languages`` if set (it is
    metadata, but the documents' identifier differs anyway), or English"""

    stop_words: Param[bool] = field(default=True, ignore_default=True)
    """Whether to filter stop words (uses Lucene defaults for the language)"""

    pipeline: Param[str | None] = field(default=None, ignore_default=True)
    """impact-index text analysis pipeline. With ``"pyserini"`` and another
    language than English, the Lucene analyzer Anserini/Pyserini uses for the
    language (``stemmer`` is then ignored unless it is not ``"porter"``). By
    default, ``"pyserini"`` for other languages than English, and the
    analysis described above for English"""

    batch_size: Param[int] = field(default=DEFAULT_BATCH_SIZE, ignore_default=True)
    """Batch size for parallel text analysis"""

    max_docs: Param[int] = field(default=0, ignore_default=True)
    """Maximum number of indexed documents (0 = all)"""

    in_memory_threshold: Param[int] = field(default=128, ignore_default=True)
    """Block size for posting lists (128 = optimal for block-max pruning)"""

    index_path: Meta[Path] = field(default_factory=PathGenerator("index"))
    """Path to store the index"""

    compress: Param[bool] = field(default=True, ignore_default=True)
    """Whether to compress the index after building (default: True)"""

    version: Constant[int] = 3
    """Version 3: Porter stemmer, stop words, batch indexing, compression by default"""

    def analysis(self) -> tuple[str, str | None]:
        """The language and pipeline used to analyze the documents"""
        language = self.language
        if language is None:
            languages = getattr(self.documents, "languages", None) or []
            if len(languages) > 1:
                raise ValueError(
                    f"Documents have several languages ({languages}): "
                    "the builder's language must be given"
                )
            language = languages[0] if languages else "english"
        pipeline = self.pipeline
        if pipeline is None and language not in ("en", "english"):
            pipeline = "pyserini"
        return language, pipeline

    def prepare(self):
        """Downloads what the text analysis needs (e.g. the Japanese and
        Korean dictionaries of the Pyserini pipeline)

        Run by experimaestro on the machine running the experimental plan,
        before launching the task -- so that it can run on nodes without
        network access, provided they share impact-index's cache directory
        (``$IMPACT_INDEX_CACHE``, or the OS cache directory)
        """
        language, pipeline = self.analysis()
        for path in impact_index.prepare_analyzer(language, pipeline=pipeline):
            logger.info("Analyzer resource for %s: %s", language, path)

    def execute(self):
        # With compression, the raw index is built aside and compressed into
        # the index path (written last); without, it is built in place
        raw_path = self.index_path
        if self.compress:
            raw_path = self.index_path.with_name(self.index_path.name + "_raw")

        # Remove the leftovers of an interrupted run (including the
        # directories of older versions): building or compressing into them
        # could keep e.g. their vocabulary, whose term IDs differ
        for suffix in ("", "_raw", "_compressed", "_old"):
            path = self.index_path.with_name(self.index_path.name + suffix)
            if path.is_dir():
                if suffix:
                    logger.warning("Removing the leftover directory %s", path)
                shutil.rmtree(path)
        raw_path.mkdir(parents=True, exist_ok=True)

        # Setup builder options
        options = impact_index.BuilderOptions()
        options.in_memory_threshold = self.in_memory_threshold

        language, pipeline = self.analysis()
        kwargs = {}
        stemmer = self.stemmer
        if pipeline is not None:
            kwargs["pipeline"] = pipeline
            # The pipeline's own stemmer unless another one is asked for
            if stemmer == "porter":
                stemmer = None
        builder = impact_index.BOWIndexBuilder(
            str(raw_path),
            options=options,
            dtype="int32",
            stemmer=stemmer,
            language=language,
            stop_words=self.stop_words,
            **kwargs,
        )

        max_docs = self.documents.documentcount
        if self.max_docs:
            max_docs = min(self.max_docs, max_docs or sys.maxsize)
            logger.warning("Limited indexing to %d documents", max_docs)

        # Batch indexing with parallel text analysis
        batch = []
        with tqdm(
            total=max_docs,
            unit="documents",
            desc="Building BOW index",
        ) as pb:
            for docid, doc in enumerate(self.documents.iter_documents()):
                if self.max_docs and docid >= max_docs:
                    break
                text = doc["text_item"].text
                batch.append((docid, text))
                if len(batch) >= self.batch_size:
                    builder.add_texts(batch)
                    pb.update(len(batch))
                    batch = []
            if batch:
                builder.add_texts(batch)
                pb.update(len(batch))

        logger.info("Building the index")
        builder.build(False)
        logger.info("BOW index built")

        if self.compress:
            logger.info("Compressing the index")
            raw_index = impact_index.Index.load(str(raw_path), False)
            raw_index.compress(str(self.index_path))

            # Drop the reference so mmap'd files are closed before rmtree
            # (otherwise NFS leaves .nfs* files and rmdir fails with ENOTEMPTY)
            del raw_index
            gc.collect()
            try:
                shutil.rmtree(raw_path)
            except OSError as e:
                logger.warning(
                    "Could not fully remove the raw index at %s: %s", raw_path, e
                )
            logger.info("Index compressed")

    def task_outputs(self, dep):
        """Returns a BOW index that can be used by a BOWRetriever"""
        return dep(
            BOWSparseRetrieverIndex.C(
                index_path=self.index_path,
                documents=self.documents,
            )
        )
