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


class BOWSparseRetrieverIndexBuilder(Task):
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

    language: Param[str] = field(default="english", ignore_default=True)
    """Language for stemming and stop words"""

    stop_words: Param[bool] = field(default=True, ignore_default=True)
    """Whether to filter stop words (uses Lucene defaults for the language)"""

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

    def execute(self):
        if self.index_path.is_dir():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Setup builder options
        options = impact_index.BuilderOptions()
        options.in_memory_threshold = self.in_memory_threshold

        builder = impact_index.BOWIndexBuilder(
            str(self.index_path),
            options=options,
            dtype="int32",
            stemmer=self.stemmer,
            language=self.language,
            stop_words=self.stop_words,
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
            raw_index = impact_index.Index.load(str(self.index_path), False)
            compressed_path = self.index_path.with_name(
                self.index_path.name + "_compressed"
            )
            raw_index.compress(str(compressed_path))

            # Drop the reference so mmap'd files are closed before rmtree
            # (otherwise NFS leaves .nfs* files and rmdir fails with ENOTEMPTY)
            del raw_index
            gc.collect()

            # Replace uncompressed index with compressed one. Move the old
            # directory aside first so the compressed index is in place even
            # if removal fails (e.g. NFS .nfs* leftovers).
            old_path = self.index_path.with_name(self.index_path.name + "_old")
            self.index_path.rename(old_path)
            compressed_path.rename(self.index_path)
            try:
                shutil.rmtree(old_path)
            except OSError as e:
                logger.warning(
                    "Could not fully remove old index at %s: %s", old_path, e
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
