Retrieval
=========

This page describes the components used for retrieval, i.e. finding the top
:math:`k` documents from a collection given a query. XPMIR supports classical
models (BM25, query-likelihood), dense retrieval (FAISS), sparse learned
retrieval, and late-interaction models (ColBERT/PLAID), as well as multi-stage
pipelines that combine them.

.. contents:: On this page
   :local:
   :depth: 2


Base classes
------------

Core data structures and the abstract retriever interface that all
implementations extend.

.. autoclass:: xpmir.rankers.ScoredDocument

.. autoxpmconfig:: xpmir.rankers.retriever.Retriever
    :members: initialize, collection, retrieve_all, retrieve

Standard IR models
------------------

Definitions for classical probabilistic retrieval models. These are
backend-agnostic specifications that can be instantiated with a concrete
engine: either :class:`~xpmir.index.bow.BOWRetriever` (impact-index, installed
by default, BM25 only) or :class:`~xpmir.interfaces.anserini.AnseriniRetriever`
(Lucene, requires the ``anserini`` extra and Java). See
:ref:`index-backends` below.

.. autoxpmconfig:: xpmir.rankers.standard.Model
.. autoxpmconfig:: xpmir.rankers.standard.BM25
.. autoxpmconfig:: xpmir.rankers.standard.QLDirichlet

Multi-stage retrievers
----------------------

In a re-ranking setting, a two-stage retriever first retrieves candidates with
a fast first-stage model, then re-scores them with a more expensive scorer.

The re-ranking process is memory-efficient: it uses lazy evaluation of
first-stage results and maximises GPU throughput by batching query-document
pairs across multiple queries.

.. autoxpmconfig:: xpmir.rankers.scorer.AbstractTwoStageRetriever
.. autoxpmconfig:: xpmir.rankers.scorer.TwoStageRetriever

Duo-retrievers
--------------

Duo-retrievers predict which of two candidate documents is more relevant to the
query (pairwise preference), rather than assigning an absolute score.

.. autoxpmconfig:: xpmir.rankers.scorer.DuoTwoStageRetriever
.. autoxpmconfig:: xpmir.rankers.scorer.DuoLearnableScorer

Miscellaneous retrievers
------------------------

Utility retrievers for loading pre-computed runs, hydrating results with
document text, or exhaustive scoring.

.. autoxpmconfig:: xpmir.rankers.full.FullRetriever
.. autoxpmconfig:: xpmir.rankers.full.FullRetrieverRescorer
.. autoxpmconfig:: xpmir.rankers.retriever.RetrieverHydrator
.. autoxpmconfig:: xpmir.rankers.retriever.RunRetriever


Distributed Retrieval
---------------------

XPMIR supports distributed retrieval and re-ranking across multiple GPUs using
`Lightning Fabric <https://lightning.ai/docs/fabric/>`_. **Currently, this
optimized distributed logic is implemented for :class:`~xpmir.index.sparse.SparseRetriever`
and :class:`~xpmir.rankers.scorer.TwoStageRetriever`.**

This is particularly useful for large-scale evaluation on thousands of queries.

How it works
^^^^^^^^^^^^

When a retriever is configured with a Fabric instance, the :meth:`~xpmir.rankers.retriever.Retriever.retrieve_all`
method leverages all available devices:

1. **Query Sharding**: The set of queries is automatically partitioned across
   the available GPUs.
2. **Parallel Processing**: Each device processes its assigned shard.

   - For **Sparse Retrieval** (:class:`~xpmir.index.sparse.SparseRetriever`),
     queries are encoded in batches and searched via asynchronous workers.
   - For **Two-Stage Retrieval** (:class:`~xpmir.rankers.scorer.TwoStageRetriever`),
     document re-ranking is batched across queries for maximum throughput.
3. **Result Gathering**: Once processing is complete, results are collected
   from all ranks and merged on the global zero rank.

Usage
^^^^^

Distributed retrieval is automatically enabled when using the :class:`~xpmir.evaluation.Evaluate`
task with a multi-GPU :class:`~xpm_torch.configuration.FabricConfiguration`.

To use it manually in a script:

.. code-block:: python

    from lightning import Fabric
    fabric = Fabric(devices=2, strategy="ddp")
    fabric.launch()

    retriever.initialize()
    retriever.setup_with_fabric(fabric)

    # Distributed retrieval
    results = retriever.retrieve_all(queries)

    # Results are gathered on rank 0
    if fabric.is_global_zero:
        print(f"Total queries retrieved: {len(results)}")


.. _index-backends:

Index backends
--------------

The sections below describe the available index backends and their associated
retrievers. The table summarises the options, and what needs to be installed
on top of the base ``experimaestro-ir`` package.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Backend
     - Type
     - Requires
   * - `Bag-of-words <Bag-of-words (impact-index)_>`_
     - BM25
     - (included)
   * - `Anserini`_
     - BM25, query-likelihood
     - ``[anserini]`` extra, Java 11+
   * - `Sparse <Sparse retrieval_>`_
     - Learned sparse (SPLADE…)
     - (included)
   * - `BMP <Sparse retrieval_>`_
     - Learned sparse
     - (included)
   * - `Seismic <Sparse retrieval_>`_
     - Learned sparse (approx.)
     - (included)
   * - `FAISS`_
     - Dense
     - ``faiss-cpu`` / ``faiss-gpu``
   * - `fast-plaid <fast-plaid (ColBERT / PLAID)_>`_
     - Late interaction
     - ``[fast-plaid]`` extra

"(included)" means the backend relies on ``impact-index``, a core dependency;
extras are installed as ``experimaestro-ir[<extra>]``.

Neural backends (learned sparse, FAISS, fast-plaid) also need an encoder,
which usually requires the ``experimaestro-ir[neural]`` extra.

Bag-of-words (impact-index)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classical BM25 retrieval backed by the
`impact-index <https://github.com/experimaestro/impact-index>`_ Rust library.
Since ``impact-index`` is a core dependency, this backend works out of the box,
without Java or Pyserini, and is the recommended way to get a BM25 baseline.

**Classes:** :class:`~xpmir.index.bow.BOWSparseRetrieverIndexBuilder` (task) →
:class:`~xpmir.index.bow.BOWSparseRetrieverIndex` →
:class:`~xpmir.index.bow.BOWRetriever`.

Documents are analysed at indexing time; the defaults of
:class:`~xpmir.index.bow.BOWSparseRetrieverIndexBuilder` mimic Lucene's
``EnglishAnalyzer`` (Porter stemmer, Lucene English stop words, UAX#29
tokenization), so results are close to Anserini's BM25. The analyzer is stored
in the index, and queries are processed with the same pipeline. Retrieval uses
MaxScore dynamic pruning. Only the :class:`~xpmir.rankers.standard.BM25` model
is currently supported (``k1``, ``b`` and the IDF ``variant`` are set at
retrieval time, so one index can serve several BM25 parameterizations). Use
``BM25.C(variant="lucene")`` for Lucene's IDF, to match Pyserini/Anserini
scores. ``BOWRetriever.retrieve_all`` searches all the topics at once on
impact-index's thread pool.

.. code-block:: python

    from xpmir.evaluation import Evaluate
    from xpmir.index.bow import BOWRetriever, BOWSparseRetrieverIndexBuilder
    from xpmir.rankers.standard import BM25

    index = BOWSparseRetrieverIndexBuilder.C(documents=ds.documents).submit()
    retriever = BOWRetriever.C(index=index, model=BM25.C(), topk=1000)
    Evaluate.C(dataset=ds, retriever=retriever).submit()

.. autoxpmconfig:: xpmir.index.bow.BOWSparseRetrieverIndexBuilder
.. autoxpmconfig:: xpmir.index.bow.BOWSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.bow.BOWRetriever

Anserini
^^^^^^^^

`Anserini <https://github.com/castorini/anserini>`_ provides classical
inverted-index retrieval (BM25, query-likelihood, etc.) via Lucene.
It requires the ``experimaestro-ir[anserini]`` extra (which installs
Pyserini) and a Java runtime (version 11 or later, found via ``JAVA_HOME``).

**Classes:** :class:`~xpmir.interfaces.anserini.IndexCollection` (task, an
:class:`~xpmir.index.anserini.Index`) →
:class:`~xpmir.interfaces.anserini.AnseriniRetriever`.

.. autoxpmconfig:: xpmir.index.anserini.Index
.. autoxpmconfig:: xpmir.interfaces.anserini.AnseriniRetriever
.. autoxpmconfig:: xpmir.interfaces.anserini.IndexCollection
.. autoxpmconfig:: xpmir.interfaces.anserini.SearchCollection

FAISS
^^^^^

`FAISS <https://github.com/facebookresearch/faiss>`_ provides approximate
nearest-neighbour search for dense vector retrieval.
FAISS is **not** installed automatically (there is no extra, since the right
package depends on your hardware): add ``faiss-cpu`` or ``faiss-gpu``
yourself.

**Classes:** :class:`~xpmir.index.faiss.IndexBackedFaiss` (task, a
:class:`~xpmir.index.faiss.FaissIndex`) →
:class:`~xpmir.index.faiss.FaissRetriever`.

.. autoxpmconfig:: xpmir.index.faiss.FaissIndex
.. autoxpmconfig:: xpmir.index.faiss.IndexBackedFaiss
.. autoxpmconfig:: xpmir.index.faiss.FaissRetriever


fast-plaid (ColBERT / PLAID)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interface to `fast-plaid <https://github.com/lightonai/fast-plaid>`_, a
Rust-based implementation of PLAID / ColBERT late-interaction retrieval.
It requires the ``experimaestro-ir[fast-plaid]`` extra.

**Classes:** :class:`~xpmir.index.plaid.PlaidIndexBuilder` (task) →
:class:`~xpmir.index.plaid.PlaidIndex` →
:class:`~xpmir.index.plaid.PlaidRetriever`.

Per-document token vectors can be reconstructed from the compressed index
via :meth:`~xpmir.index.plaid.PlaidIndex.get_document_tokens`.

.. autoxpmconfig:: xpmir.index.plaid.PlaidIndex
    :members: get_document_tokens
.. autoxpmconfig:: xpmir.index.plaid.PlaidIndexBuilder
.. autoxpmconfig:: xpmir.index.plaid.PlaidRetriever


Sparse retrieval
^^^^^^^^^^^^^^^^

Learned sparse retrieval indexes (e.g. for SPLADE), backed by the
`impact-index <https://github.com/experimaestro/impact-index>`_ Rust library
(a core dependency: nothing extra to install, including for BMP and Seismic).

.. autoxpmconfig:: xpmir.index.sparse.AbstractSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.AbstractSparseRetrieverIndexBuilder
.. autoxpmconfig:: xpmir.index.sparse.SparseRetriever

**Impact library (Rust)**

**Classes:** :class:`~xpmir.index.sparse.SparseRetrieverIndexBuilder` (task) →
:class:`~xpmir.index.sparse.SparseRetrieverIndex` →
:class:`~xpmir.index.sparse.SparseRetriever`.

.. autoxpmconfig:: xpmir.index.sparse.SparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.SparseRetrieverIndexBuilder

**Block-Max Pruning**

Adapters for `Faster Learned Sparse Retrieval with Block-Max Pruning
<https://arxiv.org/abs/2405.01117>`_.

**Classes:** either :class:`~xpmir.index.sparse.BMPSparseRetrieverIndexBuilder`
(task, indexes directly) or :class:`~xpmir.index.sparse.Sparse2BMPConverter`
(task, converts a :class:`~xpmir.index.sparse.SparseRetrieverIndex`) →
:class:`~xpmir.index.sparse.BMPSparseRetrieverIndex` →
:class:`~xpmir.index.sparse.BMPSparseRetriever`.

.. autoxpmconfig:: xpmir.index.sparse.BMPSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.BMPSparseRetrieverIndexBuilder
.. autoxpmconfig:: xpmir.index.sparse.Sparse2BMPConverter
.. autoxpmconfig:: xpmir.index.sparse.BMPSparseRetriever

**Seismic Index**

**Classes:** :class:`~xpmir.index.sparse.Sparse2SeismicConverter` (task,
converts a :class:`~xpmir.index.sparse.SparseRetrieverIndex`) →
:class:`~xpmir.index.sparse.SeismicSparseRetrieverIndex` →
:class:`~xpmir.index.sparse.SeismicSparseRetriever`.

.. autoxpmconfig:: xpmir.index.sparse.SeismicSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.SeismicSparseRetriever
.. autoxpmconfig:: xpmir.index.sparse.Sparse2SeismicConverter
