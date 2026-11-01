Evaluation
==========

The evaluation module provides tasks for measuring retrieval quality. Given a
retriever and a test collection (topics + relevance assessments), it produces
per-query and aggregate metric scores using the
`ir-measures <https://ir-measur.es/>`_ library.

.. contents:: On this page
   :local:
   :depth: 2


Evaluation tasks
----------------

These configurations define how a retrieval run is evaluated.
:class:`~xpmir.evaluation.BaseEvaluation` is the abstract base;
:class:`~xpmir.evaluation.Evaluate` is the standard concrete implementation.

The :class:`~xpmir.evaluation.Evaluate` task automatically handles multi-GPU
acceleration if the provided ``fabric_config`` specifies multiple devices.
It shards the retrieval/re-ranking task across GPUs and merges the final run
before computing metrics.

.. autoxpmconfig:: xpmir.evaluation.BaseEvaluation
.. autoxpmconfig:: xpmir.evaluation.RunEvaluation

.. autoxpmconfig:: xpmir.evaluation.Evaluate

.. autoclass:: xpmir.evaluation.Evaluations
    :members: evaluate_retriever, add, to_dataframe

.. autoclass:: xpmir.evaluation.EvaluationsCollection
    :members: evaluate_retriever, to_dataframe

Significance tests
------------------

:mod:`xpmir.significance` compares the per-topic results of two
:class:`~xpmir.evaluation.Evaluate` tasks (their
:class:`~datamaestro_ir.data.trec.TrecAdhocResults` output) with a paired
t-test.

.. code-block:: python

    from xpmir.significance import paired_ttest

    test = paired_ttest(run_results, bm25_results, "nDCG@10")
    if test.better(alpha=0.05):
        ...

.. autofunction:: xpmir.significance.paired_ttest
.. autofunction:: xpmir.significance.per_topic
.. autoclass:: xpmir.significance.PairedTest
    :members: better

Metrics
-------

Metrics are backed by the `ir-measures <https://ir-measur.es/>`_ library.
Cut-off values can be specified with the ``@`` operator.

.. autoxpmconfig:: xpmir.measures.Measure

List of built-in measures:

.. automodule:: xpmir.measures
    :members: AP, P, RR, nDCG, R, Success

Example:

.. code-block:: python

    from xpmir.measures import AP, P, nDCG, RR
    from xpmir.evaluation import Evaluate

    measures = [AP, P@20, nDCG, nDCG@10, nDCG@20, RR, RR@10]
    Evaluate.C(dataset=dataset, retriever=retriever, measures=measures)
