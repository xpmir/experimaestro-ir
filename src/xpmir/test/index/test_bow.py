import pytest
from pathlib import Path
from experimaestro import ObjectStore
from experimaestro.xpmutils import DirectoryContext
from xpmir.index.bow import (
    BOWRetriever,
    BOWSparseRetrieverIndex,
    BOWSparseRetrieverIndexBuilder,
)
from xpmir.rankers.standard import BM25
from xpmir.test.utils.utils import SampleDocumentStore


@pytest.fixture
def context(tmp_path: Path):
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    return DirectoryContext(tmp_path)


@pytest.fixture
def bow_index(context):
    """Build a BOW index from sample documents"""
    objects = ObjectStore()
    documents = SampleDocumentStore.C(num_docs=50)
    builder = BOWSparseRetrieverIndexBuilder.C(
        documents=documents,
        stemmer="snowball",
        language="english",
    )

    builder_instance: BOWSparseRetrieverIndexBuilder = builder.instance(
        context=context, objects=objects
    )
    builder_instance.execute()

    index_config = builder.task_outputs(lambda x: x)
    return index_config, objects, builder_instance.documents


@pytest.fixture
def bow_retriever(context, bow_index):
    """Create a BOWRetriever from the built index"""
    index_config, objects, _ = bow_index
    retriever = BOWRetriever.C(
        index=index_config,
        model=BM25.C(k1=1.2, b=0.75),
        topk=10,
    ).instance(context=context, objects=objects)
    retriever.initialize()
    return retriever


def test_bow_index_build(bow_index):
    """Test that the BOW index builds without errors"""
    index_config, objects, documents = bow_index
    index_instance: BOWSparseRetrieverIndex = index_config.instance(objects=objects)

    # Verify the index path and doc_meta files exist
    assert index_instance.index_path.is_dir()
    assert (index_instance.index_path / "docmeta.cbor").exists()
    assert (index_instance.index_path / "docmeta.dat").exists()


def test_bow_retrieve(bow_retriever):
    """Test that the BOW retriever returns results for a query"""
    results = bow_retriever.retrieve("document")

    assert len(results) > 0
    assert len(results) <= 10

    # Scores should be in decreasing order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score

    # Each result should have an id
    for sd in results:
        assert "id" in sd.document


def test_bow_retrieve_all(bow_retriever):
    """Test that retrieve_all is consistent with individual retrieve calls"""
    queries = {
        "q1": "first query",
        "q2": "second document",
        "q3": "information retrieval",
    }
    all_results = bow_retriever.retrieve_all(queries)

    assert set(all_results.keys()) == set(queries.keys())

    for key, query in queries.items():
        single_results = bow_retriever.retrieve(query)

        observed = [d.document["id"] for d in all_results[key]]
        expected = [d.document["id"] for d in single_results]
        assert observed == expected


def test_bow_retrieve_empty_query(bow_retriever):
    """Test retrieval with a query that has no matching terms"""
    results = bow_retriever.retrieve("xyzzyplugh")
    assert isinstance(results, list)


def test_bow_bm25_variant(context, bow_index):
    """The BM25 IDF variant is passed to impact-index"""
    index_config, objects, _ = bow_index

    def scores(variant):
        retriever = BOWRetriever.C(
            index=index_config, model=BM25.C(variant=variant), topk=10
        ).instance(context=context, objects=objects)
        retriever.initialize()
        return [sd.score for sd in retriever.retrieve("document")]

    bm25, lucene = scores("bm25"), scores("lucene")
    assert len(bm25) == len(lucene) > 0
    assert bm25 != lucene


def test_bow_structured_queries(context, bow_index):
    """Structured (matchop) queries: a weighted #combine matches the
    equivalent text query with repeated terms"""
    index_config, objects, _ = bow_index

    def retriever(structured: bool):
        r = BOWRetriever.C(
            index=index_config, model=BM25.C(), topk=10, structured=structured
        ).instance(context=context, objects=objects)
        r.initialize()
        return r

    text, structured = retriever(False), retriever(True)
    expected = text.retrieve("document document query")
    observed = structured.retrieve("#combine:0=2:1=1(document query)")
    assert [d.document["id"] for d in observed] == [d.document["id"] for d in expected]
    assert [d.score for d in observed] == pytest.approx([d.score for d in expected])

    queries = {"q1": "#combine(first query)", "q2": "#combine:0=3(second document)"}
    all_results = structured.retrieve_all(queries)
    for key, query in queries.items():
        assert [d.document["id"] for d in all_results[key]] == [
            d.document["id"] for d in structured.retrieve(query)
        ]


def test_bow_index_ignores_leftovers(context):
    """Leftovers of an interrupted run (a compressed index of other documents,
    whose vocabulary assigns other term IDs) do not end up in the index"""
    import impact_index

    objects = ObjectStore()
    builder = BOWSparseRetrieverIndexBuilder.C(
        documents=SampleDocumentStore.C(num_docs=50), stemmer="none"
    )
    instance: BOWSparseRetrieverIndexBuilder = builder.instance(
        context=context, objects=objects
    )

    # Stale compressed index: same terms, in another order
    stale_raw = instance.index_path.with_name("stale")
    stale_raw.mkdir(parents=True)
    stale = impact_index.BOWIndexBuilder(
        str(stale_raw), dtype="int32", stemmer="none", stop_words=False
    )
    stale.add_texts([(i, f"Document {49 - i}") for i in range(50)])
    stale.build(False)
    leftover = instance.index_path.with_name(instance.index_path.name + "_compressed")
    impact_index.Index.load(str(stale_raw), False).compress(str(leftover))

    instance.execute()
    assert not leftover.exists()

    retriever = BOWRetriever.C(
        index=builder.task_outputs(lambda x: x), model=BM25.C(), topk=10
    ).instance(context=context, objects=objects)
    retriever.initialize()
    for ix in (3, 7, 42):
        assert [d.document["id"] for d in retriever.retrieve(str(ix))] == [str(ix)]


def _supports(feature: str) -> bool:
    import impact_index

    from datamaestro_ir.data import Documents

    if feature == "languages":
        return "languages" in Documents.__getxpmtype__().arguments
    return hasattr(impact_index, "prepare_analyzer")


def _analysis(context, **kwargs):
    builder = BOWSparseRetrieverIndexBuilder.C(**kwargs)
    return builder.instance(context=context, objects=ObjectStore()).analysis()


def test_bow_index_builder_analysis(context):
    """Language and pipeline, given or inferred"""
    documents = SampleDocumentStore.C(num_docs=5)
    assert _analysis(context, documents=documents) == ("english", None)
    assert _analysis(context, documents=documents, language="fr") == (
        "fr",
        "pyserini",
    )
    assert _analysis(
        context, documents=documents, language="en", pipeline="pyserini"
    ) == ("en", "pyserini")


def test_bow_index_builder_prepare():
    """The builder is its own Prepare (run before launching it)"""
    from experimaestro.core.objects.config import _collect_prepares

    builder = BOWSparseRetrieverIndexBuilder.C(
        documents=SampleDocumentStore.C(num_docs=5), language="ja"
    )
    assert any(p is builder for p in _collect_prepares(builder))


@pytest.mark.skipif(not _supports("languages"), reason="needs Documents.languages")
def test_bow_index_builder_documents_language(context):
    documents = SampleDocumentStore.C(num_docs=5, languages=["fr"])
    assert _analysis(context, documents=documents) == ("fr", "pyserini")
    # The language is metadata: the identifier is the one of a default builder
    assert (
        BOWSparseRetrieverIndexBuilder.C(documents=documents).__identifier__().all
        == BOWSparseRetrieverIndexBuilder.C(
            documents=SampleDocumentStore.C(num_docs=5, languages=["fr"])
        )
        .__identifier__()
        .all
    )
    with pytest.raises(ValueError):
        _analysis(
            context,
            documents=SampleDocumentStore.C(num_docs=5, languages=["fr", "de"]),
        )


@pytest.mark.skipif(not _supports("pipeline"), reason="needs impact-index >= 1.9")
def test_bow_pyserini_french(context):
    """Index built with the French Lucene analyzer of Pyserini"""
    objects = ObjectStore()
    builder = BOWSparseRetrieverIndexBuilder.C(
        documents=SampleDocumentStore.C(num_docs=20), language="fr"
    )
    task = builder.instance(context=context, objects=objects)
    task.prepare()
    task.execute()
    index = builder.task_outputs(lambda x: x).instance(objects=objects)
    import impact_index

    analyzer = impact_index.Index.load(str(index.index_path), True).analyzer()
    # French light stemming: "documents" -> "document"
    assert analyzer.analyze_query("documents") == analyzer.analyze_query("document")
