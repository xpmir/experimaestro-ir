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
