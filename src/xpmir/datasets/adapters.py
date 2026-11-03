from typing import Iterable, Iterator, List, Optional, Set, Union
from pathlib import Path
from experimaestro import (
    field,
    Param,
    Config,
    Task,
    tqdm,
    pathgenerator,
    Annotated,
    Meta,
    PathGenerator,
)
from functools import cached_property
from datamaestro_ir.data import (
    Adhoc,
    AdhocAssessments,
    IDTextRecord,
    DocumentStore,
    Documents,
    Topics,
)


from datamaestro_ir.data.trec import TrecAdhocAssessments
from datamaestro_ir.data.csv import Topics as CSVTopics
from xpmir.rankers import Retriever
from xpmir.misc import IDList, FileIDList

import logging

logger = logging.getLogger(__name__)


class AbstractTopicFold(Topics):
    """ID-based topic selection"""

    topics: Param[Topics]
    """The collection of the topics"""

    def iter(self):
        ids = set(self.ids)
        for topic in self.topics.iter():
            if topic["id"] in ids:
                yield topic

    def __validate__(self) -> None:
        pass

    @property
    def topic_recordtype(self):
        """The class for topics"""
        return IDTextRecord


class TopicFold(AbstractTopicFold):
    ids: Param[List[str]]
    """A set of the ids for the topics where we select from"""


class IDTopicFold(AbstractTopicFold):
    id_list: Param[IDList]

    @property
    def ids(self):
        return self.id_list.ids


class AbstractAdhocAssessmentFold(AdhocAssessments):
    """Filter assessments by topic ID"""

    qrels: Param[AdhocAssessments]
    """The collection of the assessments"""

    def iter(self):
        ids = set(self.ids)
        for qrels in self.qrels.iter():
            if qrels.topic_id in ids:
                yield qrels


class IDAdhocAssessmentFold(AbstractAdhocAssessmentFold):
    id_list: Param[IDList]

    @property
    def ids(self):
        return self.id_list.ids


class AdhocAssessmentFold(AbstractAdhocAssessmentFold):
    """Filter assessments by topic ID"""

    ids: Param[List[str]]
    """A set of the ids for the assessments where we select from"""


def fold(ids: Iterable[str], dataset: Adhoc):
    """Returns a fold of a dataset, given topic ids"""
    ids = sorted(list(ids))
    topics = TopicFold(topics=dataset.topics, ids=ids)
    qrels = AdhocAssessmentFold(assessments=dataset.assessments, ids=ids)
    return Adhoc(topics=topics, assessments=qrels, documents=dataset.documents)


class ConcatFold(Task):
    """
    Concatenation of several datasets to get a full dataset.
    """

    datasets: Param[List[Adhoc]]
    """The list of Adhoc datasets to concatenate"""

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    """Generated assessments file"""

    topics: Annotated[Path, pathgenerator("topics.tsv")]
    """Generated topics file"""

    def task_outputs(self, dep) -> Adhoc:
        dataset_document_id = set(dataset.document.id for dataset in self.datasets)
        assert len(dataset_document_id) == 1, (
            "At the moment only one set of documents supported."
        )
        return Adhoc(
            id="",  # No need to have a more specific id since it is generated
            topics=dep(CSVTopics(id="", path=self.topics)),
            assessments=dep(TrecAdhocAssessments(id="", path=self.assessments)),
            documents=self.datasets[0].documents,
        )

    def execute(self):
        topics = []
        # concat the topics
        for dataset in self.datasets:
            topics.extend([topic for topic in dataset.topics.iter()])

        # Write topics and assessments
        ids = set()
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic in topics:
                ids.add(topic["id"])
                slash_t = "\t"
                fp.write(
                    f"""{topic["id"]}\t"""
                    f"""{topic["text_item"].text.replace(slash_t, " ")}\n"""
                )

        with self.assessments.open("wt") as fp:
            for dataset in self.datasets:
                for qrels in dataset.assessments.iter():
                    if qrels.topic_id in ids:
                        for qrel in qrels.assessments:
                            fp.write(
                                f"""{qrels.topic_id} 0 {qrel.doc_id} {qrel.rel}\n"""
                            )


class RandomFold(Task):
    """Extracts a random subset of topics from a dataset"""

    seed: Param[int]
    """Random seed used to compute the fold"""

    sizes: Param[List[float]]
    """Number of topics of each fold (or percentage if sums to 1)"""

    dataset: Param[Adhoc]
    """The Adhoc dataset from which a fold is extracted"""

    fold: Param[int]
    """Which fold should be taken"""

    exclude: Param[Optional[Topics]]
    """Exclude some topics from the random fold"""

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    """Generated assessments file"""

    topics: Annotated[Path, pathgenerator("topics.tsv")]
    """Generated topics file"""

    def __validate__(self):
        assert self.fold < len(self.sizes)

    @staticmethod
    def folds(
        seed: int,
        sizes: List[float],
        dataset: Param[Adhoc],
        exclude: Param[Optional[Topics]] = None,
        submit=True,
    ):
        """Creates folds

        Parameters:

        - submit: if true (default), submits the fold tasks to experimaestro
        """

        folds = []
        for ix in range(len(sizes)):
            fold = RandomFold.C(
                seed=seed, sizes=sizes, dataset=dataset, exclude=exclude, fold=ix
            )
            if submit:
                fold = fold.submit()
            folds.append(fold)

        return folds

    def task_outputs(self, dep) -> Adhoc:
        return dep(
            Adhoc.C(
                id="",  # No need to have a more specific id since it is generated
                topics=dep(CSVTopics.C(id="", path=self.topics)),
                assessments=dep(TrecAdhocAssessments.C(id="", path=self.assessments)),
                documents=self.dataset.documents,
            )
        )

    def execute(self):
        import numpy as np

        # Get topics
        badids = (
            set(topic["id"] for topic in self.exclude.iter()) if self.exclude else set()
        )
        topics = [
            topic for topic in self.dataset.topics.iter() if topic["id"] not in badids
        ]
        random = np.random.RandomState(self.seed)
        random.shuffle(topics)

        # Get the fold
        sizes = np.array([0.0] + self.sizes)
        s = sizes.sum()
        if abs(s - 1) < 1e-6:
            sizes = np.round(len(topics) * sizes)
            sizes = np.round(len(topics) * sizes / sizes.sum())

        assert sizes[self.fold + 1] > 0

        indices = sizes.cumsum().astype(int)
        topics = topics[indices[self.fold] : indices[self.fold + 1]]

        # Write topics and assessments
        ids = set()
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic in topics:
                ids.add(topic["id"])
                fp.write(f"""{topic["id"]}\t{topic["text_item"].text}\n""")

        with self.assessments.open("wt") as fp:
            for qrels in self.dataset.assessments.iter():
                if qrels.topic_id in ids:
                    for qrel in qrels.assessments:
                        fp.write(f"""{qrels.topic_id} 0 {qrel.doc_id} {qrel.rel}\n""")


class TopicsFoldGenerator(FileIDList, Task):
    """Extracts a random subset of topics from a dataset

    This task is more generic than the RandomFold one and should work
    whatever the topics/assessments as long as they are serializable (using
    pickle).
    """

    seed: Param[int]
    """Random seed used to compute the fold"""

    sizes: Param[List[float]]
    """Number of topics of each fold (or percentage if sums to 1)"""

    dataset: Param[Adhoc]
    """The Adhoc dataset from which a fold is extracted"""

    fold: Param[int]
    """Which fold should be taken"""

    exclude: Param[Optional[Topics]]
    """Exclude some topics from the random fold"""

    path: Meta[Path] = field(
        default_factory=PathGenerator("topics.lst"), overrides=True
    )
    """Selected topic IDs"""

    def __validate__(self):
        assert self.fold < len(self.sizes)

    @staticmethod
    def folds(
        seed: int,
        sizes: List[float],
        dataset: Param[Adhoc],
        exclude: Param[Optional[Topics]] = None,
        submit=True,
    ):
        """Creates folds

        Parameters:

        - submit: if true (default), submits the fold tasks to experimaestro
        """

        folds = []
        for ix in range(len(sizes)):
            fold = TopicsFoldGenerator.C(
                seed=seed, sizes=sizes, dataset=dataset, exclude=exclude, fold=ix
            )
            if submit:
                fold = fold.submit()
            folds.append(fold)

        return folds

    def task_outputs(self, dep) -> Adhoc:
        return dep(
            Adhoc(
                id="",  # No need to have a more specific id since it is generated
                topics=dep(
                    IDTopicFold.C(id="", topics=self.dataset.topics, id_list=self)
                ),
                assessments=dep(
                    IDAdhocAssessmentFold.C(
                        id="", qrels=self.dataset.assessments, id_list=self
                    )
                ),
                documents=self.dataset.documents,
            )
        )

    def execute(self):
        import numpy as np

        # Get topics
        badids = (
            set(topic["id"] for topic in self.exclude.iter()) if self.exclude else set()
        )
        topics = [
            topic["id"]
            for topic in self.dataset.topics.iter()
            if topic["id"] not in badids
        ]
        random = np.random.RandomState(self.seed)
        random.shuffle(topics)

        # Get the fold
        sizes = np.array([0.0] + self.sizes)
        s = sizes.sum()
        if abs(s - 1) < 1e-6:
            sizes = np.round(len(topics) * sizes)
            sizes = np.round(len(topics) * sizes / sizes.sum())

        assert sizes[self.fold + 1] > 0

        indices = sizes.cumsum().astype(int)
        topics = topics[indices[self.fold] : indices[self.fold + 1]]

        # Write topic IDs
        with self.path.open("wt") as fp:
            for topic in topics:
                fp.write(f"{topic}\n")


class DocumentSubset(Documents):
    """ID-based topic selection"""

    base: Param[DocumentStore]
    """The full document store"""

    docids_path: Meta[Path]
    """Path to the file containing the document IDs"""

    in_memory: Meta[bool] = field(default=False, ignore_default=True)
    """Whether to load the dataset in memory"""

    def __len__(self):
        return len(self.docids)

    def __post_init__(self):
        super().__post_init__()
        self.cache = {}

    @property
    def documentcount(self):
        return len(self.docids)

    def document_ext(self, docid: str):
        if self.in_memory:
            if doc := self.cache.get(docid, None):
                return doc

            doc = self.base.document_ext(docid)
            self.cache[docid] = doc
            return doc

        return self.base.document_ext(docid)

    def __getitem__(self, slice: Union[int, slice]):
        docids = self.docids[slice]
        if isinstance(docids, List):
            return DocumentSubsetSlice(self, self.docids[slice])
        return self.document_ext(docids)

    @cached_property
    def docids(self) -> List[str]:
        # Read document IDs
        with self.docids_path.open("rt") as fp:
            return [line.strip() for line in fp]

    def iter_ids(self):
        yield from self.docids

    def iter(self) -> Iterator[IDTextRecord]:
        for docid in self.iter_ids():
            return self.base.document_ext(docid)


class DocumentSubsetSlice:
    """A slice of a `DocumentSubset`"""

    def __init__(self, subset: DocumentSubset, doc_ids: List[int]):
        self.subset = subset
        self.doc_ids = doc_ids

    def __iter__(self):
        for docid in self.doc_ids:
            yield self.subset.document_ext(docid)

    def __len__(self):
        return len(self.doc_ids)

    def __getitem__(self, ix):
        return self.subset.document_ext(self.doc_ids[ix])


class RetrieverBasedCollection(Task):
    """Buils a subset of documents based on the output of a set of retrievers
    and on relevance assessment.
    First get all the document based on the assessment then add the retrieved ones.
    """

    relevance_threshold: Param[float] = field(default=0, ignore_default=True)
    """Relevance threshold"""

    dataset: Param[Adhoc]
    """A dataset"""

    retrievers: Param[List[Retriever]]
    """Rankers"""

    keepRelevant: Param[bool] = field(default=True, ignore_default=True)
    """Keep documents judged relevant"""

    keepNotRelevant: Param[bool] = field(default=False, ignore_default=True)
    """Keep documents judged not relevant"""

    docids_path: Annotated[Path, pathgenerator("docids.txt")]
    """The file containing the document identifiers of the collection"""

    def __validate__(self):
        assert len(self.retrievers) > 0, "At least one retriever should be given"

    def task_outputs(self, dep) -> Adhoc.C:
        return Adhoc.C(
            id="",  # No need to have a more specific id since it is generated
            topics=self.dataset.topics,
            assessments=self.dataset.assessments,
            documents=dep(
                DocumentSubset.C(
                    id="", base=self.dataset.documents, docids_path=self.docids_path
                )
            ),
        )

    def execute(self):
        for retriever in self.retrievers:
            retriever.initialize()

        # Selected document IDs
        docids: Set[str] = set()

        topics = {t.topic_id: t for t in self.dataset.assessments.iter()}

        # Retrieve all documents
        for topic in tqdm(
            self.dataset.topics.iter(), total=self.dataset.topics.count()
        ):
            qrels = topics.get(topic["id"])
            if qrels is None:
                logger.warning(
                    "Skipping topic %s [%s], (no assessment)",
                    topic["id"],
                    topic["text_item"].text,
                )
                continue

            # Add (not) relevant documents
            if self.keepRelevant:
                docids.update(
                    a.doc_id
                    for a in qrels.assessments
                    if a.rel > self.relevance_threshold
                )

            if self.keepNotRelevant:
                docids.update(
                    a.doc_id
                    for a in qrels.assessments
                    if a.rel <= self.relevance_threshold
                )

            # Retrieve and add
            # already defined the numbers to retrieve inside the retriever, so
            # don't need to worry about the threshold here
            for retriever in self.retrievers:
                docids.update(sd.document["id"] for sd in retriever.retrieve(topic))

        # Write the document IDs
        with self.docids_path.open("wt") as fp:
            fp.writelines(f"{docid}\n" for docid in docids)


class TextStore(Config):
    """Associates an ID with a text"""

    def __getitem__(self, key: str) -> str:
        raise NotImplementedError()


class MemoryTopicStore(TextStore):
    """View a set of topics as a (in memory) text store"""

    topics: Param[Topics]
    """The collection of the topics to build the store"""

    @cached_property
    def store(self):
        return {topic["id"]: topic["text_item"].text for topic in self.topics.iter()}

    def __getitem__(self, key: str) -> str:
        return self.store[key]
