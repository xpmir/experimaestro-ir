from experimaestro import field, Param, Config


class Model(Config):
    """Base class for standard IR models"""

    pass


class BM25(Model):
    """BM-25 model definition"""

    k1: Param[float] = field(default=0.9, ignore_default=True)
    b: Param[float] = field(default=0.4, ignore_default=True)

    variant: Param[str] = field(default="bm25", ignore_default=True)
    """IDF formula used by impact-index: ``"bm25"`` (Robertson/Sparck-Jones)
    or ``"lucene"`` (Lucene's ``BM25Similarity``, as in Pyserini/Anserini)"""


class QLDirichlet(Model):
    """Query likelihood (Dirichlet smoothing) model definition"""

    mu: Param[float] = field(default=1000, ignore_default=True)
