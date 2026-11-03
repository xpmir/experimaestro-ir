"""Paired significance tests between evaluation results

Works on the per-topic results (``detailed.dat``) written by
:class:`xpmir.evaluation.Evaluate`, i.e. on its
:class:`~datamaestro_ir.data.trec.TrecAdhocResults` output.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
from datamaestro_ir.data.trec import TrecAdhocResults


class PairedTest(NamedTuple):
    """Result of a paired test of a run against a baseline"""

    mean: float
    baseline_mean: float
    pvalue: float
    topics: int

    def better(self, alpha: float = 0.05) -> bool:
        """True if the run is significantly better than the baseline"""
        return self.pvalue < alpha and self.mean > self.baseline_mean


def per_topic(results: TrecAdhocResults | Path, measure: str) -> dict[str, float]:
    """Per-topic values of a measure (e.g. ``"nDCG@10"``)"""
    path = Path(results.detailed if isinstance(results, TrecAdhocResults) else results)
    values = {}
    with path.open() as fp:
        for line in fp:
            name, qid, value = line.split()
            if name == measure and qid != "all":
                values[qid] = float(value)
    return values


def paired_ttest(
    run: TrecAdhocResults | Path,
    baseline: TrecAdhocResults | Path,
    measure: str,
) -> PairedTest | None:
    """Two-sided paired t-test of ``run`` against ``baseline``

    Topics missing from one of the results count as 0.

    :returns: the test, or None if there are less than 2 topics
    """
    from scipy import stats

    a, b = per_topic(run, measure), per_topic(baseline, measure)
    qids = sorted(set(a) | set(b))
    if len(qids) < 2:
        return None
    x = np.array([a.get(q, 0.0) for q in qids])
    y = np.array([b.get(q, 0.0) for q in qids])
    pvalue = 1.0 if np.allclose(x, y) else float(stats.ttest_rel(x, y).pvalue)
    return PairedTest(float(x.mean()), float(y.mean()), pvalue, len(qids))
