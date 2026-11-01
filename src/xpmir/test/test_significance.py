import pytest

from xpmir.significance import paired_ttest, per_topic


def write(path, values: dict[str, float], measure="nDCG@10"):
    with path.open("w") as fp:
        for qid, value in values.items():
            fp.write(f"{measure:25s} {qid:10s} {value:.4f}\n")
        fp.write(f"{'RR@10':25s} {'q1':10s} 1.0000\n")
    return path


def test_per_topic(tmp_path):
    path = write(tmp_path / "a.dat", {"q1": 0.5, "q2": 0.25})
    assert per_topic(path, "nDCG@10") == {"q1": 0.5, "q2": 0.25}


def test_paired_ttest(tmp_path):
    run = write(tmp_path / "run.dat", {f"q{i}": 0.5 + 0.01 * i for i in range(20)})
    base = write(tmp_path / "base.dat", {f"q{i}": 0.3 for i in range(19)})

    test = paired_ttest(run, base, "nDCG@10")
    assert test.topics == 20  # q19 is missing from the baseline (counts as 0)
    assert test.baseline_mean == pytest.approx(0.3 * 19 / 20)
    assert test.better()
    assert not paired_ttest(base, run, "nDCG@10").better()

    same = paired_ttest(run, run, "nDCG@10")
    assert same.pvalue == 1.0 and not same.better()
