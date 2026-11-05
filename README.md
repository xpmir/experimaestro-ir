<div align="center">

# Experimaestro IR (`xpmir`)

**A modular, reproducible Information Retrieval framework for modern neural ranking, dense/sparse retrieval, and distillation.**

[![PyPI version](https://badge.fury.io/py/experimaestro-ir.svg)](https://badge.fury.io/py/experimaestro-ir)
[![Documentation Status](https://readthedocs.org/projects/experimaestro-ir/badge/?version=latest)](https://experimaestro-ir.readthedocs.io/en/latest/?badge=latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

## 📌 Overview

**`experimaestro-ir` (`xpmir`)** is the Information Retrieval extension for [Experimaestro](https://experimaestro-python.readthedocs.io/). It provides a composable, type-safe library of IR primitives for designing, training, and benchmarking neural and classical retrieval pipelines.

Built on top of [xpm-torch](https://github.com/experimaestro/xpm-torch) and [datamaestro](https://datamaestro-ir.readthedocs.io/), `xpmir` bridges research experiments and distributed production pipelines:
- **Reproducible by Design**: Every dataset preparation, index, training run, and evaluation metric is a declared Experimaestro task with content-based caching and dependency tracking.
- **Modern Neural IR Architectures**: Native support for Cross-Encoders, Learned Sparse Retrieval (SPLADE, SEISMIC), Late-Interaction / Multi-Vector (ColBERT, PLAID), and Dense Bi-Encoders, as well as direct [`sentence-transformers`](https://www.sbert.net/index.html) loading, evaluation and export.
- **Advanced LETOR & Distillation**: Pointwise, Pairwise (MarginMSE, RankNet), and Listwise (InfoNCE) training objectives, hard negative sampling, and teacher-student distillation.
- **Scalable Evaluation**: Automated benchmarking against MS MARCO, TREC-DL, BEIR, NanoBEIR, and Robust04 using `ir-measures`, featuring multi-GPU distributed evaluation.

> 💡 **Ecosystem & Projects:** Explore research papers, models, and benchmarks built on top of XPMIR in the [xpmir GitHub organization](https://github.com/xpmir).

---

## 📦 Installation

`xpmir` supports both [`uv`](https://docs.astral.sh/uv/) and `pip`.

### Using `uv` (Recommended)
```bash
# Core package (includes impact-index: BM25 and learned sparse retrieval)
uv add experimaestro-ir

# With Neural IR (PyTorch, Hugging Face Transformers, Sentence Transformers)
uv add "experimaestro-ir[neural]"

# With Classical Search (Anserini / Pyserini)
uv add "experimaestro-ir[anserini]"

# With fast PLAID multi-vector indexing
uv add "experimaestro-ir[fast-plaid]"
```

### Using `pip`
```bash
pip install "experimaestro-ir[neural]"
```

> **Note:** the Anserini backend also needs a Java runtime (11+), and FAISS is
> not installed by any extra: add `faiss-cpu` (or `faiss-gpu`) yourself for
> dense retrieval.

---

## 🧩 Core Architecture & Modules

```
xpmir/
├── neural/         # Neural IR architectures (CrossEncoder, ColBERT, SPLADE, BiEncoders)
├── rankers/        # Scorers, TwoStageRetriever, MonolithicRetriever, Fusion
├── letor/          # Learning-to-Rank: trainers, batchers, samplers (MarginMSE, InfoNCE)
├── index/          # Index interfaces: dense (FAISS), sparse inverted index, SEISMIC
├── evaluation.py   # Benchmark evaluation pipelines powered by ir-measures
├── interfaces/     # External integrations: Anserini, Hugging Face, SentenceTransformers
└── datasets/       # Dataset adapters for datamaestro-ir and ir-datasets
```

| Component | Capabilities |
|---|---|
| **[Scorers & Encoders](https://experimaestro-ir.readthedocs.io/en/latest/neural.html)** | Cross-Encoders, Dual-Encoders, ColBERT late interaction, SPLADE sparse activations, DRMM. |
| **[Samplers & Training](https://experimaestro-ir.readthedocs.io/en/latest/letor/index.html)** | Dynamic batching, hard negative mining (BM25, SPLADE, ColBERT negatives), pairwise/listwise distillation. |
| **[Retrievers & Indices](https://experimaestro-ir.readthedocs.io/en/latest/retrieval.html)** | `TwoStageRetriever`, BM25 (Anserini), FAISS dense index, Rust-accelerated sparse inverted index, PLAID. |
| **[Evaluation & Benchmarks](https://experimaestro-ir.readthedocs.io/en/latest/evaluation.html)** | Turnkey evaluation on standard collections: MS MARCO, BEIR, NanoBEIR, LoTTE, Robust04, TREC-DL via `ir-measures`. |
| **[Datasets & Collections](https://experimaestro-ir.readthedocs.io/en/latest/data/index.html)** | Datamaestro and `ir-datasets` adapters, document tokenization, collection partitioning. |
| **[Experiments & Distributed](https://experimaestro-ir.readthedocs.io/en/latest/experiments.html)** | Declarative pipelines, multi-GPU Lightning Fabric training, SLURM cluster scheduling. |


---

## ⚡ Quickstart: Evaluating a Two-Stage Pipeline

The following minimal example sets up a complete two-stage re-ranking experiment (Lightweigh NanoBEIR FEVER dataset from [Datamaestro](https://github.com/experimaestro/datamaestro), BM25 first-stage candidate retrieval, and cross-encoder re-ranking using [MiniLM](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) reranker loaded trough sentence-transformers):

(Its light enough so you can run it locally with `uv run examples/bm25_miniLM.py`)
```python
import logging
from pathlib import Path
from datamaestro import prepare_dataset
from experimaestro import experiment

from xpmir.rankers.standard import BM25
from xpmir.index.bow import BOWSparseRetrieverIndexBuilder, BOWRetriever
from xpmir.neural.sentence_transformers import STCrossEncoder
from xpmir.rankers import TwoStageRetriever
from xpmir.evaluation import Evaluate

logging.basicConfig(level=logging.INFO)

# Define experiment plan
with experiment(Path("./experiments/demo"), "nano_fever_bm25_minilm") as xp:
    # 1. Load benchmark dataset from Datamaestro
    dataset = prepare_dataset("co.huggingface.nano-beir.fever")

    # 2. Index documents (This experimaestro Task writes de BoW index)
    index = BOWSparseRetrieverIndexBuilder.C(documents=dataset.documents).submit()

    # 3. Define First stage BM25 retriever
    first_stage = BOWRetriever.C(
        index=index,
        model=BM25.C(),
        topk=100,
    ).tag("first_stage", "bm25")

    # 4. Define neural cross-encoder re-ranker
    scorer = STCrossEncoder.C(
        model_id="microsoft/MiniLM-L12-H384-uncased"
    )

    # 4. Compose two-stage retriever: rerank top 50 with Cross-Encoder
    retriever = TwoStageRetriever.C(
        retriever=first_stage,
        scorer=scorer,
        top_k=50,
        batchsize=32,
    ).tag("model", "bm25_minilm_reranker")

    # 5. Submit evaluation task (computes nDCG@20, RR, AP via ir-measures)
    eval_task = Evaluate.C(dataset=dataset, retriever=retriever).submit()

print(f"Results written to: {eval_task.results}")
with open(eval_task.results) as f:
    metrics = f.read()
    print(f"Evaluation Metrics for \"microsoft/MiniLM-L12-H384-uncased\":\n{metrics}")
```

When executed, Experimaestro prepares the dataset, indexes the documents with BM25, re-ranks candidate passages with the Cross-Encoder, and computes standard IR evaluation metrics:

```text
Evaluation Metrics for "microsoft/MiniLM-L12-H384-uncased":
P@20                      all        0.0070
nDCG@20                   all        0.0386
RR                        all        0.0239
AP                        all        0.0218
nDCG                      all        0.0869
```

> [!TIP]
> **Hint: Try re-running it!** The results appear immediately! Because [`experimaestro`](https://github.com/experimaestro/experimaestro-python) cached all `Tasks` execution for you. Each task gets a unique hash ID and is stored in its own folder, so when you try to run the same task a second time, it is not relaunched. (you can even draw the DAG of tasks with `uv run experimaestro dag -w ./experiments/demo nano_fever_bm25_minilm`)


> [!NOTE]
> **More Complete example:** You can find a more complex pipeline for training Cross-encoders (featuring Grid searches, SoTa training recipes, Evaluations across full BEIR/Lotte/TREC collections with cached SPLADE first stage pre-computation ...) in the [xpmir Cross-Encoders repo](https://github.com/xpmir/cross-encoders).


---

## 🛠️ CLI & Experiment Management

`xpmir` rely on Experimaestro's command line to run declarative YAML or Python experiments:

```bash
# Validate task dependencies and launchers before running (Dry Run)
uv run experimaestro run configs/experiment.yaml --run-mode DRY_RUN

# Launch locally or submit to a SLURM cluster
uv run experimaestro run configs/experiment.yaml
```

To inspect running tasks, monitors, and outputs in real time:
```bash
experimaestro monitor
```

---

## 🚀 Release (maintainers)

During development, dependencies that evolve together (e.g. `datamaestro-ir`)
can be pinned to git commits in `[tool.uv.sources]`. These sources are not
part of the published package, which only has the lower bounds of
`[project.dependencies]`: releases are made with
[uv-release-manager](https://github.com/experimaestro/uv-release-manager),
which checks that every git source points to a release published on PyPI and
that the lower bounds match it.

```bash
# Can we release? (git sources, lower bounds, clean and pushed tree, uv.lock)
uv run uv-release-manager check
# Move the git sources to the next (or latest) release containing the pinned
# commit -- the dependency must be released first
uv run uv-release-manager update
# Set the lower bounds to the released versions of the git sources
uv run uv-release-manager bump
# Check, then create the GitHub release (version and notes from the
# conventional commits, with git-cliff)
uv run uv-release-manager gh-release --dry-run
uv run uv-release-manager gh-release
```

`uv-release-manager` is in the `release` dependency group (included in `dev`;
the publish workflow installs it alone, without the project).

The GitHub release triggers the publish workflow
(`.github/workflows/python-publish.yml`), which runs `check` again before
building and uploading the package to PyPI.

---

## 📖 Documentation & Resources

- **Full Documentation**: [https://experimaestro-ir.readthedocs.io/](https://experimaestro-ir.readthedocs.io/)
- **Projects & Research**: [xpmir GitHub Organization](https://github.com/xpmir) (featured research repositories and paper reproductions built with `xpmir`)
- **Core Orchestrator**: [Experimaestro](https://github.com/experimaestro/experimaestro-python)
- **PyTorch Foundations**: [xpm-torch](https://github.com/experimaestro/xpm-torch)
- **Dataset Ecosystem**: [datamaestro](https://github.com/experimaestro/datamaestro) & [datamaestro-ir](https://github.com/experimaestro/datamaestro-ir)
- **Reference Implementations**: [Reproducing Cross-Encoders](https://github.com/xpmir/cross-encoders)

---

## 📑 Citation

If you use `xpmir` in your research, please cite:

```bibtex
@software{piwowarski2023xpmir,
  author = {Benjamin Piwowarski and contributors},
  title = {Experimaestro IR: A Modular Framework for Information Retrieval Research},
  url = {https://github.com/xpmir/experimaestro-ir},
  year = {2023}
}
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 ([GPL-3.0](LICENSE)).
