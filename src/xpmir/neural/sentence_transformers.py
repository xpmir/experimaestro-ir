"""Mixin for ModuleLoader subclasses that produce sentence-transformers
compatible checkpoints on HuggingFace Hub export.

Loaders that inherit from :class:`SpladeLoaderMixin` will
automatically write ST config files via
:meth:`~xpm_torch.module.ModuleLoader.write_hub_extras` and append ST
loading instructions via
:meth:`~xpm_torch.module.ModuleLoader.hub_readme_sections`.
"""

import json
import string
import torch
from pathlib import Path
from typing import Optional, List, Union, Tuple, Any, Dict

from transformers import AutoConfig

from sentence_transformers import CrossEncoder, MultiVectorEncoder

from experimaestro import Config, Param, field
from datamaestro_ir.data import IDTextRecord
from xpmir.text.huggingface.tokenizers import get_default_max_len, HFTokenizer
from xpmir.text import TokenizedTexts
from xpmir.letor.records import BaseItems, ProductItems
from xpmir.rankers import AbstractModuleScorer
from xpm_torch.module import SimpleModuleLoader
from xpm_torch.utils.fabric import fallback_fa2_if_incompatible_precision
from xpm_torch.utils import to_device
from xpmir.text.tokenizers import TokenizerOptions


import logging

logger = logging.getLogger(__name__)


MLM_TYPE = "sentence_transformers.sparse_encoder.models.MLMTransformer.MLMTransformer"
POOLING_TYPE = "sentence_transformers.sparse_encoder.models.SpladePooling.SpladePooling"
ROUTER_TYPE = "sentence_transformers.models.Router"


class SpladeLoaderMixin:
    """Mixin for ModuleLoader subclasses that adds sentence-transformers
    compatibility on HF Hub export.

    Expects the concrete loader to define ``encoder_path`` and optionally
    ``query_encoder_path`` DataPath fields. After experimaestro serialization,
    encoder files are at ``save_directory/encoder_path/`` and
    ``save_directory/query_encoder_path/``.

    Supports both symmetric (single encoder + SpladePooling) and asymmetric
    (Router with separate query/document pipelines) layouts.

    Subclasses can set :attr:`st_similarity` (default: ``"dot"``).
    """

    st_similarity: str = "dot"
    """The similarity function name."""

    def write_hub_extras(self, save_directory: Path):
        """Write sentence-transformers config files for HF Hub export."""
        # Directory names match __xpm_serialize__ mapping in SpladeModuleLoader
        DOC_DIR = "document_0_MLMTransformer"
        QUERY_DIR = "query_0_MLMTransformer"

        encoder_dir = save_directory / DOC_DIR
        has_query_encoder = (save_directory / QUERY_DIR).exists()

        # Read vocab_size from the saved encoder config
        config_path = encoder_dir / "config.json"
        vocab_size = None
        if config_path.exists():
            with open(config_path) as f:
                vocab_size = json.load(f).get("vocab_size")

        if has_query_encoder:
            self._write_asymmetric(save_directory, vocab_size)
        else:
            self._write_symmetric(save_directory, vocab_size)

        # config_sentence_transformers.json
        st_config = {
            "model_type": "SparseEncoder",
            "prompts": {},
            "default_prompt_name": None,
            "similarity_fn_name": self.st_similarity,
        }
        (save_directory / "config_sentence_transformers.json").write_text(
            json.dumps(st_config, indent=2)
        )

    def _write_symmetric(self, save_directory: Path, vocab_size):
        """Symmetric model: MLMTransformer + SpladePooling."""
        DOC_DIR = "document_0_MLMTransformer"
        modules = [
            {"idx": 0, "name": "0", "path": DOC_DIR, "type": MLM_TYPE},
            {
                "idx": 1,
                "name": "1",
                "path": "1_SpladePooling",
                "type": POOLING_TYPE,
            },
        ]
        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # SpladePooling config
        pooling_dir = save_directory / "1_SpladePooling"
        pooling_dir.mkdir(exist_ok=True)
        (pooling_dir / "config.json").write_text(
            json.dumps(
                {
                    "pooling_strategy": "max",
                    "activation_function": "relu",
                    "word_embedding_dimension": vocab_size,
                },
                indent=2,
            )
        )

        # sentence_bert_config.json
        (save_directory / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": 256, "do_lower_case": False}, indent=2)
        )

    def _write_asymmetric(self, save_directory: Path, vocab_size):
        """Asymmetric model: Router with query/document pipelines."""
        DOC_DIR = "document_0_MLMTransformer"
        QUERY_DIR = "query_0_MLMTransformer"

        # modules.json: single Router entry
        modules = [
            {"idx": 0, "name": "0", "path": "", "type": ROUTER_TYPE},
        ]
        (save_directory / "modules.json").write_text(json.dumps(modules, indent=2))

        # router_config.json
        router_config = {
            "types": {
                QUERY_DIR: MLM_TYPE,
                "query_1_SpladePooling": POOLING_TYPE,
                DOC_DIR: MLM_TYPE,
                "document_1_SpladePooling": POOLING_TYPE,
            },
            "structure": {
                "query": [QUERY_DIR, "query_1_SpladePooling"],
                "document": [DOC_DIR, "document_1_SpladePooling"],
            },
            "parameters": {
                "default_route": "document",
                "allow_empty_key": True,
            },
        }
        (save_directory / "router_config.json").write_text(
            json.dumps(router_config, indent=2)
        )

        # SpladePooling configs for both routes
        pooling_config = {
            "pooling_strategy": "max",
            "activation_function": "relu",
            "word_embedding_dimension": vocab_size,
        }
        for pooling_name in ["query_1_SpladePooling", "document_1_SpladePooling"]:
            pooling_dir = save_directory / pooling_name
            pooling_dir.mkdir(exist_ok=True)
            (pooling_dir / "config.json").write_text(
                json.dumps(pooling_config, indent=2)
            )

        # sentence_bert_config.json in each encoder dir
        sb_config = {"max_seq_length": 256, "do_lower_case": False}
        for enc_dir in [DOC_DIR, QUERY_DIR]:
            p = save_directory / enc_dir / "sentence_bert_config.json"
            p.write_text(json.dumps(sb_config, indent=2))

    def hub_readme_sections(self):
        """Return ST loading section, positioned before XPMIR usage."""
        from xpm_torch.module import ReadmeSection

        return [
            ReadmeSection(
                key="quick_loading",
                content=(
                    "## Loading with sentence-transformers\n\n"
                    "This model is also compatible with the "
                    "[sentence-transformers](https://www.sbert.net/) library:\n\n"
                    "```python\n"
                    "from sentence_transformers import SparseEncoder\n\n"
                    'model = SparseEncoder("YOUR_ORG/YOUR_MODEL")\n'
                    "```"
                ),
                before="usage",
            ),
        ]


class STModuleLoader(SimpleModuleLoader):
    """ModuleLoader for SentenceTransformers models (STCrossEncoder, STMultiVectorEncoder, etc.).

    Invert the order because model architecture is deferred to SentenceTransformers:
    load_model() is called first, and initialize() is called _after_ ST loading.
    """

    def execute(self):
        path = Path(self.path)
        logger.info(
            f"[{self.__class__.__name__}] Loading {self.value.__class__.__name__} model from disk: {path}"
        )
        self.value.load_model(path)
        self.value.initialize()


# Backwards compatibility alias
STCrossEncoderModuleLoader = STModuleLoader


class STCrossEncoder(AbstractModuleScorer):
    """A cross-encoder model leveraging the sentence-transformers library.

    It supports both direct raw text input and pre-tokenized input, utilizing the
    ST model's native tokenization and forward pass for consistency.

    Example:
        >>> from xpmir.neural.sentence_transformers import st_cross_scorer
        >>> model = st_cross_scorer(model_id="mixedbread-ai/mxbai-rerank-base-v1")
    """

    model_id: Param[str]
    """The HuggingFace model ID or path."""

    max_length: Param[Optional[int]] = field(default=None)
    """Maximum sequence length for tokenization."""

    pref_attn_implementation: Param[Optional[str]] = field(default=None)
    """Attention implementation to use (e.g. 'flash_attention_2', 'sdpa', or None)."""

    tokenizer: Param[Optional[HFTokenizer]] = field(default=None, ignore_default=True)
    """The tokenizer for the cross-scorer - if none use default from sentence-transformers (recommended)"""

    def setup_with_fabric(self, fabric) -> torch.nn.Module:
        """Sets up the module with PyTorch Lightning Fabric, checking precision compatibility.

        If Fabric runs in float32 precision, FlashAttention-2 is automatically downgraded to 'sdpa'.
        """
        fallback_fa2_if_incompatible_precision(self, fabric)
        return super().setup_with_fabric(fabric)

    def __post_init__(self):
        """called when instanciating the model, Before initialization"""
        super().__post_init__()
        self.st_model = None

        if self.max_length is None:
            # try to infer it from config
            self.max_length = get_default_max_len(self.model_id)
            if self.tokenizer is None:
                # if we have a custom tokenizer, it will handle max_len itself, so we don't need to warn
                logger.warning(
                    f"No max_len (or query/doc len) provided for STCrossEncoder, using default hf: {self.max_length}"
                )

    def _check_initialized(self):
        if self.st_model is None:
            raise RuntimeError(
                f"STCrossEncoder('{self.model_id}') has not been initialized (st_model is None). "
                f"You must call initialize() or execute an initialization task before using the model."
            )

    def __initialize__(self):
        super().__initialize__()
        if self.st_model is not None:
            self._initialized = True
            return

        if self.tokenizer is not None:
            self.tokenizer.initialize()

        attn_impl = self.pref_attn_implementation
        if attn_impl and "flash" in str(attn_impl).lower():
            from xpm_torch.utils.fabric import is_fa2_available

            if not is_fa2_available():
                logger.warning(
                    f"Flash Attention 2 requested ('{attn_impl}'), but is not supported in this environment "
                    f"(CUDA available: {torch.cuda.is_available()}, FA2 SM 8.0+ available: False). "
                    f"Falling back to `_attn_implementation='sdpa'`."
                )
                attn_impl = "sdpa"

        model_kwargs = {}
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
            logger.info(f"Using attention implementation: '{attn_impl}'")
        else:
            logger.info("Using default attention implementation (None specified)")

        try:
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            archs = getattr(config, "architectures", [])
            if not any("ForSequenceClassification" in arch for arch in archs):
                logger.warning(
                    f"No sequence classification head found in '{self.model_id}'. "
                    f"Are you sure you are loading a CrossEncoder? "
                    f"If you are fine-tuning from an encoder, use build_STCrossEncoder."
                )
        except Exception as e:
            logger.debug(f"Could not inspect config for '{self.model_id}': {e}")

        self.st_model = CrossEncoder(
            self.model_id,
            max_length=self.max_length,
            model_kwargs=model_kwargs,
        )

        actual_attn = getattr(self.st_model.model.config, "_attn_implementation", None)
        if not (actual_attn and "flash" in actual_attn.lower()):
            if attn_impl and "flash" in str(attn_impl).lower():
                logger.warning(
                    f"FA2 may not be active (attn_impl={actual_attn!r}); training will be slower."
                )

        self._initialized = True

    @property
    def device(self):
        self._check_initialized()
        try:
            return next(self.st_model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def save_model(self, path: Path):
        """Save the model in native SentenceTransformers format."""
        self._check_initialized()
        if path.suffix == ".safetensors":
            # ModuleLoader's default tries to save a safetensors file directly.
            # We intercept it to save the whole directory instead.
            path = path.parent
        self.st_model.save(str(path))

    def load_model(self, path: Path):
        """Load the model from a native SentenceTransformers directory."""
        path = Path(path)
        if path.is_file():
            path = path.parent
        self.model_id = str(path)

        if self.tokenizer is not None:
            self.tokenizer.initialize()

        attn_impl = self.pref_attn_implementation
        if attn_impl and "flash" in str(attn_impl).lower():
            from xpm_torch.utils.fabric import is_fa2_available

            if not is_fa2_available():
                logger.warning(
                    f"Flash Attention 2 requested ('{attn_impl}'), but is not supported in this environment "
                    f"(CUDA available: {torch.cuda.is_available()}, FA2 SM 8.0+ available: False). "
                    f"Falling back to `_attn_implementation='sdpa'`."
                )
                attn_impl = "sdpa"

        model_kwargs = {}
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.st_model = CrossEncoder(
            self.model_id,
            max_length=self.max_length,
            model_kwargs=model_kwargs if model_kwargs else None,
        )
        self._initialized = True

    def loader_config(
        self, path: Path, *, settings: Optional[Config] = None
    ) -> STCrossEncoderModuleLoader:
        """Returns an STCrossEncoderModuleLoader config for loading this model."""
        return STCrossEncoderModuleLoader.C(value=self, path=path, settings=settings)

    def get_tokenizer(self):
        """Returns a tokenizer for the cross-scorer (custom or ST native)."""
        if self.tokenizer is not None:
            return self.tokenizer
        self._check_initialized()
        return self.st_model.tokenizer

    def tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(input_records, options=options)

        self._check_initialized()

        # Prepare raw texts
        queries = [record["text_item"].text for record in input_records.unique_topics]
        documents = [
            record["text_item"].text for record in input_records.unique_documents
        ]

        # Reconstruct pairs
        pairs = []
        q_ix, d_ix = input_records.pairs()
        for qi, di in zip(q_ix, d_ix):
            pairs.append([queries[qi], documents[di]])

        # Route through the ST Transformer module's preprocess so any prompt /
        # chat template (e.g. Qwen3/mxbai generative rerankers) is applied
        # identically to predict().
        max_length = options.max_length if options else self.max_length
        processing_kwargs = (
            {"text": {"max_length": max_length, "truncation": True}}
            if max_length
            else None
        )
        r = self.st_model[0].preprocess(pairs, processing_kwargs=processing_kwargs)

        standard_keys = {"input_ids", "length", "attention_mask", "token_type_ids"}
        kwargs = {k: v for k, v in r.items() if k not in standard_keys}

        return TokenizedTexts(
            tokens=None,
            ids=r["input_ids"],
            lens=r.get("length", None),
            mask=r.get("attention_mask", None),
            token_type_ids=r.get("token_type_ids", None),
            kwargs=kwargs if kwargs else None,
        )

    def vocabulary_size(self) -> int:
        """Returns the size of the vocabulary used by the model's tokenizer."""
        self._check_initialized()
        return self.st_model.tokenizer.vocab_size

    def tok2id(self, tok: str) -> int:
        self._check_initialized()
        return self.st_model.tokenizer.convert_tokens_to_ids(tok)

    def id2tok(self, idx: int) -> str:
        self._check_initialized()
        return self.st_model.tokenizer.convert_ids_to_tokens(idx)

    def batch_tokenize(
        self,
        input_records: BaseItems,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        """Transform the text to tokens by using the tokenizer."""
        return self.tokenize(input_records, options=options)

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    def forward(
        self,
        inputs: Optional[BaseItems] = None,
        tokenized: Optional[TokenizedTexts] = None,
        **kwargs,
    ):
        self._check_initialized()

        if tokenized is None:
            assert inputs is not None, "Either inputs or tokenized must be provided"
            if self.tokenizer is not None:
                tokenized = self.batch_tokenize(inputs)
            else:
                # Extract raw pairs from inputs
                queries = [record["text_item"].text for record in inputs.unique_topics]
                documents = [
                    record["text_item"].text for record in inputs.unique_documents
                ]
                pairs = []
                q_ix, d_ix = inputs.pairs()
                for qi, di in zip(q_ix, d_ix):
                    pairs.append([queries[qi], documents[di]])

                # Use raw ST predict function
                scores = self.st_model.predict(
                    pairs,
                    batch_size=len(pairs),
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )

                return to_device(scores, self.device)

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            features = {
                "input_ids": to_device(tokenized.ids, self.device),
            }
            if tokenized.mask is not None:
                features["attention_mask"] = to_device(tokenized.mask, self.device)
            if tokenized.token_type_ids is not None:
                features["token_type_ids"] = to_device(
                    tokenized.token_type_ids, self.device
                )
            if getattr(tokenized, "kwargs", None) is not None:
                for k, v in tokenized.kwargs.items():
                    features[k] = (
                        to_device(v, self.device) if isinstance(v, torch.Tensor) else v
                    )

            # Run the ST module pipeline so that CausalLM-based generative
            # rerankers (5.4+) get their LogitScore reduction applied;
            # classification CEs keep the same head-logit semantics.
            output = self.st_model(features)
            result = output["scores"]

            # st_model.forward() only runs the module pipeline and does not apply activation_fn
            # (which is handled inside st_model.predict()). Apply it here for parity when tokenized inputs are provided.
            if self.st_model.activation_fn is not None:
                result = self.st_model.activation_fn(result)

            # predict() returns scores without a trailing singleton dim
            if result.ndim > 1 and result.shape[-1] == 1:
                result = result.squeeze(-1)

        return result


def st_cross_scorer(
    model_id: str,
    max_length: Optional[int] = None,
    max_query_length: Optional[int] = None,
    max_doc_length: Optional[int] = None,
    pref_attn_implementation: Optional[str] = None,
    tokenizer: Optional[HFTokenizer] = None,
) -> STCrossEncoder:
    """Creates an STCrossEncoder model.

    :param model_id: The HuggingFace model ID
    :param max_length: Maximum sequence length
    :param max_query_length: Maximum query sequence length
    :param max_doc_length: Maximum document sequence length
    :param pref_attn_implementation: Preferred attention implementation
    :param tokenizer: Optional custom tokenizer configuration overriding ST native tokenization
    :returns: (STCrossEncoder, init_tasks)
    """
    default_max_len = get_default_max_len(model_id)

    max_len = None
    if max_length is not None:
        if max_length > default_max_len:
            logging.warning(
                f"Provided max_length {max_length} is larger than default max_length {default_max_len} for {model_id}."
            )
        max_len = min(default_max_len, max_length)

    if max_query_length is not None and max_doc_length is not None:
        combined_bound = min(default_max_len, max_query_length + max_doc_length + 3)
        if max_len is None or max_len > combined_bound:
            max_len = combined_bound
    elif (max_query_length is not None) ^ (max_doc_length is not None):
        if max_len is None:
            raise ValueError(
                f"Incomplete sequence length configuration for model '{model_id}': "
                f"both max_query_length ({max_query_length}) and max_doc_length ({max_doc_length}) "
                "must be specified when max_length is not provided."
            )

    if max_len is None:
        logging.warning(
            f"No max_length (or query/doc len) provided for STCrossEncoder {model_id}. "
            f"Using default model max_len: {default_max_len}"
        )
        max_len = default_max_len

    if tokenizer is None and (
        max_query_length is not None or max_doc_length is not None
    ):
        from xpmir.neural.huggingface import HFQueryDocTokenizer

        tokenizer = HFQueryDocTokenizer.C(
            model_id=model_id,
            max_length=max_len,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
            pref_attn_implementation=pref_attn_implementation,
        )

    scorer = STCrossEncoder.C(
        model_id=model_id,
        max_length=max_len,
        pref_attn_implementation=pref_attn_implementation,
        tokenizer=tokenizer,
    )
    return scorer


def _extract_text(record: Union[IDTextRecord, str, dict]) -> str:
    """Extracts raw text string from an IDTextRecord, dict, or string."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict) and "text_item" in record:
        item = record["text_item"]
        return getattr(item, "text", item) if not isinstance(item, str) else item
    if hasattr(record, "text"):
        return record.text
    return str(record)


def _extract_text_pairs(
    inputs: Union[BaseItems, List[Any]],
) -> Tuple[List[str], List[str], bool]:
    """Extracts query and document texts from inputs (BaseItems or list of items).

    Returns:
        Tuple of (queries, documents, is_product)
    """
    if hasattr(inputs, "unique_topics"):
        queries = [_extract_text(record) for record in inputs.unique_topics]
        documents = [_extract_text(record) for record in inputs.unique_documents]
        if isinstance(inputs, ProductItems) or getattr(inputs, "is_product", False):
            return queries, documents, True
        q_ix, d_ix = inputs.pairs()
        return [queries[qi] for qi in q_ix], [documents[di] for di in d_ix], False

    if isinstance(inputs, (list, tuple)):
        paired_queries = []
        paired_documents = []
        for item in inputs:
            if hasattr(item, "query") and hasattr(item, "document"):
                q, d = item.query, item.document
            elif hasattr(item, "topic") and hasattr(item, "document"):
                q, d = item.topic, item.document
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                q, d = item[0], item[1]
            elif isinstance(item, dict):
                q = item.get("query", item.get("topic"))
                d = item.get("document", item.get("doc"))
            else:
                raise ValueError(
                    f"Cannot extract query and document from item of type {type(item)}: {item}"
                )

            paired_queries.append(_extract_text(q))
            paired_documents.append(_extract_text(d))
        return paired_queries, paired_documents, False

    raise ValueError(
        f"Unsupported inputs type for STMultiVectorEncoder: {type(inputs)}"
    )


class STMultiVectorEncoder(AbstractModuleScorer):
    """A multi-vector / late-interaction encoder model leveraging sentence-transformers (v6+).

    Supports ColBERT-style token-level MaxSim scoring and integrates directly with Fast-PLAID indexing.

    Example:
        >>> from xpmir.neural.sentence_transformers import st_multivector_scorer
        >>> model = st_multivector_scorer(model_id="answerdotai/answerai-colbert-small-v1")
    """

    model_id: Param[str]
    """The HuggingFace model ID or path."""

    dim: Param[Optional[int]] = field(default=None)
    """Projection dimension of the per-token vectors (if None, inferred from model)."""

    query_maxlen: Param[Optional[int]] = field(default=32)
    """Maximum sequence length for queries."""

    doc_maxlen: Param[Optional[int]] = field(default=300)
    """Maximum sequence length for documents."""

    pref_attn_implementation: Param[Optional[str]] = field(default=None)
    """Attention implementation to use (e.g. 'flash_attention_2', 'sdpa', or None)."""

    similarity_fn_name: Param[Optional[str]] = field(default="maxsim")
    """Similarity function name to use ('meanmaxsim' or 'maxsim'). Defaults to 'maxsim'."""

    query_prompt: Param[Optional[str]] = field(default="[Q] ")
    """Optional query prompt prefix override (e.g. '[Q] ' or '[unused0] ')."""

    doc_prompt: Param[Optional[str]] = field(default="[D] ")
    """Optional document prompt prefix override (e.g. '[D] ' or '[unused1] ')."""

    query_expansion: Param[Optional[Dict[str, Union[str, int, bool]]]] = field(
        default=None
    )
    """ColBERT query expansion configuration dict, e.g. {'strategy': 'fixed', 'attend': False, 'token': '[MASK]', 'length': 32}."""

    skiplist_words: Param[Optional[List[str]]] = field(default=None)
    """List of words/tokens to drop from document scoring (defaults to string.punctuation for ColBERT)."""

    normalize_embeddings: Param[bool] = field(default=True)
    """Whether to L2-normalize token embeddings."""

    def setup_with_fabric(self, fabric) -> torch.nn.Module:
        """Sets up the module with PyTorch Lightning Fabric, checking precision compatibility."""
        fallback_fa2_if_incompatible_precision(self, fabric)
        return super().setup_with_fabric(fabric)

    def __post_init__(self):
        super().__post_init__()
        self.st_model = None

    def _check_initialized(self):
        if self.st_model is None:
            raise RuntimeError(
                f"STMultiVectorEncoder('{self.model_id}') has not been initialized (st_model is None). "
                f"You must call initialize() or execute an initialization task before using the model."
            )

    def __initialize__(self):
        super().__initialize__()
        if self.st_model is not None:
            self._initialized = True
            return

        attn_impl = self.pref_attn_implementation
        if attn_impl and "flash" in str(attn_impl).lower():
            from xpm_torch.utils.fabric import is_fa2_available

            if not is_fa2_available():
                logger.warning(
                    f"Flash Attention 2 requested ('{attn_impl}'), but is not supported in this environment "
                    f"(CUDA available: {torch.cuda.is_available()}, FA2 SM 8.0+ available: False). "
                    f"Falling back to `_attn_implementation='sdpa'`."
                )
                attn_impl = "sdpa"

        model_kwargs = {}
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
            logger.info(f"Using attention implementation: '{attn_impl}'")
        else:
            logger.info("Using default attention implementation (None specified)")

        init_kwargs = {
            "model_kwargs": model_kwargs if model_kwargs else None,
        }
        sim_name = (
            self.similarity_fn_name.lower() if self.similarity_fn_name else "maxsim"
        )
        init_kwargs["similarity_fn_name"] = sim_name

        self.st_model = MultiVectorEncoder(self.model_id, **init_kwargs)

        # Inspect checkpoint / base model configuration
        loaded_prompts = getattr(self.st_model, "prompts", {}) or {}
        loaded_qe = (
            getattr(self.st_model[0], "query_expansion", None)
            if len(self.st_model) > 0
            else None
        )
        loaded_skip = (
            getattr(self.st_model[2], "skiplist_words", [])
            if len(self.st_model) > 2 and hasattr(self.st_model[2], "skiplist_words")
            else []
        )
        loaded_sim = getattr(self.st_model, "similarity_fn_name", None)

        def apply_override(
            attr_name: str, self_val, ckpt_val, default_val=None, setter_fn=None
        ):
            has_ckpt_val = ckpt_val not in (None, "", [], {})
            if has_ckpt_val:
                if self_val is not None and self_val != ckpt_val:
                    logger.warning(
                        f"[STMultiVectorEncoder Config Override] Overriding checkpoint parameter '{attr_name}' "
                        f"from '{ckpt_val}' (loaded from {self.model_id}) to '{self_val}' "
                        f"(specified on {self.__class__.__name__}). "
                        "This creates an architectural/configuration discrepancy with pre-trained weights and may require retraining!"
                    )
                    final_val = self_val
                else:
                    final_val = ckpt_val
            else:
                final_val = self_val if self_val is not None else default_val

            if setter_fn is not None:
                setter_fn(final_val)
            setattr(self, attr_name, final_val)
            return final_val

        apply_override(
            "query_prompt",
            self.query_prompt,
            loaded_prompts.get("query"),
            default_val="[Q] ",
            setter_fn=lambda v: (
                self.st_model.prompts.update({"query": v})
                if hasattr(self.st_model, "prompts")
                else None
            ),
        )
        apply_override(
            "doc_prompt",
            self.doc_prompt,
            loaded_prompts.get("document"),
            default_val="[D] ",
            setter_fn=lambda v: (
                self.st_model.prompts.update({"document": v})
                if hasattr(self.st_model, "prompts")
                else None
            ),
        )

        # Normalize query_expansion if provided before applying override
        if self.query_expansion is True:
            self.query_expansion = {
                "strategy": "fixed",
                "attend": False,
                "token": "[MASK]",
                "length": self.query_maxlen or 32,
            }
        elif isinstance(self.query_expansion, dict):
            qe_dict = dict(self.query_expansion)
            if "strategy" not in qe_dict:
                qe_dict["strategy"] = "fixed"
            if "token" not in qe_dict:
                qe_dict["token"] = "[MASK]"
            qe_dict["attend"] = bool(qe_dict.get("attend", False))
            if "length" not in qe_dict:
                qe_dict["length"] = int(self.query_maxlen or 32)
            else:
                qe_dict["length"] = int(qe_dict["length"])
                if self.query_maxlen and qe_dict["length"] != self.query_maxlen:
                    logger.warning(
                        f"query_expansion length ({qe_dict['length']}) differs from query_maxlen ({self.query_maxlen}). "
                        f"Updating query_maxlen to match query_expansion length."
                    )
            self.query_maxlen = qe_dict["length"]
            self.query_expansion = qe_dict

        apply_override(
            "query_expansion",
            self.query_expansion,
            loaded_qe,
            default_val=None,
            setter_fn=lambda v: (
                setattr(self.st_model[0], "query_expansion", v)
                if len(self.st_model) > 0
                else None
            ),
        )

        def _set_skiplist(v):
            if len(self.st_model) > 2 and hasattr(self.st_model[2], "skiplist_words"):
                self.st_model[2].skiplist_words = list(v) if v else []
                if (
                    hasattr(self.st_model, "tokenizer")
                    and self.st_model.tokenizer is not None
                ):
                    self.st_model[2].resolve_with_tokenizer(self.st_model.tokenizer)

        apply_override(
            "skiplist_words",
            self.skiplist_words,
            loaded_skip,
            default_val=list(string.punctuation),
            setter_fn=_set_skiplist,
        )

        apply_override(
            "similarity_fn_name",
            self.similarity_fn_name.lower() if self.similarity_fn_name else "maxsim",
            loaded_sim,
            default_val="maxsim",
            setter_fn=lambda v: setattr(
                self.st_model, "similarity_fn_name", v.lower() if v else "maxsim"
            ),
        )

        self._initialized = True

    @property
    def model(self) -> torch.nn.Module:
        """Underlying HuggingFace transformer model (for Fabric FA2 inspection and PlaidIndexBuilder)."""
        self._check_initialized()
        if hasattr(self.st_model, "transformers_model"):
            return self.st_model.transformers_model
        if len(self.st_model) > 0 and hasattr(self.st_model[0], "model"):
            return self.st_model[0].model
        return self.st_model

    @property
    def dimension(self) -> int:
        """Token embedding dimension."""
        if self.dim is not None:
            return self.dim
        self._check_initialized()
        return self.st_model.get_embedding_dimension()

    @property
    def device(self) -> torch.device:
        self._check_initialized()
        try:
            return next(self.st_model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def save_model(self, path: Path):
        """Save the model in native SentenceTransformers format."""
        self._check_initialized()
        path = Path(path)
        if path.suffix in {".pth", ".safetensors", ".bin"}:
            path = path.parent
        self.st_model.save(str(path))

    def load_model(self, path: Path):
        """Load the model from a native SentenceTransformers directory."""
        path = Path(path)
        if path.is_file():
            path = path.parent
        self.model_id = str(path)
        self.st_model = None
        self.__initialize__()

    def loader_config(
        self, path: Path, *, settings: Optional[Config] = None
    ) -> STModuleLoader:
        """Returns an STModuleLoader config for loading this model."""
        return STModuleLoader.C(value=self, path=path, settings=settings)

    # ------------------------------------------------------------------ Tokenizer
    def get_tokenizer(self):
        """Returns the underlying tokenizer from the SentenceTransformer model."""
        self._check_initialized()
        return self.st_model.tokenizer

    def vocabulary_size(self) -> int:
        """Returns the size of the vocabulary used by the model's tokenizer."""
        self._check_initialized()
        return self.st_model.tokenizer.vocab_size

    def tok2id(self, tok: str) -> int:
        self._check_initialized()
        return self.st_model.tokenizer.convert_tokens_to_ids(tok)

    def id2tok(self, idx: int) -> str:
        self._check_initialized()
        return self.st_model.tokenizer.convert_ids_to_tokens(idx)

    def batch_tokenize(
        self,
        input_records: Union[BaseItems, List[Any]],
        options: Optional[TokenizerOptions] = None,
    ) -> dict:
        """Tokenize query and document texts in parallel worker processes."""
        self._check_initialized()
        queries, documents, is_product = _extract_text_pairs(input_records)
        # For queries: if query_expansion is active, SentenceTransformers Transformer module
        # enforces padding='max_length' to expansion['length']. Passing max_length in processing_kwargs
        # triggers a warning: "processing_kwargs overrides the text padding or max_length...".
        # Therefore, only pass processing_kwargs when query_expansion is NOT active!
        q_kwargs = {}
        if self.query_maxlen and not self.query_expansion:
            q_kwargs = {
                "processing_kwargs": {
                    "text": {"max_length": self.query_maxlen, "truncation": True}
                }
            }
        d_kwargs = (
            {
                "processing_kwargs": {
                    "text": {"max_length": self.doc_maxlen, "truncation": True}
                }
            }
            if self.doc_maxlen
            else {}
        )
        q_prompt = self.query_prompt or (
            self.st_model._resolve_prompt(None, "query")
            if hasattr(self.st_model, "_resolve_prompt")
            else None
        )
        d_prompt = self.doc_prompt or (
            self.st_model._resolve_prompt(None, "document")
            if hasattr(self.st_model, "_resolve_prompt")
            else None
        )
        q_feat = self.st_model.preprocess(
            queries, prompt=q_prompt, task="query", **q_kwargs
        )
        d_feat = self.st_model.preprocess(
            documents, prompt=d_prompt, task="document", **d_kwargs
        )
        return {"query": q_feat, "document": d_feat, "is_product": is_product}

    def get_tokenizer_fn(self):
        return self.batch_tokenize

    # ------------------------------------------------------------------ Fast-PLAID
    def document_token_embeddings(
        self, records: List[IDTextRecord], max_doc_length: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Encodes documents into a list of per-token embedding tensors."""
        self._check_initialized()
        texts = [_extract_text(r) for r in records]
        effective_max_doc = (
            max_doc_length if max_doc_length is not None else self.doc_maxlen
        )
        d_kwargs = {"batch_size": len(texts)}
        if effective_max_doc:
            d_kwargs["processing_kwargs"] = {
                "text": {"max_length": effective_max_doc, "truncation": True}
            }
        return self.st_model.encode_document(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
            device=self.device,
            show_progress_bar=False,
            **d_kwargs,
        )

    def query_token_embeddings(
        self, records: List[IDTextRecord], max_query_length: Optional[int] = None
    ) -> torch.Tensor:
        """Encodes queries into a batch tensor of token embeddings."""
        self._check_initialized()
        texts = [_extract_text(r) for r in records]
        effective_max_q = (
            max_query_length if max_query_length is not None else self.query_maxlen
        )
        q_kwargs = {}
        if effective_max_q and not self.query_expansion:
            q_kwargs["processing_kwargs"] = {
                "text": {"max_length": effective_max_q, "truncation": True}
            }
        embs = self.st_model.encode_query(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=False,
            device=self.device,
            show_progress_bar=False,
            **q_kwargs,
        )
        if isinstance(embs, torch.Tensor):
            if embs.ndim == 2:
                return embs.unsqueeze(0)
            return embs
        return torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)

    # ------------------------------------------------------------------ Scoring
    def score_product(
        self,
        queries: Union[List[torch.Tensor], torch.Tensor],
        documents: Union[List[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Compute all-pairs MaxSim matrix between queries and documents."""
        self._check_initialized()
        return self.st_model.similarity(queries, documents, **kwargs)

    def score_pairs(
        self,
        queries: Union[List[torch.Tensor], torch.Tensor],
        documents: Union[List[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Compute pairwise MaxSim scores for paired queries and documents."""
        self._check_initialized()
        return self.st_model.similarity_pairwise(queries, documents, **kwargs)

    def forward(
        self,
        inputs: Optional[Union[BaseItems, List[Any]]] = None,
        tokenized: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass conforming to the xpmir.letor contract using pure ST native MaxSim."""
        self._check_initialized()

        if (
            tokenized is not None
            and isinstance(tokenized, dict)
            and "query" in tokenized
        ):
            q_feat = tokenized["query"]
            d_feat = tokenized["document"]
            is_product = tokenized.get("is_product", False)
        else:
            assert inputs is not None, "Either inputs or tokenized must be provided"
            features = self.batch_tokenize(inputs)
            q_feat, d_feat, is_product = (
                features["query"],
                features["document"],
                features["is_product"],
            )

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            q_feat = {
                k: to_device(v, self.device) if isinstance(v, torch.Tensor) else v
                for k, v in q_feat.items()
            }
            d_feat = {
                k: to_device(v, self.device) if isinstance(v, torch.Tensor) else v
                for k, v in d_feat.items()
            }

            q_out = self.st_model(q_feat, task="query")
            d_out = self.st_model(d_feat, task="document")

            if is_product:
                scores = self.st_model.similarity(
                    q_out["token_embeddings"],
                    d_out["token_embeddings"],
                    a_mask=q_out.get("attention_mask", q_feat.get("attention_mask")),
                    b_mask=d_out.get("attention_mask", d_feat.get("attention_mask")),
                )
                scores = to_device(scores.flatten(), self.device)
            else:
                scores = self.st_model.similarity_pairwise(
                    q_out["token_embeddings"],
                    d_out["token_embeddings"],
                    a_mask=q_out.get("attention_mask", q_feat.get("attention_mask")),
                    b_mask=d_out.get("attention_mask", d_feat.get("attention_mask")),
                )
                scores = to_device(scores, self.device)

            return scores


def st_multivector_scorer(
    model_id: str,
    dim: Optional[int] = None,
    query_maxlen: Optional[int] = 32,
    doc_maxlen: Optional[int] = 300,
    pref_attn_implementation: Optional[str] = None,
    similarity_fn_name: Optional[str] = "maxsim",
    query_prompt: Optional[str] = "[Q] ",
    doc_prompt: Optional[str] = "[D] ",
    query_expansion: Optional[Union[Dict[str, Union[str, int, bool]], bool]] = None,
    skiplist_words: Optional[List[str]] = None,
    normalize_embeddings: bool = True,
) -> STMultiVectorEncoder:
    """Creates an STMultiVectorEncoder model.

    :param model_id: The HuggingFace model ID or path
    :param dim: The embedding dimension (optional, inferred by default)
    :param query_maxlen: Maximum sequence length for queries
    :param doc_maxlen: Maximum sequence length for documents
    :param pref_attn_implementation: Preferred attention implementation ('sdpa', 'flash_attention_2')
    :param similarity_fn_name: Similarity function ('maxsim' or 'meanmaxsim')
    :param query_prompt: Optional prompt prefix for queries
    :param doc_prompt: Optional prompt prefix for documents
    :param query_expansion: Query expansion configuration or bool
    :param skiplist_words: Optional list of words/punctuation to skip in similarity
    :param normalize_embeddings: Whether to normalize embeddings
    :returns: Configured STMultiVectorEncoder
    """
    # Normalize query_expansion to Dict or None
    qe_dict: Optional[Dict[str, Union[str, int, bool]]] = None
    if query_expansion is True:
        qe_dict = {
            "strategy": "fixed",
            "attend": False,
            "token": "[MASK]",
            "length": query_maxlen or 32,
        }
    elif isinstance(query_expansion, dict):
        qe_dict = dict(query_expansion)

    return STMultiVectorEncoder.C(
        model_id=model_id,
        dim=dim,
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
        pref_attn_implementation=pref_attn_implementation,
        similarity_fn_name=similarity_fn_name,
        query_prompt=query_prompt,
        doc_prompt=doc_prompt,
        query_expansion=qe_dict,
        skiplist_words=skiplist_words,
        normalize_embeddings=normalize_embeddings,
    )
