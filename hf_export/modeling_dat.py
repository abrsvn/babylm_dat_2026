"""HuggingFace PreTrainedModel wrapper for the dual-attention (DAT) decoder LM.

Source-of-truth copy. scripts/convert_dat_to_hf.py copies it into a generated HF
repository together with configuration_dat.py and the flattened model source
(dat_config.py, dat_core.py, dat_symbols.py, dat_lm.py, transformer_core.py,
transformer_components.py).

Why every model module is imported directly here: with trust_remote_code on a
local directory, transformers 4.46.3 only copies the entry module's *direct*
relative imports into its dynamic-module cache (it does not recurse). Importing
all flattened modules here forces every one of them to be copied, after which
their own relative imports resolve because the files are co-located.
"""

import collections

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Force every flattened module to be copied into the dynamic-module cache.
from .attention_config import SUPPORTED_SEQUENCE_BOUNDARY_POLICIES  # noqa: F401
from .masks import build_decoder_attention_mask  # noqa: F401
from .transformer_components import FeedForward, PositionalEncoding  # noqa: F401
from .transformer_core import SelfAttention  # noqa: F401
from .dat_symbols import SymbolicAttentionRetriever  # noqa: F401
from .dat_core import RelationalAttention  # noqa: F401
from .dat_config import DatLMConfig
from .dat_lm import DatDecoderLM
from .configuration_dat import DatConfig, DAT_LM_FIELDS


def _refresh_rope_buffers(model: DatDecoderLM, rope_theta: float) -> None:
    pe = model.position_encoder
    if hasattr(pe, "pe_type") and pe.pe_type == "rope":
        cos, sin = pe._precompute_rope_freqs(pe.embedding_dim, pe.max_len, rope_theta)
        pe.register_buffer("rope_cos", cos, persistent=False)
        pe.register_buffer("rope_sin", sin, persistent=False)
        pe._rope_uninitialized = True


def _collect_shared_weight_keys(module: torch.nn.Module) -> dict[str, str]:
    # remove_duplicate=False is essential: by default named_parameters yields
    # each shared tensor only once, which would hide exactly the duplicates we
    # need to declare (and which state_dict still emits under every name).
    names_by_storage: dict[int, list[str]] = collections.defaultdict(list)
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names_by_storage[id(parameter)].append(name)
    shared_keys: dict[str, str] = {}
    for names in names_by_storage.values():
        if len(names) > 1:
            names = sorted(names)
            # Prefer token_embeddings.weight as the canonical source
            # (HF convention: output weights tied to input embeddings).
            source = names[0]
            for n in names:
                if n.endswith("token_embeddings.weight"):
                    source = n
                    break
            for target in names:
                if target != source:
                    shared_keys[target] = source
    return shared_keys


class _ExpandedTiedWeightsMixin:
    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        # Return the dynamically computed shared-weight mapping directly so
        # from_pretrained ties exactly the weights the model shares, whether
        # or not tie_word_embeddings is set. Mirrors modeling_transformer.py.
        return dict(self._tied_weights_keys)


class DatModel(_ExpandedTiedWeightsMixin, PreTrainedModel):
    config_class = DatConfig
    base_model_prefix = ""

    def __init__(self, config: DatConfig) -> None:
        super().__init__(config)
        dat_config = DatLMConfig(**{field: getattr(config, field) for field in DAT_LM_FIELDS})
        self.model = DatDecoderLM(dat_config)
        self._tied_weights_keys = _collect_shared_weight_keys(self)
        self.post_init()

    def post_init(self) -> None:
        super().post_init()
        # Work around accelerate/transformers leaving persistent=False buffers uninitialized
        _refresh_rope_buffers(self.model, self.config.rope_theta)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.token_embeddings

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.token_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        _, hidden_states = self.model.encode_for_objective(input_ids, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)


class DatForCausalLM(_ExpandedTiedWeightsMixin, PreTrainedModel):
    config_class = DatConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"model.lm_head.weight": "model.token_embeddings.weight"}

    def __init__(self, config: DatConfig) -> None:
        super().__init__(config)
        dat_config = DatLMConfig(**{field: getattr(config, field) for field in DAT_LM_FIELDS})
        self.model = DatDecoderLM(dat_config)
        self._tied_weights_keys = _collect_shared_weight_keys(self)
        self.post_init()

    def post_init(self) -> None:
        super().post_init()
        # Work around accelerate/transformers leaving persistent=False buffers uninitialized
        _refresh_rope_buffers(self.model, self.config.rope_theta)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.token_embeddings

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.token_embeddings = value

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: torch.nn.Module) -> None:
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        logits, _ = self.model(input_ids, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)


class DatForSequenceClassification(_ExpandedTiedWeightsMixin, PreTrainedModel):
    config_class = DatConfig
    base_model_prefix = "model"

    def __init__(self, config: DatConfig) -> None:
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", 2)
        dat_config = DatLMConfig(**{field: getattr(config, field) for field in DAT_LM_FIELDS})
        self.model = DatDecoderLM(dat_config)
        self.score = torch.nn.Linear(config.hidden_dim, self.num_labels, bias=False)
        self._tied_weights_keys = _collect_shared_weight_keys(self)
        self.post_init()

    def post_init(self) -> None:
        super().post_init()
        _refresh_rope_buffers(self.model, self.config.rope_theta)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.token_embeddings

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.token_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        _, hidden_states = self.model.encode_for_objective(input_ids, attention_mask=attention_mask)
        logits = self.score(hidden_states)

        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        else:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            if (non_pad_mask.sum(-1) == 0).any().item():
                raise ValueError("Cannot pool sequence-classification logits for all-padding sequences")
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if getattr(self.config, "problem_type", None) is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=hidden_states,
        )
