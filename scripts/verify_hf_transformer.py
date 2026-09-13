"""Verify a converted self-attention Hugging Face repository against its checkpoint.

The verified export can be evaluated with the official BabyLM 2026 pipeline:
https://github.com/babylm-org/babylm-eval
"""

import argparse
import pathlib
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from models.attention.transformer.config import SelfAttentionLMConfig  # noqa: E402
from models.attention.transformer.lm import SelfAttentionDecoderLM  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a converted Transformer HF repo.")
    parser.add_argument("--hf-dir", required=True, type=pathlib.Path)
    parser.add_argument("--experiment-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-label", default="9")
    return parser.parse_args()


def build_live_model(
    experiment_dir: pathlib.Path, checkpoint_label: str
) -> SelfAttentionDecoderLM:
    config_dict = yaml.safe_load(
        (experiment_dir / "logging" / "model_config.yaml").read_text(encoding="utf-8")
    )
    model = SelfAttentionDecoderLM(SelfAttentionLMConfig(**config_dict))
    state_path = experiment_dir / "checkpoints" / f"checkpoint_{checkpoint_label}" / "model.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def main() -> None:
    args = _parse_arguments()

    live_model = build_live_model(args.experiment_dir, args.checkpoint_label)
    hf_model = AutoModelForCausalLM.from_pretrained(str(args.hf_dir), trust_remote_code=True)
    hf_model.eval()

    torch.manual_seed(0)
    vocab_size = live_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        live_logits, _ = live_model(input_ids, attention_mask=attention_mask)
        hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask)

    hf_logits = hf_output["logits"]
    max_abs_diff = (hf_logits - live_logits).abs().max().item()
    print(f"logits shape: {tuple(hf_logits.shape)}")
    print(f"max abs logit diff vs live checkpoint: {max_abs_diff:.3e}")
    if not torch.allclose(hf_logits, live_logits, atol=1e-4):
        raise RuntimeError("HF logits diverge from the checkpoint")

    hf_base_model = AutoModel.from_pretrained(str(args.hf_dir), trust_remote_code=True)
    hf_base_model.eval()

    with torch.no_grad():
        _, live_hidden = live_model.encode_for_objective(input_ids, attention_mask=attention_mask)
        hf_base_output = hf_base_model(input_ids=input_ids, attention_mask=attention_mask)

    hf_hidden = hf_base_output["last_hidden_state"]
    max_abs_hidden_diff = (hf_hidden - live_hidden).abs().max().item()
    print(f"hidden shape: {tuple(hf_hidden.shape)}")
    print(f"max abs hidden diff vs live checkpoint: {max_abs_hidden_diff:.3e}")
    if not torch.allclose(hf_hidden, live_hidden, atol=1e-4):
        raise RuntimeError("HF AutoModel hidden states diverge from the checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_dir), trust_remote_code=True)
    encoded = tokenizer("The boy found a", return_tensors="pt")
    with torch.no_grad():
        sentence_logits = hf_model(**encoded)["logits"]
    print(f"tokenizer ok; sentence logits shape: {tuple(sentence_logits.shape)}")

    print("\nPASS: HF repo loads via trust_remote_code and matches the checkpoint.")


if __name__ == "__main__":
    main()
