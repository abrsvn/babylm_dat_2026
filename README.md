# Relational BabyLM

Source release accompanying the paper **"Relational Attention for
Data-Efficient Language Modeling" (Adrian Brasoveanu, Ece Takmaz,
Jakub Dotlačil — BabyLM Workshop, EMNLP 2026)**. Contains decoder-only
Dual Attention Transformer (DAT; our own reimplementation of Altabaa and
Lafferty, 2025) and standard self-attention transformer language models,
a training pipeline with next-token prediction and Next Latent auxiliary
loss support (see paper for more details), Hugging Face export wrappers,
example configurations for the reported model families, the analysis pipeline
behind the paper's tables and statistical models, and a pytest suite.

**Paper PDF**: [Relational Attention for Data-Efficient Language Modeling](./relational_babylm_2026_paper.pdf)

**Abstract**:
> We present Relational BabyLM, a system submission to the BabyLM 2026 challenge that combines two cognitively motivated inductive biases in a single decoder-only Transformer. Architecturally, we replace standard self-attention with a Dual Attention Transformer (DAT), which separates the routing of object-level ("sensory") lexical features from structural/relational information (Altabaa and Lafferty, 2025; Altabaa et al., 2024; Webb et al., 2024; Kerg et al., 2022; Webb et al., 2021). Relational attention (RA) disentangled from self-attention greatly increases data efficiency and out-of-training-sample generalization on purely relational tasks, but language modeling requires object-level and relational information to be integrated as well as disentangled, and RA-based LMs have remained largely unexplored. BabyLM's data-constrained training and comprehensive evaluation is an ideal testing ground for whether that data efficiency transfers. As a training intervention, we add a Next-Latent Prediction (NextLat; Teoh et al. 2026) objective that encourages hidden states to compress history incrementally into a dense belief state. Architecture is the dominant factor for structural linguistic generalization; the objective is secondary but still significant. DAT's three relational attention types (full RA vs. the simpler RCA and DisRCA variants) are largely interchangeable at 10M words; full RA pulls ahead at 100M. We also introduce a novel symbol-retrieval mechanism (RoPE-based, as opposed to learned, relative symbols) that matches learned symbol libraries while adding no parameters. On the strict (100M-word) track, our best model ranks 6th of 55 overall and 3rd of 55 on the leaderboard's NLP-task subset at the time of writing; our two strongest models outperform the GPT-2 baseline on most benchmarks, with one attaining the highest EWoK score among strict-track entries.

## Repository Layout

- `models/`: DAT and self-attention decoder-only language models.
- `train/`: configuration, corpus serialization, cache construction,
  checkpointing, objectives, optimizers, and training loop. Includes
  `portable_config.py`, which rewrites a saved train config into a
  shareable form with placeholder asset paths (used by the analysis
  export tooling).
- `hf_export/`: Hugging Face model wrappers used by checkpoint export.
- `scripts/`: small reusable utilities (tokenizer construction, Hugging
  Face conversion, verification).
- `configs/`: example configurations with explicit asset-path
  placeholders. Four strict-small examples cover the reported comparison
  families (`dat_rca_ntp/`, `dat_rca_nextlat/`, `self_attention_baseline/`,
  `dat_rca_rope_symbols/`), and `submitted_16l_wide/` and `submitted_18l/`
  hold the Phase-B configurations of the two submitted strict-track models.
  The full run matrix of the paper's comparisons is tabulated in the paper's
  configuration appendix.
- `analysis/`: the Python and R pipeline that produced the paper's tables,
  figures, and mixed-effects models (see `analysis/README.md`).
- `test/`: real pytest coverage using small local fixtures only.

## Installation

```bash
pip install -r requirements.txt
```

## Assets

This source release contains no corpora, tokenizer assets, checkpoints, or
evaluation data. All runtime assets are acquired externally and passed as
explicit paths:

- Training corpora: obtained under BabyLM terms; supply the directory
  containing the six official training corpus files via --train_data_dir.
- Tokenizer: a 16,384-token BPE tokenizer with PAD/BOS/EOS IDs 3/1/2.
  Build one with scripts/create_tokenizer.py and supply its directory via
  --tokenizer_dir.
- Evaluation data: required by the official evaluator and by the analysis
  pipeline's --eval_data_root; the official evaluation data directory.

## Training

Command-line:

```bash
  python -m train.cli \
    --tokenizer_dir /path/to/tokenizer \
    --train_data_dir /path/to/babylm-training-corpus \
    --experiment_name my_run
```

YAML config:

```bash
  python -m train.cli_yaml \
    --train-config configs/dat_rca_ntp/train_config.yaml \
    --model-config configs/dat_rca_ntp/model_config.yaml
```

Training optimizes next-token prediction. Pass --nextlat_enabled to add the
NextLat auxiliary objective. The LM head is tied for next-token prediction
and untied for NextLat; pass `--tie_lm_head true` or `--tie_lm_head false`
to override the derivation. Checkpoints are written per epoch under
experiments/<name>/checkpoints/, plus word-budget milestone checkpoints
(e.g. 100M) during training.

The configs/ directory holds six examples, each a matched train and
model config pair for `train.cli_yaml`: `dat_rca_ntp`, `dat_rca_nextlat`,
`self_attention_baseline`, and `dat_rca_rope_symbols` cover the
architecture, objective, and symbol-retrieval comparisons on strict-small;
`submitted_16l_wide/` and `submitted_18l/` hold the submitted strict-track
models' Phase-B configurations. All have placeholder asset paths that must
be replaced with locally acquired BabyLM assets before running. The
complete settings of every run in the paper's comparisons are tabulated in
the configuration appendix of the paper.

### Submitted Models

The two submitted strict-track models were trained in two phases with
`python -m train.cli`. Phase A trains on line-EOS documents
(`--sequence_boundary_policy eos_document`); Phase B warm-starts from
Phase A's final epoch checkpoint and switches to document-boundary
packing. Checkpoint labels are 0-based epoch indices, so the warm-start
label is `checkpoint_<phase_A_epochs - 1>`. `--tokenizer_dir` and
`--train_data_dir` are user-supplied; the BabyLM corpora are not
redistributed here. Both models used a 16,384-token GPT-BERT tokenizer:
the 18L model the 2024 release (byte-identical to the official GPT-BERT
baseline tokenizer), the 16L wide model the 2026 retraining of the same
recipe. Both runs kept an exponential moving average of the weights
(`--ema_decay 0.999`) and warm-started Phase B from Phase A's EMA
checkpoint (`--init_from_use_ema`).

18L curriculum model (RCA, 9 SA / 3 RA heads, 768 hidden, 18 layers,
191M parameters):

```bash
# Phase A: 6 epochs, line-EOS boundary
python -m train.cli \
  --tokenizer_dir /path/to/tokenizer \
  --train_data_dir /path/to/babylm-strict-corpus \
  --experiment_name 18l_phase_a \
  --corpus_id strict --model_type dat \
  --hidden_dim 768 --n_layers 18 --n_heads_sa 9 --n_heads_ra 3 \
  --ra_type rca --ra_rel_activation identity \
  --pe_type rope --symbol_retrieval relative --relative_symbols_rope \
  --max_rel_pos 512 --norm_type layernorm --ffn_activation swiglu \
  --init_scheme normal_0_02_scaled_projection --dropout 0.1 --no_use_bias_out \
  --datapoint_length 512 \
  --nextlat_enabled --nextlat_horizon 1 \
  --nextlat_lambda_mse 1.0 --nextlat_lambda_kl 0.5 --nextlat_lambda_ce 0.0 \
  --optimizer muon --muon_aux_optimizer lambw \
  --muon_momentum 0.95 --muon_n_schulz_steps 5 \
  --lr_scheduler_type cosine_min_lr --min_lr_rate 0.1 --warmup_ratio 0.01 \
  --gradient_accumulation_steps 1 \
  --sequence_boundary_policy eos_document --n_epochs 6 \
  --muon_lr 0.02 --learning_rate 0.001 --weight_decay 0.01 \
  --batch_size 48 --seed 1 --ema_decay 0.999

# Phase B: 4 epochs, document-boundary, warm-started from
# Phase A's final epoch checkpoint (label checkpoint_5)
python -m train.cli \
  --tokenizer_dir /path/to/tokenizer \
  --train_data_dir /path/to/babylm-strict-corpus \
  --experiment_name 18l_phase_b \
  --corpus_id strict --model_type dat \
  --hidden_dim 768 --n_layers 18 --n_heads_sa 9 --n_heads_ra 3 \
  --ra_type rca --ra_rel_activation identity \
  --pe_type rope --symbol_retrieval relative --relative_symbols_rope \
  --max_rel_pos 512 --norm_type layernorm --ffn_activation swiglu \
  --init_scheme normal_0_02_scaled_projection --dropout 0.1 --no_use_bias_out \
  --datapoint_length 512 \
  --nextlat_enabled --nextlat_horizon 1 \
  --nextlat_lambda_mse 1.0 --nextlat_lambda_kl 0.5 --nextlat_lambda_ce 0.0 \
  --optimizer muon --muon_aux_optimizer lambw \
  --muon_momentum 0.95 --muon_n_schulz_steps 5 \
  --lr_scheduler_type cosine_min_lr --min_lr_rate 0.1 --warmup_ratio 0.01 \
  --gradient_accumulation_steps 1 \
  --sequence_boundary_policy document_boundary --n_epochs 4 \
  --muon_lr 0.008 --learning_rate 0.0004 --weight_decay 0.01 \
  --init_from_checkpoint experiments/18l_phase_a/checkpoints/checkpoint_5 \
  --init_from_use_ema \
  --batch_size 48 --seed 1 --ema_decay 0.999
```

16L wide model (RA, 12 SA / 4 RA heads, 1,024 hidden, 16 layers, 304M
parameters): the same two-phase invocations with
`--experiment_name 16l_phase_a` and `--experiment_name 16l_phase_b`,
`--hidden_dim 1024 --n_layers 16 --n_heads_sa 12 --n_heads_ra 4`,
`--ra_type ra --ra_n_relations 4`, `--batch_size 36`, and
`--weight_decay 0.05`; Phase A is `--n_epochs 2`, and Phase B is
`--n_epochs 8 --muon_lr 0.005 --learning_rate 0.00025` with
`--init_from_checkpoint experiments/16l_phase_a/checkpoints/checkpoint_1`.

Both models train for 10 total epochs. The complete settings
for every hyperparameter are in `configs/submitted_18l/` and
`configs/submitted_16l_wide/`. Trained checkpoints for both submitted
models are published on Hugging Face:
[babylm-dat-strict-nextlat-final](https://huggingface.co/abe123/babylm-dat-strict-nextlat-final)
(16L wide) and
[babylm-dat-strict](https://huggingface.co/abe123/babylm-dat-strict)
(18L curriculum).

## Evaluation

For BabyLM benchmark evaluation, use the official evaluator:

https://github.com/babylm-org/babylm-eval

## Testing

```bash
  python -m pytest test/
```

## License

Apache License 2.0. See LICENSE.
