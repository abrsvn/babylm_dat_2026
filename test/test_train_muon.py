"""Pytest coverage for the Muon optimizer and Muon auxiliary branches."""

import math

import pytest
import torch
import torch.nn as nn

from train.config import TrainConfig, parse_train_config
from train.optim import (
    LambW,
    MuonWithAux,
    _adam_moment_update,
    _classify_muon_parameters,
    _is_muon_hidden_weight,
    _lambw_trust_ratio,
    _muon_update,
    _zeropower_via_newtonschulz5,
    build_optimizer,
)


# ---------------------------------------------------------------------------
# Newton-Schulz core
# ---------------------------------------------------------------------------


def test_ns_preserves_shape_and_is_finite() -> None:
    G = torch.randn(4, 6)
    result = _zeropower_via_newtonschulz5(G, 5, torch.float32)
    assert result.shape == G.shape
    assert torch.isfinite(result).all()


def test_ns_positive_scale_invariance() -> None:
    """Orthogonalization is scale-invariant: NS(10*G) == NS(G)."""
    torch.manual_seed(0)
    G = torch.randn(3, 5)
    result1 = _zeropower_via_newtonschulz5(G, 5, torch.float32)
    result2 = _zeropower_via_newtonschulz5(10 * G, 5, torch.float32)
    assert torch.allclose(result1, result2, atol=1e-5)


def test_ns_transpose_equivariance() -> None:
    """NS(G.T).T == NS(G): the orthogonalization commutes with transposition."""
    torch.manual_seed(0)
    G = torch.randn(4, 7)
    result_G = _zeropower_via_newtonschulz5(G, 5, torch.float32)
    result_GT = _zeropower_via_newtonschulz5(G.T, 5, torch.float32)
    rel_diff = (result_G.T - result_GT).norm() / result_G.norm()
    assert rel_diff.item() < 1e-5


def test_ns_batched_3d_input() -> None:
    """The NS iteration must support batched 3D matrices (e.g. wr_proj)."""
    G = torch.randn(2, 4, 4)
    result = _zeropower_via_newtonschulz5(G, 5, torch.float32)
    assert result.shape == G.shape
    assert torch.isfinite(result).all()


def test_muon_update_preserves_shape() -> None:
    grad = torch.randn(8, 3)
    momentum = torch.zeros_like(grad)
    update = _muon_update(grad, momentum, 0.95, 5)
    assert update.shape == grad.shape
    assert torch.isfinite(update).all()


def test_muon_update_does_not_mutate_grad() -> None:
    grad = torch.randn(4, 4)
    grad_pre = grad.clone()
    momentum = torch.zeros_like(grad)
    _muon_update(grad, momentum, 0.95, 5)
    assert torch.allclose(grad, grad_pre), "Muon update mutated grad"


def test_muon_update_aspect_ratio_scaling() -> None:
    """Compare update for G against transposed update for G.T.

    The aspect-ratio scaling multiplies tall matrices (rows > cols) by
    sqrt(rows/cols). For a wide matrix G.T the scaling is 1. So the
    norm ratio of update(G) / update(G.T) should equal sqrt(rows/cols).
    """
    torch.manual_seed(0)
    rows, cols = 8, 3
    grad = torch.randn(rows, cols)
    momentum = torch.zeros_like(grad)
    update = _muon_update(grad, momentum, 0.95, 5)

    grad_t = grad.T.contiguous()
    momentum_t = torch.zeros_like(grad_t)
    update_t = _muon_update(grad_t, momentum_t, 0.95, 5)

    norm_ratio = update.norm().item() / update_t.norm().item()
    expected_ratio = math.sqrt(rows / cols)
    assert abs(norm_ratio - expected_ratio) < 0.01, (
        f"Aspect-ratio norm ratio {norm_ratio:.4f} != expected {expected_ratio:.4f}"
    )


# ---------------------------------------------------------------------------
# Shared helpers (Adam moment update + LambW trust ratio)
# ---------------------------------------------------------------------------


def test_adam_moment_update_produces_finite_update() -> None:
    grad = torch.randn(4, 4)
    state: dict = {}
    update = _adam_moment_update(grad, state, (0.9, 0.999), 1e-8)
    assert update.shape == grad.shape
    assert torch.isfinite(update).all()
    assert state["step"] == 1
    assert "exp_avg" in state
    assert "exp_avg_sq" in state


def test_lambw_trust_ratio_returns_one_for_zero_norms() -> None:
    p = torch.zeros(4, 4)
    adam_update = torch.zeros(4, 4)
    ratio = _lambw_trust_ratio(p, adam_update)
    assert torch.allclose(ratio, torch.ones_like(ratio))


def test_lambw_trust_ratio_returns_ratio_for_nonzero_norms() -> None:
    p = torch.ones(4, 4) * 2.0  # ||p|| = 4.0
    adam_update = torch.ones(4, 4) * 1.0  # ||adam_update|| = 2.0
    ratio = _lambw_trust_ratio(p, adam_update)
    # ratio = ||p|| / ||adam_update|| = 4.0 / 2.0 = 2.0
    assert torch.allclose(ratio, torch.tensor(2.0))


# ---------------------------------------------------------------------------
# MuonWithAux optimizer
# ---------------------------------------------------------------------------


def _make_muon_param_groups(
    update_type_aux: str = "lambw",
) -> list[dict]:
    muon_param = torch.randn(4, 4, requires_grad=True)
    aux_decay_param = torch.randn(4, 4, requires_grad=True)
    aux_no_decay_param = torch.randn(4, 4, requires_grad=True)
    return [
        {
            "params": [muon_param],
            "lr": 0.02,
            "weight_decay": 0.0,
            "update_type": "muon",
            "momentum": 0.95,
            "n_schulz_steps": 5,
        },
        {
            "params": [aux_decay_param],
            "lr": 0.001,
            "weight_decay": 0.01,
            "update_type": update_type_aux,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
        {
            "params": [aux_no_decay_param],
            "lr": 0.001,
            "weight_decay": 0.0,
            "update_type": update_type_aux,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        },
    ]


def test_muon_with_aux_step_changes_params() -> None:
    groups = _make_muon_param_groups()
    opt = MuonWithAux(groups)
    for group in groups:
        for p in group["params"]:
            p.grad = torch.randn_like(p) * 0.1
    pre = [p.detach().clone() for group in groups for p in group["params"]]
    opt.step()
    post = [p.detach().clone() for group in groups for p in group["params"]]
    for i, (p_pre, p_post) in enumerate(zip(pre, post)):
        assert not torch.allclose(p_pre, p_post), f"Param {i} did not change"


def test_muon_with_aux_no_grad_is_noop() -> None:
    groups = _make_muon_param_groups()
    opt = MuonWithAux(groups)
    pre = [p.detach().clone() for group in groups for p in group["params"]]
    opt.step()  # No grads assigned
    post = [p.detach().clone() for group in groups for p in group["params"]]
    for i, (p_pre, p_post) in enumerate(zip(pre, post)):
        assert torch.allclose(p_pre, p_post), f"Param {i} changed without grad"


def test_muon_with_aux_closure() -> None:
    groups = _make_muon_param_groups()
    opt = MuonWithAux(groups)
    muon_param = groups[0]["params"][0]
    pre = muon_param.detach().clone()

    def closure():
        opt.zero_grad()
        loss = (muon_param ** 2).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert loss.requires_grad
    assert not torch.allclose(pre, muon_param.detach())


def test_muon_with_aux_rejects_invalid_update_type() -> None:
    groups = _make_muon_param_groups()
    groups[0]["update_type"] = "sgd"
    with pytest.raises(ValueError, match="Invalid update_type"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_invalid_lr() -> None:
    groups = _make_muon_param_groups()
    groups[0]["lr"] = -1.0
    with pytest.raises(ValueError, match="lr must be positive and finite"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_invalid_weight_decay() -> None:
    groups = _make_muon_param_groups()
    groups[0]["weight_decay"] = -0.1
    with pytest.raises(ValueError, match="weight_decay must be non-negative and finite"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_invalid_momentum() -> None:
    groups = _make_muon_param_groups()
    groups[0]["momentum"] = 1.0
    with pytest.raises(ValueError, match="momentum must be in"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_invalid_n_schulz_steps() -> None:
    groups = _make_muon_param_groups()
    groups[0]["n_schulz_steps"] = 0
    with pytest.raises(ValueError, match="n_schulz_steps must be a positive integer"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_invalid_betas() -> None:
    groups = _make_muon_param_groups()
    groups[1]["betas"] = (1.0, 0.999)
    with pytest.raises(ValueError, match="beta1 must be in"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_invalid_eps() -> None:
    groups = _make_muon_param_groups()
    groups[1]["eps"] = 0.0
    with pytest.raises(ValueError, match="eps must be positive and finite"):
        MuonWithAux(groups)


def test_muon_with_aux_rejects_low_rank_muon_param() -> None:
    groups = _make_muon_param_groups()
    groups[0]["params"] = [nn.Parameter(torch.randn(4))]
    with pytest.raises(ValueError, match="ndim >= 2"):
        MuonWithAux(groups)


def test_muon_with_aux_state_dict_round_trip() -> None:
    groups = _make_muon_param_groups()
    opt = MuonWithAux(groups)
    for group in groups:
        for p in group["params"]:
            p.grad = torch.randn_like(p) * 0.1
    opt.step()
    state = opt.state_dict()

    # Create a new optimizer with the same param groups
    groups2 = _make_muon_param_groups()
    opt2 = MuonWithAux(groups2)
    opt2.load_state_dict(state)

    # Verify state is restored
    for group1, group2 in zip(opt.param_groups, opt2.param_groups):
        for p1, p2 in zip(group1["params"], group2["params"]):
            state1 = opt.state[p1]
            state2 = opt2.state[p2]
            if "momentum_buffer" in state1:
                assert torch.allclose(state1["momentum_buffer"], state2["momentum_buffer"])
            if "exp_avg" in state1:
                assert torch.allclose(state1["exp_avg"], state2["exp_avg"])
                assert torch.allclose(state1["exp_avg_sq"], state2["exp_avg_sq"])


# ---------------------------------------------------------------------------
# Auxiliary branch parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wd", [0.0, 0.01])
def test_lambw_auxiliary_branch_matches_standalone_lambw(wd: float) -> None:
    """MuonWithAux 'lambw' branch must match standalone LambW exactly."""
    torch.manual_seed(42)
    p_ours = nn.Linear(4, 4, bias=False)
    p_ref = nn.Linear(4, 4, bias=False)
    p_ref.weight.data = p_ours.weight.data.clone()

    grad = torch.randn(4, 4)
    p_ours.weight.grad = grad.clone()
    p_ref.weight.grad = grad.clone()

    betas = (0.9, 0.999)
    eps = 1e-8
    lr = 0.01

    opt_ours = MuonWithAux([{
        "params": [p_ours.weight], "lr": lr, "weight_decay": wd,
        "update_type": "lambw", "betas": betas, "eps": eps,
    }])
    opt_ref = LambW([p_ref.weight], lr=lr, betas=betas, eps=eps, weight_decay=wd)

    for _ in range(5):
        p_ours.weight.grad = torch.randn(4, 4)
        p_ref.weight.grad = p_ours.weight.grad.clone()
        opt_ours.step()
        opt_ref.step()

    diff = (p_ours.weight - p_ref.weight).abs().max().item()
    assert diff < 1e-6, f"LambW parity (wd={wd}) failed: max diff = {diff}"


# ---------------------------------------------------------------------------
# Parameter routing
# ---------------------------------------------------------------------------


def _make_muon_model() -> nn.Module:
    """Build a small DAT model with ``layers.*`` naming for Muon tests."""
    from models.attention.dat.config import DatLMConfig
    from models.attention.dat.lm import DatDecoderLM

    cfg = DatLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36,
        n_heads_sa=2, n_heads_ra=1, n_layers=1, n_symbols=33,
        symbol_retrieval="relsymbolic", tie_lm_head=False,
    )
    return DatDecoderLM(cfg)


def test_is_muon_hidden_weight_classification() -> None:
    # layers.* 2D weight -> Muon
    p = nn.Parameter(torch.randn(4, 4))
    assert _is_muon_hidden_weight("layers.0.sensory_attention.q_proj.weight", p)

    # layers.* 3D weight (wr_proj) -> Muon
    p3d = nn.Parameter(torch.randn(2, 32, 3))
    assert _is_muon_hidden_weight("layers.0.relational_attention.wr_proj", p3d)

    # symbol_retriever.* projection weight -> Muon
    assert _is_muon_hidden_weight("symbol_retriever.q_proj.weight", p)
    assert _is_muon_hidden_weight("symbol_retriever.k_proj.weight", p)
    assert _is_muon_hidden_weight("symbol_retriever.rel_to_hidden.weight", p)
    assert _is_muon_hidden_weight(
        "symbol_retriever.symbolic_attention.q_proj.weight", p
    )

    # symbol_retriever.* symbol tables -> NOT Muon
    assert not _is_muon_hidden_weight(
        "symbol_retriever.symbolic_attention.template_features", p
    )
    assert not _is_muon_hidden_weight(
        "symbol_retriever.symbolic_attention.symbol_library", p
    )

    # token_embeddings -> NOT Muon
    assert not _is_muon_hidden_weight("token_embeddings.weight", p)

    # lm_head -> NOT Muon
    assert not _is_muon_hidden_weight("lm_head.weight", p)

    # final_norm (1D) -> NOT Muon
    p1d = nn.Parameter(torch.randn(192))
    assert not _is_muon_hidden_weight("final_norm.weight", p1d)

    # bias (1D) -> NOT Muon
    assert not _is_muon_hidden_weight("layers.0.sensory_attention.o_proj.bias", p1d)

    # NextLat dynamics MLP weight -> Muon
    assert _is_muon_hidden_weight("mlp.0.weight", p)
    assert _is_muon_hidden_weight("mlp.2.weight", p)

    # NextLat dynamics MLP bias -> NOT Muon
    assert not _is_muon_hidden_weight("mlp.0.bias", p1d)

    # Norm weight -> NOT Muon
    assert not _is_muon_hidden_weight("norm.weight", p1d)


def test_classify_muon_parameters_covers_all_trainable_params() -> None:
    """Every trainable parameter must appear in exactly one group, no duplicates."""
    from models.attention.dat.config import DatLMConfig
    from models.attention.dat.lm import DatDecoderLM
    from models.attention.transformer.components import RMSNorm
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

    cfg = DatLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36,
        n_heads_sa=2, n_heads_ra=1, n_layers=2, n_symbols=33,
        symbol_retrieval="relsymbolic", tie_lm_head=False,
    )
    model = DatDecoderLM(cfg)

    forbidden = tuple(ALL_LAYERNORM_LAYERS) + (RMSNorm,)
    muon_params, aux_decay, aux_no_decay = _classify_muon_parameters(
        (model,), forbidden
    )

    # Every trainable param appears exactly once
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    all_classified = muon_params + aux_decay + aux_no_decay
    assert len(all_classified) == len(all_trainable), (
        f"Classified {len(all_classified)} but model has {len(all_trainable)} trainable params"
    )

    # No duplicates
    seen_ids = set()
    for p in all_classified:
        pid = id(p)
        assert pid not in seen_ids, "Duplicate parameter in classification"
        seen_ids.add(pid)

    # Muon group is nonempty
    assert len(muon_params) > 0, "Muon group is empty"

    # Verify routing: Muon params should be hidden weights, aux params should not
    muon_param_ids = {id(p) for p in muon_params}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_muon = id(p) in muon_param_ids
        is_hidden = _is_muon_hidden_weight(name, p)
        assert is_muon == is_hidden, (
            f"Param {name!r} routing mismatch: is_muon={is_muon}, is_hidden={is_hidden}"
        )


def test_build_optimizer_dispatches_muon() -> None:
    config = TrainConfig(optimizer="muon", muon_aux_optimizer="lambw")
    modules = (_make_muon_model(),)
    opt = build_optimizer(config, modules)
    assert isinstance(opt, MuonWithAux)


def test_build_optimizer_muon_aux_lambw() -> None:
    config = TrainConfig(optimizer="muon", muon_aux_optimizer="lambw")
    modules = (_make_muon_model(),)
    opt = build_optimizer(config, modules)
    assert isinstance(opt, MuonWithAux)
    # Check aux groups use lambw
    for group in opt.param_groups:
        if group["update_type"] != "muon":
            assert group["update_type"] == "lambw"


def test_build_optimizer_muon_raises_on_empty_muon_group() -> None:
    """If there are no hidden weight parameters, Muon should raise."""
    config = TrainConfig(optimizer="muon", muon_aux_optimizer="lambw")
    # A module with only 1D parameters (no hidden weights)
    modules = (nn.LayerNorm(4),)
    with pytest.raises(ValueError, match="no hidden weight parameters"):
        build_optimizer(config, modules)


# ---------------------------------------------------------------------------
# Routing coverage across model variants and NextLat modules
# ---------------------------------------------------------------------------


def _assert_routing_covers_all(module: nn.Module) -> None:
    """Assert every trainable param is classified exactly once and routing
    matches ``_is_muon_hidden_weight``."""
    from models.attention.transformer.components import RMSNorm
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

    forbidden = tuple(ALL_LAYERNORM_LAYERS) + (RMSNorm,)
    muon_params, aux_decay, aux_no_decay = _classify_muon_parameters(
        (module,), forbidden
    )

    all_trainable = [p for p in module.parameters() if p.requires_grad]
    all_classified = muon_params + aux_decay + aux_no_decay
    assert len(all_classified) == len(all_trainable), (
        f"Classified {len(all_classified)} but module has {len(all_trainable)} trainable params"
    )

    seen_ids: set[int] = set()
    for p in all_classified:
        pid = id(p)
        assert pid not in seen_ids, "Duplicate parameter in classification"
        seen_ids.add(pid)

    muon_param_ids = {id(p) for p in muon_params}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        is_muon = id(p) in muon_param_ids
        is_hidden = _is_muon_hidden_weight(name, p)
        assert is_muon == is_hidden, (
            f"Param {name!r} routing mismatch: is_muon={is_muon}, is_hidden={is_hidden}"
        )


def test_routing_self_attention_model() -> None:
    from models.attention.transformer.config import SelfAttentionLMConfig
    from models.attention.transformer.lm import SelfAttentionDecoderLM

    cfg = SelfAttentionLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36, n_heads=2, n_layers=2,
    )
    model = SelfAttentionDecoderLM(cfg)
    _assert_routing_covers_all(model)


def test_routing_dat_symbolic_tied() -> None:
    from models.attention.dat.config import DatLMConfig
    from models.attention.dat.lm import DatDecoderLM

    cfg = DatLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36,
        n_heads_sa=2, n_heads_ra=1, n_layers=2, n_symbols=33,
        symbol_retrieval="symbolic", tie_lm_head=True,
    )
    model = DatDecoderLM(cfg)
    _assert_routing_covers_all(model)


def test_routing_dat_positional_untied() -> None:
    from models.attention.dat.config import DatLMConfig
    from models.attention.dat.lm import DatDecoderLM

    cfg = DatLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36,
        n_heads_sa=2, n_heads_ra=1, n_layers=2, n_symbols=33,
        symbol_retrieval="positional", tie_lm_head=False,
    )
    model = DatDecoderLM(cfg)
    _assert_routing_covers_all(model)


def test_routing_dat_relative_tied() -> None:
    from models.attention.dat.config import DatLMConfig
    from models.attention.dat.lm import DatDecoderLM

    cfg = DatLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36,
        n_heads_sa=2, n_heads_ra=1, n_layers=2, n_symbols=33,
        symbol_retrieval="relative", tie_lm_head=True,
    )
    model = DatDecoderLM(cfg)
    _assert_routing_covers_all(model)


def test_routing_nextlat_dynamics_model() -> None:
    from train.nextlat import NextLatDynamicsModel

    model = NextLatDynamicsModel(hidden_dim=36, dropout=0.0, proj_factor=1.0, use_bias=True)
    _assert_routing_covers_all(model)


def test_routing_combined_model_and_nextlat_modules() -> None:
    """The runner passes (model, nextlat_dynamics) to build_optimizer.
    Verify routing across both modules together."""
    from models.attention.dat.config import DatLMConfig
    from models.attention.dat.lm import DatDecoderLM
    from models.attention.transformer.components import RMSNorm
    from train.nextlat import NextLatDynamicsModel
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

    cfg = DatLMConfig(
        vocab_size=100, max_seq_len=33, hidden_dim=36,
        n_heads_sa=2, n_heads_ra=1, n_layers=2, n_symbols=33,
        symbol_retrieval="relsymbolic", tie_lm_head=False,
    )
    model = DatDecoderLM(cfg)
    nextlat_clm = NextLatDynamicsModel(hidden_dim=36, dropout=0.0, proj_factor=1.0, use_bias=True)
    modules = (model, nextlat_clm)

    forbidden = tuple(ALL_LAYERNORM_LAYERS) + (RMSNorm,)
    muon_params, aux_decay, aux_no_decay = _classify_muon_parameters(modules, forbidden)

    # Every trainable param across both modules appears exactly once
    all_trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
    all_classified = muon_params + aux_decay + aux_no_decay
    assert len(all_classified) == len(all_trainable), (
        f"Classified {len(all_classified)} but modules have {len(all_trainable)} trainable params"
    )

    # No duplicates
    seen_ids: set[int] = set()
    for p in all_classified:
        pid = id(p)
        assert pid not in seen_ids, "Duplicate parameter in classification"
        seen_ids.add(pid)

    # NextLat MLP weights should be in the Muon group
    muon_param_ids = {id(p) for p in muon_params}
    for name, p in nextlat_clm.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("mlp.") and name.endswith(".weight"):
            assert id(p) in muon_param_ids, f"NextLat MLP weight {name!r} not in Muon group"


# ---------------------------------------------------------------------------
# Scheduler LR ratio preservation
# ---------------------------------------------------------------------------


def test_scheduler_preserves_muon_to_aux_lr_ratio() -> None:
    """The LR scheduler must scale Muon and aux groups proportionally,
    preserving the initial muon_lr:learning_rate ratio across at least
    one step and after state restoration."""
    from train.optim import build_scheduler

    config = TrainConfig(
        optimizer="muon",
        muon_aux_optimizer="lambw",
        muon_lr=0.02,
        learning_rate=0.001,
        warmup_ratio=0.0,
        n_epochs=1,
    )
    modules = (_make_muon_model(),)
    opt = build_optimizer(config, modules)
    scheduler = build_scheduler(config, opt, total_training_steps=10)

    # Initial LR ratio
    initial_lrs = [group["lr"] for group in opt.param_groups]
    initial_ratio = initial_lrs[0] / initial_lrs[1]  # muon_lr / aux_lr

    # Step optimizer then scheduler to avoid step-order warning
    for p in modules[0].parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    opt.step()
    scheduler.step()
    stepped_lrs = [group["lr"] for group in opt.param_groups]
    stepped_ratio = stepped_lrs[0] / stepped_lrs[1]

    assert abs(initial_ratio - stepped_ratio) < 1e-5, (
        f"LR ratio changed: initial={initial_ratio}, stepped={stepped_ratio}"
    )

    # State restoration
    state = scheduler.state_dict()
    scheduler2 = build_scheduler(config, opt, total_training_steps=10)
    scheduler2.load_state_dict(state)
    restored_lrs = [group["lr"] for group in opt.param_groups]
    restored_ratio = restored_lrs[0] / restored_lrs[1]

    assert abs(initial_ratio - restored_ratio) < 1e-5, (
        f"LR ratio changed after restore: initial={initial_ratio}, restored={restored_ratio}"
    )


def test_optimizer_state_restoration_preserves_continued_behavior(tmp_path) -> None:
    """Restore optimizer state into fresh parameters. Apply identical
    gradients and assert parameters remain identical.

    Uses torch.save/torch.load to match production checkpoint behavior
    and avoid tensor aliasing between live optimizer states."""
    config = TrainConfig(
        optimizer="muon",
        muon_aux_optimizer="lambw",
        muon_lr=0.02,
        learning_rate=0.001,
        n_epochs=1,
    )

    # Original optimizer
    model_a = _make_muon_model()
    opt_a = build_optimizer(config, (model_a,))

    # Apply a few steps with deterministic gradients
    torch.manual_seed(42)
    for _ in range(3):
        opt_a.zero_grad()
        for p in model_a.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1
        opt_a.step()

    # Save optimizer state via serialization (production-faithful)
    state_path = tmp_path / "opt_state.pt"
    torch.save(opt_a.state_dict(), state_path)
    opt_state = torch.load(state_path, weights_only=True)

    # Build fresh params with same weights
    model_b = _make_muon_model()
    with torch.no_grad():
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            p_b.copy_(p_a)
    opt_b = build_optimizer(config, (model_b,))
    opt_b.load_state_dict(opt_state)

    # Apply identical next gradients
    torch.manual_seed(99)
    for opt_orig, opt_rest, mod_orig, mod_rest in [
        (opt_a, opt_b, model_a, model_b)
    ]:
        torch.manual_seed(99)
        opt_orig.zero_grad()
        for p in mod_orig.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1
        opt_orig.step()

        torch.manual_seed(99)
        opt_rest.zero_grad()
        for p in mod_rest.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1
        opt_rest.step()

    # Assert parameters remain identical
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6), (
            f"Parameter mismatch after restoration: max diff = {(p_a - p_b).abs().max().item()}"
        )


def test_scheduler_state_restoration_preserves_continued_behavior() -> None:
    """Restore scheduler state into a fresh scheduler. Step both and
    assert LRs remain identical."""
    from train.optim import build_scheduler

    config = TrainConfig(
        optimizer="muon",
        muon_aux_optimizer="lambw",
        muon_lr=0.02,
        learning_rate=0.001,
        warmup_ratio=0.0,
        n_epochs=1,
    )

    # Original optimizer + scheduler
    model_a = _make_muon_model()
    opt_a = build_optimizer(config, (model_a,))
    sched_a = build_scheduler(config, opt_a, total_training_steps=10)

    # Apply a few steps
    torch.manual_seed(42)
    for _ in range(3):
        opt_a.zero_grad()
        for p in model_a.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1
        opt_a.step()
        sched_a.step()

    # Save scheduler state
    sched_state = sched_a.state_dict()

    # Build fresh optimizer + scheduler
    model_b = _make_muon_model()
    with torch.no_grad():
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            p_b.copy_(p_a)
    opt_b = build_optimizer(config, (model_b,))
    sched_b = build_scheduler(config, opt_b, total_training_steps=10)
    sched_b.load_state_dict(sched_state)

    # Step both with identical gradients
    torch.manual_seed(77)
    for _ in range(2):
        torch.manual_seed(77)
        opt_a.zero_grad()
        for p in model_a.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1
        opt_a.step()
        sched_a.step()

        torch.manual_seed(77)
        opt_b.zero_grad()
        for p in model_b.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.1
        opt_b.step()
        sched_b.step()

    # Assert LRs remain identical
    lrs_a = [group["lr"] for group in opt_a.param_groups]
    lrs_b = [group["lr"] for group in opt_b.param_groups]
    for i, (lr_a, lr_b) in enumerate(zip(lrs_a, lrs_b)):
        assert abs(lr_a - lr_b) < 1e-9, (
            f"LR mismatch in group {i}: original={lr_a}, restored={lr_b}"
        )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_accepts_muon_optimizer() -> None:
    config = TrainConfig(optimizer="muon", muon_aux_optimizer="lambw")
    assert config.optimizer == "muon"
    assert config.muon_aux_optimizer == "lambw"


def test_config_accepts_muon_aux_lambw() -> None:
    config = TrainConfig(optimizer="muon", muon_aux_optimizer="lambw")
    assert config.muon_aux_optimizer == "lambw"


def test_config_rejects_invalid_muon_lr() -> None:
    with pytest.raises(ValueError, match="muon_lr must be a positive finite value"):
        TrainConfig(optimizer="muon", muon_lr=0.0)
    with pytest.raises(ValueError, match="muon_lr must be a positive finite value"):
        TrainConfig(optimizer="muon", muon_lr=float("nan"))
    with pytest.raises(ValueError, match="muon_lr must be a positive finite value"):
        TrainConfig(optimizer="muon", muon_lr=float("inf"))


def test_config_rejects_invalid_muon_momentum() -> None:
    with pytest.raises(ValueError, match="muon_momentum must be in"):
        TrainConfig(optimizer="muon", muon_momentum=1.0)


def test_config_rejects_invalid_muon_n_schulz_steps() -> None:
    with pytest.raises(ValueError, match="muon_n_schulz_steps must be a positive integer"):
        TrainConfig(optimizer="muon", muon_n_schulz_steps=0)
    with pytest.raises(ValueError, match="muon_n_schulz_steps must be a positive integer"):
        TrainConfig(optimizer="muon", muon_n_schulz_steps=1.5)
    with pytest.raises(ValueError, match="muon_n_schulz_steps must be a positive integer"):
        TrainConfig(optimizer="muon", muon_n_schulz_steps=True)


def test_config_muon_defaults() -> None:
    config = TrainConfig()
    assert config.muon_lr == pytest.approx(0.02)
    assert config.muon_momentum == pytest.approx(0.95)
    assert config.muon_n_schulz_steps == 5
    assert config.muon_aux_optimizer == "lambw"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_cli_parser_accepts_muon_optimizer() -> None:
    config = parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            
        "--optimizer", "muon",
        "--muon_aux_optimizer", "lambw",
        "--muon_lr", "0.02",
        "--muon_momentum", "0.95",
        "--muon_n_schulz_steps", "5",
    ])
    assert config.optimizer == "muon"
    assert config.muon_aux_optimizer == "lambw"
    assert config.muon_lr == pytest.approx(0.02)
    assert config.muon_momentum == pytest.approx(0.95)
    assert config.muon_n_schulz_steps == 5


def test_cli_parser_accepts_muon_aux_lambw() -> None:
    config = parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            
        "--optimizer", "muon",
        "--muon_aux_optimizer", "lambw",
    ])
    assert config.muon_aux_optimizer == "lambw"


def test_cli_parser_rejects_invalid_muon_lr() -> None:
    with pytest.raises(ValueError, match="muon_lr must be a positive finite value"):
        parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--optimizer", "muon", "--muon_lr", "0.0"])


def test_cli_parser_defaults_muon_fields() -> None:
    config = parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            ])
    assert config.muon_lr == pytest.approx(0.02)
    assert config.muon_momentum == pytest.approx(0.95)
    assert config.muon_n_schulz_steps == 5
    assert config.muon_aux_optimizer == "lambw"
