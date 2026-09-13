"""Pytest coverage for the LambW optimizer and optimizer dispatch."""

import pytest
import torch
import torch.nn as nn

from train.config import TrainConfig
from train.optim import LambW, build_optimizer


def make_one_param_modules(weight: float) -> tuple[nn.Module, ...]:
    module = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(weight)
    return (module,)


def test_lambw_optimizer_runs_one_step() -> None:
    module = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    optimizer = LambW(module.parameters(), lr=0.01, weight_decay=0.0)
    loss = module(torch.tensor([[1.0, 1.0]])).sum()
    loss.backward()
    pre_step = module.weight.detach().clone()
    optimizer.step()
    post_step = module.weight.detach().clone()
    assert not torch.allclose(pre_step, post_step), "LambW step did not change parameters"


def test_lambw_closure_runs_with_grad_enabled() -> None:
    module = nn.Linear(2, 2, bias=False)
    optimizer = LambW(module.parameters(), lr=0.01, weight_decay=0.0)
    inputs = torch.ones(1, 2)
    pre_step = module.weight.detach().clone()

    def closure():
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = module(inputs).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert loss is not None
    assert loss.requires_grad
    assert not torch.allclose(pre_step, module.weight.detach())


def test_lambw_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="Invalid learning rate"):
        LambW([torch.zeros(1, requires_grad=True)], lr=-0.1)
    with pytest.raises(ValueError, match="Invalid epsilon"):
        LambW([torch.zeros(1, requires_grad=True)], eps=-1e-8)
    with pytest.raises(ValueError, match="Invalid beta parameter at index 0"):
        LambW([torch.zeros(1, requires_grad=True)], betas=(-0.1, 0.999))
    with pytest.raises(ValueError, match="Invalid beta parameter at index 1"):
        LambW([torch.zeros(1, requires_grad=True)], betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="Invalid weight_decay"):
        LambW([torch.zeros(1, requires_grad=True)], weight_decay=-0.1)


def test_lambw_step_with_no_grad_is_noop() -> None:
    module = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    optimizer = LambW(module.parameters(), lr=0.01, weight_decay=0.0)
    pre = module.weight.detach().clone()
    optimizer.step()
    assert torch.allclose(pre, module.weight.detach())


def test_lambw_relative_update_equals_learning_rate() -> None:
    """The strongest LAMBW invariant: with zero weight decay and nonzero
    gradients, ||parameter_after - parameter_before|| / ||parameter_before||
    == learning_rate. This holds because the trust ratio normalizes the
    update to the parameter's own norm.

    Initialize weights to 2.0 (w_norm = 4.0 for a 2x2 of 2s). With grad
    0.1 everywhere, step 1 gives adam_update ~= 1.0 everywhere (g_norm
    ~= 2.0), so ratio ~= 2.0. Plain Adam would give ratio 1.0 and a
    relative update of lr * g_norm / w_norm = 0.1 * 2 / 4 = 0.05, not lr.
    """
    module = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(2.0)
    module.weight.grad = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    pre = module.weight.detach().clone()
    optimizer = LambW(
        module.parameters(),
        lr=0.1,
        weight_decay=0.0,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
    optimizer.step()
    delta_norm = torch.norm((module.weight - pre).flatten()).item()
    pre_norm = torch.norm(pre.flatten()).item()
    relative_update = delta_norm / pre_norm
    assert abs(relative_update - 0.1) < 1e-5, (
        f"relative_update={relative_update}, expected lr=0.1"
    )


def test_lambw_decoupled_weight_decay_independent_of_trust_ratio() -> None:
    """With decoupled weight decay and zero gradients, the shrinkage is
    exactly (1 - lr * wd) regardless of the trust ratio."""
    module_a = nn.Linear(1, 1, bias=False)
    module_b = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module_a.weight.fill_(2.0)
        module_b.weight.fill_(3.0)
    module_a.weight.grad = torch.zeros_like(module_a.weight)
    module_b.weight.grad = torch.zeros_like(module_b.weight)
    optimizer = LambW(
        list(module_a.parameters()) + list(module_b.parameters()),
        lr=0.1,
        weight_decay=0.5,
        eps=1e-8,
    )
    optimizer.step()
    # With zero gradient, adam_update = 0, g_norm = 0, ratio = 1.0, so
    # the update term is -lr * 1.0 * 0 = 0. Only the decay term applies.
    # Both should be multiplied by (1 - lr * wd) = (1 - 0.1 * 0.5) = 0.95.
    assert abs(module_a.weight.item() - 2.0 * 0.95) < 1e-6
    assert abs(module_b.weight.item() - 3.0 * 0.95) < 1e-6


def test_lambw_state_dict_round_trip() -> None:
    module = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)
    opt = LambW(module.parameters(), lr=0.01, weight_decay=0.01)
    loss = module(torch.randn(1, 4)).sum()
    loss.backward()
    opt.step()
    state = opt.state_dict()

    module2 = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        module2.weight.fill_(1.0)
    opt2 = LambW(module2.parameters(), lr=0.01, weight_decay=0.01)
    opt2.load_state_dict(state)
    for group, group2 in zip(opt.param_groups, opt2.param_groups):
        for p, p2 in zip(group["params"], group2["params"]):
            state1 = opt.state[p]
            state2 = opt2.state[p2]
            assert torch.allclose(state1["exp_avg"], state2["exp_avg"])
            assert torch.allclose(state1["exp_avg_sq"], state2["exp_avg_sq"])


def test_build_optimizer_dispatches_adamw() -> None:
    config = TrainConfig(optimizer="adamw")
    modules = make_one_param_modules(1.0)
    opt = build_optimizer(config, modules)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_uses_shared_config_fields() -> None:
    """AdamW should use optimizer_beta1, optimizer_beta2, and optimizer_eps
    from the config."""
    config_adamw = TrainConfig(
        optimizer="adamw",
        learning_rate=0.005,
        optimizer_beta1=0.8,
        optimizer_beta2=0.95,
        optimizer_eps=1e-7,
    )
    modules = make_one_param_modules(1.0)
    opt_adamw = build_optimizer(config_adamw, modules)
    assert isinstance(opt_adamw, torch.optim.AdamW)
    group_aw = opt_adamw.param_groups[0]
    assert group_aw["lr"] == pytest.approx(0.005)
    assert group_aw["betas"] == (0.8, 0.95)
    assert group_aw["eps"] == pytest.approx(1e-7)


def test_config_rejects_invalid_optimizer() -> None:
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        TrainConfig(optimizer="sgd")
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        TrainConfig(optimizer="lamb")


def test_config_defaults_to_adamw() -> None:
    config = TrainConfig()
    assert config.optimizer == "adamw"
    assert config.optimizer_beta1 == pytest.approx(0.9)
    assert config.optimizer_beta2 == pytest.approx(0.999)
    assert config.optimizer_eps == pytest.approx(1e-8)


def test_config_rejects_invalid_beta1() -> None:
    with pytest.raises(ValueError, match="optimizer_beta1 must be in"):
        TrainConfig(optimizer_beta1=0.0)
    with pytest.raises(ValueError, match="optimizer_beta1 must be in"):
        TrainConfig(optimizer_beta1=1.0)


def test_config_rejects_invalid_beta2() -> None:
    with pytest.raises(ValueError, match="optimizer_beta2 must be in"):
        TrainConfig(optimizer_beta2=0.0)
    with pytest.raises(ValueError, match="optimizer_beta2 must be in"):
        TrainConfig(optimizer_beta2=1.0)


def test_config_rejects_invalid_eps() -> None:
    with pytest.raises(ValueError, match="optimizer_eps must be a positive finite value"):
        TrainConfig(optimizer_eps=0.0)
    with pytest.raises(ValueError, match="optimizer_eps must be a positive finite value"):
        TrainConfig(optimizer_eps=-1e-8)
    with pytest.raises(ValueError, match="optimizer_eps must be a positive finite value"):
        TrainConfig(optimizer_eps=float("nan"))
    with pytest.raises(ValueError, match="optimizer_eps must be a positive finite value"):
        TrainConfig(optimizer_eps=float("inf"))
    with pytest.raises(ValueError, match="optimizer_eps must be a positive finite value"):
        TrainConfig(optimizer_eps=float("-inf"))


def test_cli_parser_accepts_muon_and_shared_fields() -> None:
    """The CLI parser should accept --optimizer muon and the shared
    optimizer_beta1, optimizer_beta2, optimizer_eps flags."""
    from train.config import parse_train_config

    argv = [
        "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
        "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
        "--optimizer", "muon",
        "--optimizer_beta1", "0.9",
        "--optimizer_beta2", "0.98",
        "--optimizer_eps", "1e-8",
    ]
    config = parse_train_config(argv)
    assert config.optimizer == "muon"
    assert config.optimizer_beta1 == pytest.approx(0.9)
    assert config.optimizer_beta2 == pytest.approx(0.98)
    assert config.optimizer_eps == pytest.approx(1e-8)


def test_cli_parser_rejects_invalid_optimizer() -> None:
    from train.config import parse_train_config

    with pytest.raises(SystemExit):
        parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--optimizer", "sgd"])


def test_cli_parser_defaults_optimizer_fields() -> None:
    """The CLI parser should default optimizer to adamw, beta1 to 0.9,
    beta2 to 0.999, and eps to 1e-8."""
    from train.config import parse_train_config

    config = parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            ])
    assert config.optimizer == "adamw"
    assert config.optimizer_beta1 == pytest.approx(0.9)
    assert config.optimizer_beta2 == pytest.approx(0.999)
    assert config.optimizer_eps == pytest.approx(1e-8)


def test_cli_parser_rejects_invalid_eps() -> None:
    """The CLI parser should reject non-finite or non-positive optimizer_eps."""
    from train.config import parse_train_config

    # -0.0001 instead of -1e-8 because argparse treats -1e-8 as an
    # unknown option flag (it starts with - and is not a plain number).
    for bad_eps in ["0.0", "-0.0001", "inf", "nan"]:
        with pytest.raises(ValueError, match="optimizer_eps must be a positive finite value"):
            parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--optimizer_eps", bad_eps])
