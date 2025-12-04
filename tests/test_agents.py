import torch
import torch.nn as nn
import torch.distributions as D

from safemeta_rl.agents.base import BasePPOAgent
from safemeta_rl.agents.ppo_agent import PPOAgent
from safemeta_rl.agents.ppo_lag_agent import PPOLagAgent
from safemeta_rl.agents.epo_agent import EPOAgent
from safemeta_rl.agents.lstc_agent import LSTCAgent


class DummyPolicy(nn.Module):
    """Minimal policy emitting a fixed Gaussian to make tests deterministic."""

    def __init__(self, action_dim: int = 2):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=False)

    def forward(self, obs: torch.Tensor) -> D.Independent:
        dist = D.Normal(self.mean.expand(obs.shape[0], -1), self.log_std.exp().expand(obs.shape[0], -1))
        return D.Independent(dist, 1)


class DummyValue(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 1)


def base_config():
    return {
        "constraints": {
            "lambda_long_init": 0.1,
            "lambda_short_init": 0.5,
            "lambda_long_lr": 0.025,
            "lambda_short_lr": 0.01,
            "cost_limit": 1.0,
            "short_term_horizon": 5,
        }
    }


def test_penalty_terms_match_algorithms():
    obs = torch.zeros(3, 4)
    actions = torch.zeros(3, 2)
    advantages = torch.ones(3, 1)
    cost_advantages = torch.full((3, 1), 2.0)
    B = torch.full((3, 1), 0.3)
    ratio = torch.ones(3, 1)

    policy = DummyPolicy()
    value = DummyValue()
    cost_value = DummyValue()
    cfg = base_config()

    ppo = PPOAgent(policy, value, cost_value, cfg)
    ppo_penalty = ppo.penalty_term(advantages, cost_advantages, B, ratio)
    assert torch.allclose(ppo_penalty, advantages)

    lag = PPOLagAgent(policy, value, cost_value, cfg)
    lag_penalty = lag.penalty_term(advantages, cost_advantages, B, ratio)
    expected_lag = advantages - lag.lambda_long * cost_advantages
    assert torch.allclose(lag_penalty, expected_lag)

    epo = EPOAgent(policy, value, cost_value, cfg, penalty=0.5)
    epo_penalty = epo.penalty_term(advantages, cost_advantages, B, ratio)
    assert torch.allclose(epo_penalty, advantages - 0.5 * cost_advantages)

    lstc = LSTCAgent(policy, value, cost_value, None, cfg, use_validation=False, update_long=True, update_short=False, fixed_lambda_long=None)
    lstc.lambda_short.zero_()  # disable short constraint for this check
    lstc_penalty = lstc.penalty_term(advantages, cost_advantages, B, ratio)
    expected_lstc = advantages - lstc.lambda_long * cost_advantages
    assert torch.allclose(lstc_penalty, expected_lstc)


def test_validation_losses_hinge_behavior():
    preds = torch.tensor([0.5, -0.2, -0.5, 1.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    # feasible labels=1 penalize positive preds, infeasible labels=0 penalize negative preds
    loss = LSTCAgent.validation_loss(preds, labels)
    # feasible part: mean(relu(0.5), relu(-0.2=0)) = 0.25; infeasible: mean(relu(0.5), relu(0)) = 0.25; total 0.5
    assert torch.isclose(loss, torch.tensor(0.5), atol=1e-5)

    violation = LSTCAgent.short_term_violation(preds, labels)
    # only feasible labels considered -> mean relu over feasible preds = (0.5 + 0.0) / 2
    assert abs(violation - 0.25) < 1e-5


def test_compute_advantages_shapes_and_terminal_handling():
    rewards = torch.tensor([[1.0], [1.0], [0.0]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last_value = torch.tensor([[0.0]])
    adv, ret = BasePPOAgent.compute_advantages(rewards, values, dones, last_value, gamma=1.0, gae_lambda=1.0)
    # With done at step 1, gae should reset for final step
    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    # First two steps should accumulate, third only its own reward (0)
    assert torch.allclose(ret[0], torch.tensor([2.0]))
    assert torch.allclose(ret[1], torch.tensor([1.0]))
    assert torch.allclose(ret[2], torch.tensor([0.0]))
