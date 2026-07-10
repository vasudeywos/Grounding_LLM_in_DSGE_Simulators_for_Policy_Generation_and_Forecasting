import torch

from dsge_rl.rollout import TurnRollout
from dsge_rl.trainers.grpo import GRPOTrainer


def test_grpo_uses_stored_generation_time_log_probabilities():
    trainer = object.__new__(GRPOTrainer)
    trainer.model = type("Policy", (), {"device": torch.device("cpu")})()
    first = TurnRollout(
        input_ids=torch.tensor([1, 2, 3]),
        action_mask=torch.tensor([False, True]),
        reward=1.0,
        text="first",
        behavior_log_probs=torch.tensor([-0.4]),
    )
    second = TurnRollout(
        input_ids=torch.tensor([1, 4, 5]),
        action_mask=torch.tensor([False, True]),
        reward=2.0,
        text="second",
        behavior_log_probs=torch.tensor([-0.2]),
    )
    values = trainer._behavior_log_probabilities([[first, second]])
    assert torch.equal(values[0], torch.tensor([-0.4, -0.2]))

