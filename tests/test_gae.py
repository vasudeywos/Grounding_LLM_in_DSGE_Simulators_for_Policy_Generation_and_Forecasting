import torch

from dsge_rl.trainers.ppo import PPOTrainer


def test_terminal_reward_propagates_to_earlier_tokens():
    trainer = object.__new__(PPOTrainer)
    trainer.config = type(
        "Config",
        (),
        {"training": type("Training", (), {"gamma": 0.99, "gae_lambda": 0.95})()},
    )()
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.zeros(3)
    advantages = trainer._gae(rewards, values)
    assert torch.all(advantages > 0)
    assert advantages[-1] == 1.0

