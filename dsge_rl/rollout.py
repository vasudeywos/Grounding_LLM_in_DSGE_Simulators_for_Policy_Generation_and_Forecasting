from __future__ import annotations

from dataclasses import dataclass

import torch

from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.modeling import token_log_probs


@dataclass
class TurnRollout:
    input_ids: torch.Tensor
    action_mask: torch.Tensor
    reward: float
    text: str
    semantic_features: torch.Tensor | None = None
    behavior_log_probs: torch.Tensor | None = None


def render_chat(tokenizer, system: str, history: list[dict[str, str]], user: str) -> str:
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def collect_trajectory(model, tokenizer, environment: DSGEPolicyEnvironment, scenario, max_new_tokens: int, temperature: float) -> list[TurnRollout]:
    observation = environment.reset(scenario)
    system = environment.system_prompt()
    history: list[dict[str, str]] = []
    turns: list[TurnRollout] = []
    for _ in range(environment.config.turns):
        user = environment.user_prompt(observation)
        prompt = render_chat(tokenizer, system, history, user)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = encoded.input_ids.shape[1]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        transition = environment.step_text(text)
        mask = torch.zeros(output.shape[1] - 1, dtype=torch.bool, device=output.device)
        mask[prompt_length - 1 :] = True
        with torch.no_grad():
            logits = model(input_ids=output, use_cache=False).logits
            behavior_log_probs = token_log_probs(logits, output)[0][mask].detach()
        if was_training:
            model.train()
        turns.append(TurnRollout(output[0], mask, transition.reward, text, behavior_log_probs=behavior_log_probs))
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": text}])
        observation = transition.observation
    return turns


def collect_sfr_trajectory(
    model,
    tokenizer,
    semantic_encoder,
    environment: DSGEPolicyEnvironment,
    scenario,
    max_new_tokens: int,
    temperature: float,
) -> list[TurnRollout]:
    observation = environment.reset(scenario)
    system = environment.system_prompt()
    history: list[dict[str, str]] = []
    turns: list[TurnRollout] = []
    discourse = scenario.discourse or (scenario.name,) * (environment.config.turns + 1)
    for turn_index in range(environment.config.turns):
        previous_text = discourse[min(turn_index, len(discourse) - 1)]
        current_text = discourse[min(turn_index + 1, len(discourse) - 1)]
        semantic_features = semantic_encoder.encode_shift(previous_text, current_text).unsqueeze(0)
        user = environment.user_prompt(observation)
        prompt = render_chat(tokenizer, system, history, user)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = encoded.input_ids.shape[1]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                semantic_features=semantic_features,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output[0, prompt_length:] if output.shape[1] > prompt_length else output[0]
        full_ids = torch.cat([encoded.input_ids[0], completion_ids.to(encoded.input_ids.device)])
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        transition = environment.step_text(text)
        mask = torch.zeros(full_ids.shape[0] - 1, dtype=torch.bool, device=full_ids.device)
        mask[prompt_length - 1 :] = True
        with torch.no_grad():
            logits = model(full_ids.unsqueeze(0), semantic_features).logits
            behavior_log_probs = token_log_probs(logits, full_ids.unsqueeze(0))[0][mask].detach()
        if was_training:
            model.train()
        turns.append(
            TurnRollout(
                full_ids,
                mask,
                transition.reward,
                text,
                semantic_features.squeeze(0).cpu(),
                behavior_log_probs,
            )
        )
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": text}])
        observation = transition.observation
    return turns
