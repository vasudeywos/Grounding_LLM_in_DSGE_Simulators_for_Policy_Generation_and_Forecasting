# SFR-Conditioned Cross-Simulator PPO

This path keeps the Qwen policy, separate Qwen critic, PPO clipping, token-level GAE, reward definition, horizon, and four-turn interaction structure unchanged. It adds a frozen `Salesforce/SFR-Embedding-2_R` discourse encoder and trainable semantic bridges.

For each policy turn, SFR embeds the previous and current discourse states. The conditioning vector concatenates the current normalized embedding, its directional change from the preceding embedding, and cosine shift. A policy bridge maps this vector into Qwen's hidden dimension and adds it to every prompt-token embedding. The separate critic has its own independently trained bridge.

The SFR model is frozen during PPO. PPO trains the Qwen LoRA policy adapter, policy semantic bridge, separate Qwen critic adapter, critic value head, and critic semantic bridge. The checkpoint does not duplicate the SFR weights; it records the Salesforce model identifier and bridge settings.

`Salesforce/SFR-Embedding-2_R` is a 7B embedding model. Four-bit loading is enabled by default because the PPO process also holds the Qwen policy and separate Qwen critic. Review the model's CC-BY-NC-4.0 license before commercial use.

## Simulator registry

`dsge_rl/simulator_registry.py` registers every YAML model in `DSGE_Simulator/`:

- `model.yaml` as `qpm`
- `sw_model.yaml` as `sw`
- `gsw_model.yaml` as `gsw`
- `Ireland2004.yaml` as `ireland`
- `MVF_US.yaml` as `mvf_us`
- `RBC.yaml` as `rbc`

Each entry defines model-appropriate state targets, shock magnitudes, policy levers, steady-state targets, crisis scenarios, and a turn-by-turn discourse sequence. The registry replaces runtime experiment YAML configuration for SFR-PPO. Existing files under `configs/` remain available for the earlier standalone QPM and semantic experiments but are not read by the new commands.

## Training

```bash
pip install -e .
dsge-ppo-sfr --simulators all --output-dir outputs/sfr_ppo
```

Select a subset with comma-separated registry names:

```bash
dsge-ppo-sfr --simulators qpm,sw,gsw --output-dir outputs/sfr_ppo_macro
```

## Cross-simulator testing

```bash
dsge-cross-test \
  --checkpoint outputs/sfr_ppo/final \
  --simulators all \
  --output outputs/cross_simulator_results.json
```

Every scenario receives an independent multi-turn rollout. The report records simulator name, scenario, per-turn simulator rewards, cumulative return, actions, and failures. A failure in one model is recorded without suppressing results from the remaining models.

