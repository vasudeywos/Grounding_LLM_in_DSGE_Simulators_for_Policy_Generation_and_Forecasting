# Grounded DSGE Policy RL

## System flow

### DSGE reinforcement learning

```text
DSGE_Simulator/*.yaml
        ↓
SnowdropSimulator
        ↓
DSGEPolicyEnvironment
        ↓
Multi-turn trajectory collection
        ↓
GRPO or PPO
        ↓
Policy checkpoints and metrics
```

### Simulator interaction

```text
Economic state
        ↓
Qwen policy action
        ↓
Validated shock JSON
        ↓
Snowdrop shock path
        ↓
Full-horizon DSGE simulation
        ↓
Economic loss and turn reward
        ↓
Next economic state
```

### PPO

```text
Qwen policy model
        ↓
Policy token probabilities

Separate Qwen critic model
        ↓
Token value estimates

Simulator rewards + token values
        ↓
GAE
        ↓
Clipped policy and value objectives
```

### Semantic-shift model

```text
Time-indexed economic discourse
        ↓
Chronological discourse windows
        ↓
Transformer document states
        ↓
Temporal semantic difference
        ↓
Latent shock embedding
        ↓
Named economic-shock estimates
```

### SFR-conditioned PPO

```text
Previous and current discourse
        ↓
Salesforce/SFR-Embedding-2_R
        ↓
Current embedding + directional shift + cosine shift
        ↓
Independent policy and critic bridges
        ↓
Conditioned Qwen policy and critic
        ↓
Multi-simulator PPO
```

### Cross-simulator evaluation

```text
Trained policy
        ↓
Simulator registry
        ↓
QPM | SW | GSW | Ireland | MVF-US | RBC
        ↓
Scenario rollouts
        ↓
Actions, rewards, returns, and failures
```

### Pandemic validation

```text
Mendeley pandemic shock workbook
        ↓
Country and scenario selection
        ↓
G-Cubed-to-GSW shock mapping
        ↓
GSW simulation and optional policy rollout
        ↓
Published G-Cubed outcome comparison
        ↓
RMSE, correlation, and policy-loss improvement
```

## Repository structure

### DSGE models

| Path | Description |
|---|---|
| `DSGE_Simulator/model.yaml` | Quarterly Projection Model |
| `DSGE_Simulator/sw_model.yaml` | Smets-Wouters model |
| `DSGE_Simulator/gsw_model.yaml` | Galí-Smets-Wouters pandemic model |
| `DSGE_Simulator/Ireland2004.yaml` | Ireland New Keynesian technology-shock model |
| `DSGE_Simulator/MVF_US.yaml` | US multivariate-filter model |
| `DSGE_Simulator/RBC.yaml` | Real Business Cycle model |

### Simulator and environment

| Path | Description |
|---|---|
| `dsge_rl/snowdrop.py` | Snowdrop model loading, shock injection, parameter updates, cloning, and simulation output |
| `dsge_rl/environment.py` | Multi-turn economic state, accumulated shock paths, economic loss, and reward calculation |
| `dsge_rl/semantics.py` | Model semantic extraction and policy-action JSON validation |
| `dsge_rl/simulator_registry.py` | Model-specific targets, policy levers, crisis scenarios, discourse, and simulator paths |
| `dsge_rl/config.py` | Shared model, environment, scenario, target, lever, and training data structures |

### Policy models and rollouts

| Path | Description |
|---|---|
| `dsge_rl/modeling.py` | Qwen policy loading, LoRA configuration, separate token-value model, and adapter loading |
| `dsge_rl/rollout.py` | Standard and SFR-conditioned multi-turn trajectory collection |
| `dsge_rl/sfr_conditioning.py` | SFR shift encoder and independent policy and critic conditioning bridges |

### Reinforcement-learning trainers

| Path | Description |
|---|---|
| `dsge_rl/trainers/common.py` | Random seeding, trainable-parameter selection, and metric persistence |
| `dsge_rl/trainers/grpo.py` | Group-relative trajectory optimization |
| `dsge_rl/trainers/ppo.py` | PPO with separate critic, token-level GAE, policy clipping, and value clipping |
| `dsge_rl/trainers/sfr_ppo.py` | SFR-conditioned PPO over multiple registered simulators |

### Semantic-shift model

| Path | Description |
|---|---|
| `semantic_shift/config.py` | Semantic dataset, model, and training structures |
| `semantic_shift/data.py` | JSONL loading, chronological pairing, tokenization, document padding, and batching |
| `semantic_shift/model.py` | Temporal transformer encoder, semantic-shift estimator, shock embedding, and learning objectives |
| `semantic_shift/trainer.py` | Semantic model optimization, chronological validation, early stopping, and checkpoint export |

### Pandemic validation

| Path | Description |
|---|---|
| `The Global Economic Impacts of the COVID-19 Pandemic/` | Published epidemiological assumptions, shocks, G-Cubed results, figure data, README, and R script |
| `dsge_rl/pandemic_validation.py` | Pandemic workbook loading, GSW shock mapping, annual result alignment, and validation metrics |
| `dsge_rl/cli/pandemic_validate.py` | Pandemic validation entry point with optional SFR-PPO evaluation |

### Command entry points

| Path | Description |
|---|---|
| `dsge_rl/cli/grpo.py` | GRPO entry point |
| `dsge_rl/cli/ppo.py` | PPO entry point |
| `dsge_rl/cli/ppo_sfr.py` | Cross-simulator SFR-PPO entry point |
| `dsge_rl/cli/evaluate.py` | Policy adapter evaluation entry point |
| `dsge_rl/cli/cross_test.py` | Cross-simulator evaluation entry point |
| `semantic_shift/cli/train.py` | Semantic model training entry point |
| `semantic_shift/cli/encode.py` | Semantic embedding export entry point |

### Configuration and data

| Path | Description |
|---|---|
| `configs/qpm_grpo.yaml` | QPM GRPO experiment values |
| `configs/qpm_ppo.yaml` | QPM PPO experiment values |
| `configs/semantic_bert.yaml` | Standalone semantic-model experiment values |
| `data/economic_discourse.example.jsonl` | Example time-indexed discourse and shock-target schema |
| `pyproject.toml` | Package metadata, dependencies, and command registration |

### Tests

| Path | Description |
|---|---|
| `tests/test_semantics.py` | Policy-action parsing and constraints |
| `tests/test_gae.py` | Delayed-reward propagation through GAE |
| `tests/test_environment.py` | Per-turn simulator execution and reward production |
| `tests/test_semantic_data.py` | Chronological discourse pairing and leakage prevention |
| `tests/test_simulator_registry.py` | Simulator-file coverage and scenario discourse |
| `tests/test_pandemic_validation.py` | Pandemic shock loading, Scenario 6 handling, and GSW mapping |
