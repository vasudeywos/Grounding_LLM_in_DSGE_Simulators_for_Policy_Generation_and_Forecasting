# Grounded DSGE Policy RL

This repository contains a Snowdrop DSGE interface, multi-turn GRPO and PPO policy training, a standalone semantic-shift model, and SFR-conditioned PPO across multiple economic simulators.

## Code flow

### DSGE policy training

```text
DSGE_Simulator/*.yaml
        ↓
dsge_rl/snowdrop.py
        ↓
dsge_rl/environment.py
        ↓
dsge_rl/rollout.py
        ↓
dsge_rl/trainers/grpo.py or dsge_rl/trainers/ppo.py
        ↓
outputs/<experiment>/
```

### Semantic-shift training

```text
data/economic_discourse.jsonl
        ↓
semantic_shift/data.py
        ↓
semantic_shift/model.py
        ↓
semantic_shift/trainer.py
        ↓
outputs/semantic_bert/
```

### SFR-conditioned PPO

```text
Consecutive economic discourse
        ↓
Salesforce/SFR-Embedding-2_R
        ↓
dsge_rl/sfr_conditioning.py
        ↓
Qwen policy and separate Qwen critic
        ↓
dsge_rl/trainers/sfr_ppo.py
        ↓
All registered DSGE simulators
```

## Important files

| File | Purpose |
|---|---|
| `DSGE_Simulator/` | Snowdrop YAML models used by the environments |
| `dsge_rl/snowdrop.py` | Loads models, injects shock paths, runs Snowdrop, and returns simulation DataFrames |
| `dsge_rl/simulator_registry.py` | Defines targets, levers, scenarios, and discourse for every simulator |
| `dsge_rl/environment.py` | Converts simulator trajectories into observations and rewards |
| `dsge_rl/semantics.py` | Extracts model descriptions and validates policy-action JSON |
| `dsge_rl/modeling.py` | Loads the Qwen policy and separate PPO value model |
| `dsge_rl/rollout.py` | Runs multi-turn policy and simulator interaction |
| `dsge_rl/trainers/grpo.py` | GRPO training |
| `dsge_rl/trainers/ppo.py` | PPO training with a separate critic and token-level GAE |
| `dsge_rl/sfr_conditioning.py` | SFR discourse embeddings and Qwen conditioning bridges |
| `dsge_rl/trainers/sfr_ppo.py` | SFR-conditioned PPO across simulator tasks |
| `semantic_shift/data.py` | Loads and pairs chronological discourse windows |
| `semantic_shift/model.py` | Temporal semantic-shift encoder and embedding heads |
| `semantic_shift/trainer.py` | Semantic model training and checkpointing |
| `configs/` | Settings for standalone GRPO, PPO, and semantic experiments |
| `tests/` | Unit tests for parsing, GAE, environments, temporal data, and simulator registration |

## Environment setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the project:

```bash
pip install -e ".[test]"
```

The training commands download Qwen, BERT, or SFR weights from Hugging Face when the models are not already cached.

## Snowdrop setup

The interface in `dsge_rl/snowdrop.py` imports:

```text
snowdrop.src.driver.importModel
snowdrop.src.driver.run
```

Install Snowdrop from the Framework repository:

```bash
git clone https://github.com/gumilevskij/Framework.git
pip install -r Framework/requirements.txt
pip install Framework/dist/snowdrop-1.0.8-py3-none-any.whl
```

Verify the installation:

```bash
python -c "from snowdrop.src.driver import importModel, run; print('Snowdrop ready')"
```

The simulator models must remain under:

```text
DSGE_Simulator/Ireland2004.yaml
DSGE_Simulator/MVF_US.yaml
DSGE_Simulator/RBC.yaml
DSGE_Simulator/gsw_model.yaml
DSGE_Simulator/model.yaml
DSGE_Simulator/sw_model.yaml
```

`dsge_rl/simulator_registry.py` resolves these paths automatically for SFR-PPO and cross-simulator testing.

## Run the standalone semantic model

Prepare a dataset from the included example:

```bash
cp data/economic_discourse.example.jsonl data/economic_discourse.jsonl
```

Train the model:

```bash
semantic-train --config configs/semantic_bert.yaml
```

Export semantic and shock embeddings:

```bash
semantic-encode \
  --checkpoint outputs/semantic_bert/best \
  --input data/economic_discourse.jsonl \
  --output outputs/semantic_embeddings.jsonl
```

The required JSONL schema and checkpoint contents are described in `SEMANTIC_MODEL.md`.

## Run GRPO

```bash
dsge-grpo --config configs/qpm_grpo.yaml
```

Output:

```text
outputs/qpm_grpo/
├── checkpoint-*/
├── final/
└── metrics.jsonl
```

## Run PPO

```bash
dsge-ppo --config configs/qpm_ppo.yaml
```

Output:

```text
outputs/qpm_ppo/final/
├── policy/
└── value_model/
```

Evaluate the PPO policy adapter:

```bash
dsge-evaluate \
  --config configs/qpm_ppo.yaml \
  --adapter outputs/qpm_ppo/final/policy
```

## Run SFR-conditioned PPO

This command uses the in-code simulator registry and does not require a configuration file:

```bash
dsge-ppo-sfr \
  --simulators all \
  --output-dir outputs/sfr_ppo
```

Run a subset of simulators:

```bash
dsge-ppo-sfr \
  --simulators qpm,sw,gsw \
  --output-dir outputs/sfr_ppo_macro
```

Available registry names are:

```text
qpm, sw, gsw, ireland, mvf_us, rbc
```

The SFR model uses four-bit loading by default. Disable it with `--no-sfr-4bit` when appropriate for the available hardware.

## Run cross-simulator testing

```bash
dsge-cross-test \
  --checkpoint outputs/sfr_ppo/final \
  --simulators all \
  --output outputs/cross_simulator_results.json
```

The result file records the simulator, scenario, generated actions, per-turn rewards, trajectory return, and any simulator-specific failure.

## Run tests

```bash
pytest -q
```

