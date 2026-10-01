# mechanics

**Mechanics** is Machine Exploration's science: the programs that run on the computer, and what they find. How training creates computation (how representations, algorithms and circuits form, change and give rise to behaviour), and what training puts inside a model before behaviour shows it.

It is built on [`explorers`](https://github.com/machine-exploration/explorers), the computer, open source. This repository holds research only: experiments, datasets, figures and papers. Library code goes to `explorers`; environments go to the [`verifiers` fork](https://github.com/machine-exploration/verifiers/tree/main/environments).

Part of [Machine Exploration](https://github.com/machine-exploration/public): see the [vision and architecture](https://github.com/machine-exploration/public#readme) and the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md).

## Experiments

| Experiment | Question | Stage | Status |
|---|---|---|---|
| Watching a model learn to cheat (first program) | During RL on [`impossible_code`](https://github.com/machine-exploration/verifiers/tree/main/environments/impossible_code), where any pass is a reward hack, does the representation of hacking rise inside before the hack rate does? Read at every step's checkpoint by a monitor built from words alone, a difference-of-means probe and the logit lens. Controls: a planted concept and an environment that cannot be hacked. | Post-training | Next. The environment and episode replay are built; the RL run is not. |
| [`experiments/q0_quanta`](experiments/q0_quanta) | When, and in what order, does a model learn what it learns? | Pretraining | On a toy with tasks of Zipf frequencies: frequent tasks are learned first (rank correlation −0.74), each suddenly (median sharpness 0.74). Pythia next. |
| [`experiments/q1_verbalizable_space`](experiments/q1_verbalizable_space) | When does the space read by the Jacobian lens form during pretraining? | Pretraining | Ready to run on one GPU; the dry run passes. Later. |

Each experiment ships with the run that reproduces it: config, seed, data order, model version and code version. Negative results are published too.

## Setup

```bash
uv sync                                   # installs explorers from GitHub
uv run python experiments/q0_quanta/quanta_toy.py
```

The library's history up to 2026-09-29 (the core primitives, the learning package and the Jacobian lens) was merged into `explorers`, where it continues.
