# mechanics

**Mechanics** is Machine Exploration's research program: how training creates computation — how representations, algorithms and circuits form, change and give rise to behavior.

It is built on [`explorers`](https://github.com/machine-exploration/explorers), the open-source library. This repository holds research only: experiments, datasets and papers. Library code goes to `explorers`.

Part of [Machine Exploration](https://github.com/machine-exploration/public): see the [vision](https://github.com/machine-exploration/public#readme) and the [roadmap](https://github.com/machine-exploration/public/blob/main/ROADMAP.md). The first studies run on evals and post-training runs; pretraining studies (q0, q1) come later on the same engine.

## Experiments

| Folder | Question | Status |
|---|---|---|
| [`experiments/q0_quanta`](experiments/q0_quanta) | When, and in what order, does a model learn what it learns? | On a toy with tasks of Zipf frequencies: frequent tasks are learned first (rank correlation −0.74), each suddenly (median sharpness 0.74). Pythia next. |
| [`experiments/q1_verbalizable_space`](experiments/q1_verbalizable_space) | When does the space read by the Jacobian lens form during pretraining? | Ready to run on one GPU; the dry run passes. Later (pretraining). |
| `experiments/o2_planted_concept` (planned) | A concept planted by a prime-rl LoRA fine-tune: which white-box methods (J-lens, probe, logit lens) see it inside before behaviour, at what cost? | Next (roadmap O2). |

Each experiment ships with the run that reproduces it: config, seed, data order, model version and code version. Negative results are published too.

## Setup

```bash
uv sync                                   # installs explorers from GitHub
uv run python experiments/q0_quanta/quanta_toy.py
```

The library's history up to 2026-09-29 (the core primitives, the learning package and the Jacobian lens) was merged into `explorers`, where it continues.
