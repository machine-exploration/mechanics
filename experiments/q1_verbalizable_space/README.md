# Q1 — When does the verbalizable space form during training?

Finished language models hold a shared space of content they are disposed to say, which the Jacobian
lens reads (Anthropic, 2026). This experiment asks when that space appears across training, at which
layers, and how suddenly, on Pythia checkpoints.

**Measure.** At each checkpoint, the lens is fitted on 100 windows and evaluated on 100 others. Per
layer: how often the lens top-1 token differs from the model's own top-1 next token (`jlens_error`),
next to the same measure without transport (`logit_lens_error`, the baseline), and the loss. Then the
onset (when a layer's error has made half its drop) and the sharpness (1 = one sudden drop).

**First check.** On the final checkpoint, the J-lens error must be well below the logit-lens error in
the middle layers. If not, the finished-model result does not hold on Pythia, and that is the first
result.

## Run it (one GPU)

```bash
cd mechanics
uv sync

# 1. Texts: 400 documents of the Pile, one per line (needs Hugging Face access)
uv run --with datasets python -c "
from datasets import load_dataset
ds = load_dataset('NeelNanda/pile-10k', split='train')
open('texts.txt', 'w').write('\n'.join(t.replace('\n', ' ') for t in ds['text'][:400]))"

# 2. Check the pipeline without downloading a model (seconds)
uv run python experiments/q1_verbalizable_space/run.py --dry-run

# 3. Pythia 70m, 24 checkpoints
uv run python experiments/q1_verbalizable_space/run.py --texts texts.txt --size 70m --device cuda
```

Results are cached by content in `runs/q1/store`: if the run stops, run the same command again and
it resumes. The summary is written to `runs/q1/q1_pythia-70m.json` (config, study key, code version,
curves, onsets).

## Cost (estimates, to be replaced by measurements)

The J-lens needs `d_model` rows of the Jacobian per checkpoint; `--dim-batch 8` computes 8 per
backward pass. For Pythia 70m (d_model 512, 6 layers): 24 checkpoints × 13 batches × 64 passes ≈
20,000 backward passes on 64 sequences of 128 tokens. Rough order on an RTX 4070: tens of minutes to
about an hour. Each checkpoint is also a separate download (about 0.3 GB for 70m in the Hugging Face
cache).

- Out of memory: lower `--dim-batch` (4, 2, 1) or `--batch-size`. Results do not change.
- Pythia 160m (d_model 768, 12 layers) costs about 4× more; 410m (d_model 1024, 24 layers) about 15×.
  Start with 70m.

## Done when

A result note in `machine-exploration/public`, with this command, the study key and the code version,
positive or negative.
