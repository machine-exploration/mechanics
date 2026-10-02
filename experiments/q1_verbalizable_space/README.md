# Q1 — When does the workspace form during training?

Finished language models hold a small, privileged set of verbalizable representations, read by the
Jacobian lens, that behaves like a global workspace (Gurnee et al., "Verbalizable Representations Form
a Global Workspace in Language Models", Transformer Circuits, 2026). The paper finds it in base models
already, and leaves open "how much earlier in pretraining it emerges, whether it appears gradually or
abruptly, or how it scales with model size" (its section 9.1). This experiment asks exactly that, on
Pythia checkpoints.

**Measures.** At each checkpoint the lens is fitted on 100 windows (Jacobian of the penultimate
residual, the paper's default) and evaluated on 100 others. Per layer, the paper's four workspace
signatures (section 4.1):

| Signature | Measure | At the workspace onset |
|---|---|---|
| dimension | `jlens_dimension` | the J-lens vectors fan out from a low-dimensional subspace |
| kurtosis | `lens_kurtosis` (J-lens and logit lens) | readouts become peaked on a few tokens |
| persistence | `lens_persistence` | the top readout repeats across positions of the same text, more than across texts |
| CKA | `jlens_cka` | layers group into early / workspace / motor blocks |

Plus, for the late "motor" regime, `jlens_error` and `logit_lens_error` (disagreement with the
model's next token: the J-lens is not built to predict it, so these are not workspace measures), and
the loss. Then per layer: the onset (step at which a signature has made half its rise) and the
sharpness (1 = one sudden jump).

**First check.** On the final checkpoint, the signatures must show the paper's layer structure: low
dimension and kurtosis in the first third of the layers, a middle block with high persistence, a
separate late block in the CKA. If not, the finished-model result does not hold on this Pythia size,
and that is the first result.

**Size.** Pythia 70m has 6 layers; the workspace starts about a third of the way in, and the paper
does not know whether small models have one. Run 70m to check the pipeline and as the small-model
point; the answer needs 410m (and 1.4b if it fits).

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
it resumes. The summary is written to `runs/q1/q1_pythia-70m.json` (config, experiment key, code version,
every signature per checkpoint and layer, onsets).

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

A result note in `machine-exploration/public`, with this command, the experiment key and the code version,
positive or negative.
