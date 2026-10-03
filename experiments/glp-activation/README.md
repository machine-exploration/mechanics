# glp-activation — a generative model of activations, across training

Luo, Feng, Darrell, Radford and Steinhardt ([*Learning a Generative Meta-Model of LLM Activations*](https://arxiv.org/abs/2602.06964), ICML 2026; [code](https://github.com/g-luo/generative_latent_prior), MIT) train diffusion models on one billion residual-stream activations of Llama models. The model, GLP (generative latent prior), is a deep MLP trained by flow matching, with no structural assumptions, unlike PCA or sparse autoencoders. They report that the diffusion loss falls smoothly with compute and predicts downstream utility; that using the prior to bring steered activations back onto the learned distribution improves fluency, more so as the loss falls; and that the meta-model's neurons isolate concepts more as the loss falls, with sparse-probing scores that scale with it.

They fit the prior on a finished model. This experiment fits it along training: **when does the distribution of a network's internal states acquire its structure, and how does that relate to when the network learns its skills?** A macroscopic quantity (the prior's loss) and a microscopic one (concepts isolated in its neurons), measured at every checkpoint.

## Stages

| Stage | What | Done when |
|---|---|---|
| 1. Reproduce | Fit priors on the activations of their Llama 1B model with the paper's recipe, first on their 1M-activation quickstart, then at scale; compare with their released checkpoints (`glp-llama1b-d3`, `-d6`, `-d12`, `-d24`) | Their probing scores matched within a stated tolerance, on our pipeline |
| 2. Across training | Fit a prior on the activations of every checkpoint of a run: a small organism trained on designed data (planted skills at controlled frequencies), then Pythia | Prior loss and meta-neuron probing scores over training steps, per layer, across seeds |
| 3. Against skills | Compare the onsets of structure in the prior with the onsets of the planted skills (accuracy per skill over steps) | The relation measured, whatever it is |

## Hypotheses (fixed before stage 2 runs)

- **H1.** Structure in the prior (concept-isolating meta-neurons, by sparse probing) appears in steps that line up with skill onsets, rather than smoothly.
- **H2.** For each planted skill, its concept becomes isolated in the prior at or before the step its accuracy jumps.
- **H3.** A prior fitted at the final checkpoint, evaluated on earlier checkpoints' activations, has a loss that falls in steps at the same onsets (the distribution shifts when skills are learned).

Controls: an organism trained on shuffled-frequency data; priors fitted on different seeds of the same run (universality of meta-neurons); a prior on activations of a randomly initialised model.

## How it runs

The experiment calls the explorers API; no library code lives here. It is a job (`train.py:main`) sent by a `Client` to a `Runtime`: first `LocalRuntime` on the CPU with a tiny model (built; checks the contract), then GPU workers. It needs `stream` (one model's activations fed to the prior's training loop) and the Modal runtime (roadmap O3).

```python
import explorers as ex

# train.py: the job, run on any runtime
def main(model, step, layer, seed, tokens):
    m = ex.open(model, revision=step)
    acts = ex.stream(m, env, reads=f"residual[{layer}]", tokens=tokens)
    glp = ex.model(ex.configs.mlp_denoiser(depth=6, d=m.d_model), seed=seed)
    for x in acts:
        glp.forward_backward(x, loss=ex.losses.flow_matching)
        glp.optim_step()
    return {"loss": glp.loss, "probe": sparse_probing(glp)}

# from the client
ex.Client(ex.LocalRuntime()).run("experiments/glp-activation/train.py:main",
                                 model="tiny", step=0, layer=1, seed=0, tokens=10_000)   # CPU: contract check
client = ex.Client(ModalRuntime())
jobs = [client.submit("experiments/glp-activation/train.py:main", ex.Resources(gpu="A100"),
                      model=RUN, step=t, layer=l, seed=s, tokens=TOKENS)
        for t in STEPS for l in (4, 8, 12) for s in range(3)]
```

Illustrative until roadmap O3 is built. Cost reference from the paper's repository: a billion activations take about 5.6 days on two A100 80 GB GPUs (one caching activations, one training); most of its scripts fit in 24 GB. Stage 2 uses far fewer tokens per checkpoint; the budget per checkpoint is fixed after stage 1.

## Status

Planned, after Collective Adaptive Stress Testing. Blocked on the explorers training primitives (roadmap O3). Stage 1 can start with the paper's own code to pin down the target numbers.

Each result ships with the run that reproduces it: config, seed, data order, model version and code version. Negative results are published too.
