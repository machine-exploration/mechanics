"""Q1: when does the workspace (the verbalizable space) form during training?

At each of N log-spaced Pythia checkpoints, fit the Jacobian lens on half the examples and evaluate it
on the other half. Per layer, the four workspace signatures of Gurnee et al. (2026, section 4.1):
  dimension    fraction of dimensions holding 90% of the variance of the J-lens vectors
  kurtosis     how peaked the readouts are (J-lens, and the logit lens as a baseline)
  persistence  how much more often the top readout repeats 4 positions later in the same text
  CKA          similarity of the J-lens vector sets between layers (early / workspace / motor blocks)
and, for the late "motor" regime, how often the lens top-1 differs from the model's next token
(`jlens_error`, `logit_lens_error`), and the loss. Then: when does each signature rise, at which
layers, and how suddenly?

    uv run python experiments/q1_verbalizable_space/run.py --texts texts.txt --size 70m --device cuda
    uv run python experiments/q1_verbalizable_space/run.py --dry-run      # tiny local models, no download

Results are cached by content in --store: rerun the same command after a crash and it resumes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import explorers as ex
from explorers import analysis, measures
from explorers.data import Examples

OFFSETS = (1, 4, 16)


def examples_from_texts(path: Path, tokenizer, seq_len: int, n_fit: int, n_eval: int) -> Examples:
    texts = [t for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
    ex_ = Examples.from_texts(texts, tokenizer, seq_len=seq_len, name=f"texts:{path.name}", max_examples=n_fit + n_eval)
    if len(ex_) < n_fit + n_eval:
        raise SystemExit(f"only {len(ex_)} windows of {seq_len} tokens in {path}; need {n_fit + n_eval}")
    return ex_.with_meta(split=np.array(["fit"] * n_fit + ["eval"] * n_eval))


def dry_run_setup():
    import torch
    import transformers

    def tiny(seed):
        torch.manual_seed(seed)
        cfg = transformers.GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                                         intermediate_size=64, max_position_embeddings=64)
        return ex.Model(transformers.GPTNeoXForCausalLM(cfg), name="tiny", revision=f"step{seed}")

    models = [tiny(s) for s in (0, 1, 2)]
    tokens = np.random.default_rng(0).integers(0, 64, size=(12, 24))
    examples = Examples(tokens=tokens, name="dry-run").with_meta(split=np.array(["fit"] * 6 + ["eval"] * 6))
    return models, examples, 4


def estimate(n_checkpoints: int, d_model: int, n_fit: int, batch_size: int, dim_batch: int) -> str:
    passes = n_checkpoints * int(np.ceil(n_fit / batch_size)) * int(np.ceil(d_model / dim_batch))
    return (f"{passes:,} backward passes in total ({n_checkpoints} checkpoints × "
            f"{int(np.ceil(n_fit / batch_size))} batches × {int(np.ceil(d_model / dim_batch))} passes per batch)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texts", type=Path, help="one document per line")
    p.add_argument("--size", default="70m", help="Pythia size: 70m, 160m, 410m …")
    p.add_argument("--checkpoints", type=int, default=24, help="log-spaced, always including step 0 and the last")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--n-fit", type=int, default=100, help="windows used to fit the lens")
    p.add_argument("--n-eval", type=int, default=100, help="windows used to evaluate it")
    p.add_argument("--skip-first", type=int, default=16, help="leading positions excluded (attention sinks)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dim-batch", type=int, default=8, help="rows of J per backward pass; lower it if memory runs out")
    p.add_argument("--target", default="penultimate", choices=["penultimate", "final"],
                   help="residual the Jacobian differentiates (the paper's default is penultimate)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default=None, help="e.g. float16; float32 by default")
    p.add_argument("--store", type=Path, default=Path("runs/q1/store"))
    p.add_argument("--out", type=Path, default=Path("runs/q1"))
    p.add_argument("--dry-run", action="store_true", help="tiny random local models, no download")
    args = p.parse_args(argv)

    if args.dry_run:
        models, examples, n_layers = dry_run_setup()
        skip, d_model, name = 2, 32, "dry-run"
        args.batch_size, args.dim_batch = 4, 4
    else:
        if args.texts is None:
            raise SystemExit("--texts is required (see the README in this folder), or use --dry-run")
        from transformers import AutoConfig, AutoTokenizer

        name = f"EleutherAI/pythia-{args.size}"
        cfg = AutoConfig.from_pretrained(name)
        n_layers, d_model, skip = cfg.num_hidden_layers, cfg.hidden_size, args.skip_first
        examples = examples_from_texts(args.texts, AutoTokenizer.from_pretrained(name), args.seq_len,
                                       args.n_fit, args.n_eval)
        models = ex.checkpoints(name, steps=ex.pick(ex.pythia_steps(), args.checkpoints), device=args.device,
                                dtype=args.dtype)
    layers = list(range(1, n_layers))
    print(f"{name}: {len(models)} checkpoints, layers {layers[0]}..{layers[-1]}, d_model {d_model}")
    print("estimate:", estimate(len(models), d_model, int((examples.meta['split'] == 'fit').sum()),
                                args.batch_size, args.dim_batch))

    t, k = args.target, skip
    experiment = (ex.Experiment(models, examples, batch_size=args.batch_size, dim_batch=args.dim_batch)
             .measure(measures.jlens_dimension(layers, k, t), measures.jlens_cka(layers, k, t),
                      measures.lens_kurtosis(layers, k, t), measures.lens_kurtosis(layers, k, lens="logit"),
                      measures.lens_persistence(layers, k, t, offsets=OFFSETS),
                      measures.lens_persistence(layers, k, lens="logit", offsets=OFFSETS),
                      measures.jlens_error(layers, k, t), measures.logit_lens_error(layers, k),
                      measures.loss))
    t0 = time.time()
    ds = experiment.compute(store=args.store, verbose=True)
    elapsed = time.time() - t0

    if "model" in ds.dims:                                  # dry run: Models without steps
        ds = ds.rename(model="step").assign_coords(step=[int(m.revision.removeprefix("step")) for m in models])
    steps = [int(s) for s in ds.step.values]
    persist = ds.jlens_persistence.sel(offset=4)
    # onsets() finds drops: pass rising signatures negated
    on_dim = analysis.onsets(-ds.jlens_dimension, min_drop=0.05)
    on_per = analysis.onsets(-persist, min_drop=0.2)
    on_err = analysis.onsets(ds.jlens_error, min_drop=0.05)

    def first_last(a):
        return f"{a[0]:>6.2f} → {a[-1]:<6.2f}"

    print(f"\n{'layer':>5} {'dimension':>16} {'onset':>7} {'kurtosis J':>16} {'kurtosis logit':>16} "
          f"{'persistence':>16} {'onset':>7} {'J-lens err':>16}")
    for layer in layers:
        sel = dict(layer=layer)
        print(f"{layer:>5} {first_last(ds.jlens_dimension.sel(**sel).values):>16} "
              f"{on_dim.onset.sel(**sel).item():>7.0f} {first_last(ds.jlens_kurtosis.sel(**sel).values):>16} "
              f"{first_last(ds.logit_lens_kurtosis.sel(**sel).values):>16} {first_last(persist.sel(**sel).values):>16} "
              f"{on_per.onset.sel(**sel).item():>7.0f} {first_last(ds.jlens_error.sel(**sel).values):>16}")
    print("\nCKA between layers at the last checkpoint:")
    print(np.array2string(ds.jlens_cka.isel(step=-1).values, precision=2, suppress_small=True))
    print(f"\nloss: {ds.loss.values[0]:.3f} → {ds.loss.values[-1]:.3f}; {elapsed:.0f}s")

    import importlib.metadata as md
    try:
        source = json.loads(md.distribution("explorers").read_text("direct_url.json") or "{}")
    except Exception:
        source = {}
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "question": "Q1: when does the workspace form during training?",
        "model": name, "experiment_key": experiment.key(), "explorers": ex.__version__, "explorers_source": source,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "examples": {"fingerprint": examples.fingerprint, "n": len(examples)},
        "steps": steps, "layers": layers, "offsets": list(OFFSETS), "seconds": elapsed,
        **{v: ds[v].values.tolist() for v in ds.data_vars},
        "dimension_onset": on_dim.onset.values.tolist(), "dimension_sharpness": on_dim.sharpness.values.tolist(),
        "persistence_onset": on_per.onset.values.tolist(), "persistence_sharpness": on_per.sharpness.values.tolist(),
        "jlens_error_onset": on_err.onset.values.tolist(),
    }
    out = args.out / f"q1_{name.split('/')[-1]}.json"
    out.write_text(json.dumps(result, indent=1, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
