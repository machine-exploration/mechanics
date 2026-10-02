"""The quanta hypothesis on a toy where each quantum is known. Runs on a CPU in about two minutes.

    uv run python experiments/q0_quanta/quanta_toy.py
"""

import explorers as ex
from explorers import analysis, measures, toy

task = toy.MultitaskLookup(n_tasks=16, n_symbols=16, alpha=1.3)
trajectory = toy.train(task, steps=1500, every=50)
ds = (ex.Experiment(trajectory, task.examples())
      .measure(measures.example_loss, measures.stable_rank, measures.update_norm)
      .compute(store="runs/store"))

on = analysis.onsets(ds.example_loss)
print("median onset step per task (task 0 is the most frequent):")
print(on.onset.groupby("task").median().round().values)
print(f"rank correlation, task frequency vs onset: {analysis.spearman(ds.task_frequency, on.onset):+.2f}")
print(f"median sharpness (1 = one sudden drop): {float(on.sharpness.median()):.2f}")
