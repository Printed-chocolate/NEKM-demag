# Boundary Operator Feasibility Experiment

This folder tests the fixed-geometry boundary map described in
`work-outline/main.tex`:

```text
g|Gamma -> phi2|Gamma
```

The script reads the 60 extracted files from:

```text
~/AMGX-afemag/examples/demag_timing_5/results_extracted_r3_from_r6/
```

Each file is expected to contain:

```text
boundary_dof x y z g phi2_boundary_integral
```

Run:

```bash
python3 run_experiment.py --out-dir outputs
```

Outputs:

- `outputs/metrics.json`: train/test split, data ranges, and error metrics.
- `outputs/predictions.npz`: boundary coordinates, test targets, and predictions.
- `outputs/plots/latent_mlp_training_loss.png`: latent MLP training curve.
- `outputs/plots/test_relative_l2_errors.png`: held-out relative errors.
- `outputs/plots/worst_case_boundary_errors.png`: boundary scatter plot for the
  worst held-out case, showing true `phi2` and absolute errors.

Current result from the default split:

| model | mean relative L2 |
|---|---:|
| mean phi baseline | 0.3160 |
| scaled g baseline | 2.0055 |
| linear kernel ridge | 0.0072 |
| latent MLP | 0.0216 |

Interpretation: on these 60 samples, learning the boundary operator is feasible
for the sampled distribution. The strong linear ridge result is expected because
the single-layer boundary operator is linear for fixed geometry. A neural model
should be compared against this linear baseline, not just against trivial
baselines.
