# Compact Paper Artifact

This directory contains a small processed-data subset for inspecting the main
Evo-Split claims without redistributing models, datasets, device binaries, or
multi-gigabyte raw traces.

## Included evidence

- `e3_*_summary.csv`: profile-free controller and baseline summaries for four
  device-model settings and multiple network trajectories.
- `e4_end_to_end_summary.csv`: measured end-to-end latency and payload summary.
- `e4_pareto_points.csv`: measured latency, payload, and accuracy points.
- `e5_update_ground_truth.csv`: deterministic model-update interventions.
- `e5_recovery_trials.csv`: request-level recovery trials.
- `e5_recovery_summary.csv`: update-type and method-level recovery summary.
- `data_manifest.json`: file hashes and sizes for integrity checking.

## Reproduce compact figures

```bash
python -m pip install -r artifact/requirements.txt
python artifact/scripts/verify_artifact.py
python artifact/scripts/reproduce_figures.py --out artifact/generated
```

Maintainers can regenerate the integrity manifest after intentionally replacing
the frozen CSV files:

```bash
python artifact/scripts/build_manifest.py
```

The scripts generate a normalized-regret comparison and an update-recovery
comparison. They are intended as a lightweight inspection path, not as a claim
that every paper table can be regenerated from the compact subset.

## Provenance

The CSV files are copied from frozen experiment outputs produced by the
Evo-Split research prototype. They contain processed measurements only. Device
identifiers are model labels rather than hardware serial numbers. Public server
addresses, local paths, credentials, and user identifiers are excluded.

