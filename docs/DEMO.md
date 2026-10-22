# Minimal Demo

The demo is intentionally host-only. It demonstrates the controller and update
recovery mechanisms without requiring MNN, an Android phone, model weights, a
dataset, or a cloud server.

## Scenario

The executable uses a compact EfficientNet-B0 structural manifest with five
representative semantic boundaries and RAW32, FP16, and INT8 codecs. It runs 30
requests:

- requests 1-10: moderate network condition and cold start;
- requests 11-20: bandwidth degradation and higher RTT;
- request 21: a local model-structure update;
- requests 21-30: post-update online operation.

The same trace is evaluated by Evo-Split and a cold-restart controller. Latency
observations are deterministic analytical measurements used only to keep the
demo portable. The paper artifact contains measurements from real devices.

## Run

Windows:

```powershell
.\examples\minimal_demo\run_demo.ps1
```

Linux:

```bash
bash examples/minimal_demo/run_demo.sh
```

## Outputs

```text
demo-results/demo_trace.csv
demo-results/demo_summary.png
```

The CSV records the selected action, actual and oracle latency, per-request
regret, probe flag, shortlist size, and update migration counters. The figure
contains an action timeline, cumulative regret curves, and the reuse,
recalibration, and removal counts at the update.

The demo illustrates mechanism behavior. It is not a replacement for the
real-device evaluation reported in the paper.

