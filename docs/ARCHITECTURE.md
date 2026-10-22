# Architecture

Evo-Split separates the system into a portable control plane and an optional
MNN data plane.

## Split inference substrate

`CutSetPlanner` treats a neural network as a DAG and finds every tensor crossing
a candidate split boundary. `SplitInferenceEngine` materializes the complete
cut-set on the edge. `TensorSerializer` packages tensor identity, shape, type,
layout, payload, and codec metadata. `CloudInjectionEngine` binds the bundle to
the corresponding consumer inputs in the cloud graph.

This design avoids assuming that a split is represented by one visible layer
output. It remains applicable to residual and multi-branch models and to
runtime configurations that fuse operators or reuse memory.

## Activation codec

`ActivationCodec` provides training-free RAW32, FP16, block INT8, block INT6,
block INT4, and LZ4 modes. A codec profile records the original and encoded
sizes, encode and decode time, reconstruction error, and fallback status.

## Profile-free online control

`ProfileFreeRuntime` builds joint actions consisting of local execution, cloud
execution, or a semantic boundary plus codec. It then:

1. removes unsafe or incompatible actions;
2. prunes clearly uncompetitive actions using structural lower bounds;
3. predicts latency from shared edge, cloud, codec, capture, injection, and
   network components;
4. performs bounded safe exploration;
5. updates component rates and action residuals from observations.

The controller therefore does not require a complete device-model action table
before serving requests.

## Model-evolution-aware recovery

`VersionedUpdateManager` compares versioned model manifests and creates a model
change set. A decision dependency graph computes the transitive closure of
affected artifacts. Semantic boundary contracts are mapped as exact,
contract-compatible, calibration-required, or incompatible. Controller state is
then reused, migrated, requalified, or removed accordingly.

The update coordinator implements PREPARE, READY, COMMIT, ABORT, and ROLLBACK
states. Edge/cloud hashes, manifests, and protocol versions must agree before a
split action is admitted.

## Packaging

`EvoSplit::core` contains platform-independent control, codec, serialization,
and update logic. `EvoSplit::mnn_runtime` contains the optional MNN capture,
injection, and TCP data plane. Models and datasets are external inputs.

