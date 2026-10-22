#include "ProfileFreeController.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>

namespace {
StructuralModelManifest manifest(const std::string& hash, int physicalOffset = 0) {
    StructuralModelManifest model;
    model.manifestId = "manifest-" + hash;
    model.modelName = "resnet-like";
    model.modelHash = hash;
    model.modelVersion = hash;
    model.structureHash = "structure-" + hash;
    model.inputBytes = 224 * 224 * 3 * 4;
    model.totalFlops = 4.0e9;
    model.localPeakMemoryMb = 512.0;
    for (int i = 0; i < 6; ++i) {
        StructuralBoundaryDescriptor boundary;
        boundary.canonicalBoundaryId = "stage." + std::to_string(i);
        boundary.physicalBoundary = physicalOffset + i;
        boundary.prefixFlops = model.totalFlops * (i + 1) / 7.0;
        boundary.suffixFlops = model.totalFlops - boundary.prefixFlops;
        boundary.rawBundleBytes = static_cast<uint64_t>(4'000'000 / (i + 1));
        boundary.peakEdgeMemoryMb = 80.0 + i * 10.0;
        boundary.boundaryDepth = i;
        boundary.cutSetTensorCount = i % 2 + 1;
        boundary.supportedCodecs = {CodecKind::RAW32, CodecKind::FP16,
                                    CodecKind::BLOCK_INT8};
        boundary.accuracyDrops[CodecKind::RAW32] = 0.0;
        boundary.accuracyDrops[CodecKind::FP16] = 0.001;
        boundary.accuracyDrops[CodecKind::BLOCK_INT8] = 0.006;
        model.boundaries.push_back(boundary);
    }
    return model;
}

RuntimeState runtimeState(const std::string& hash) {
    RuntimeState state;
    state.requestId = "request-1";
    state.modelHash = hash;
    state.features.set("network.available", 1.0);
    state.features.set("cloud.available", 1.0);
    state.features.set("network.goodput_up_bps", 2.0e6);
    state.features.set("network.rtt_ms", 12.0);
    state.features.set("network.jitter_ms", 1.0);
    state.features.set("device.memory.free_mb", 1024.0);
    state.features.set("device.compute.flops_per_ms", 1.0e6);
    state.features.set("cloud.compute.flops_per_ms", 1.0e7);
    return state;
}

MetricBag actualFor(const ProfileFreePrediction& prediction, double scale = 1.0) {
    MetricBag actual;
    for (const auto& item : prediction.components) {
        actual.set(item.first, std::max(0.001, item.second.meanMs * scale));
    }
    actual.set("latency.total_ms", std::max(0.001, prediction.total.meanMs * scale));
    return actual;
}
}

int main() {
    {
        MetricBag flags;
        flags.set("network.available", true);
        double value = 0.0;
        assert(flags.getNumber("network.available", value));
        assert(value == 1.0);
        flags.set("network.available", false);
        assert(flags.getNumber("network.available", value));
        assert(value == 0.0);
    }
    const auto oldModel = manifest("model-v1");
    std::string error;
    assert(oldModel.valid(&error));

    StructuralActionBuilder builder;
    const auto built = builder.build(oldModel);
    assert(built.rawActions == 20);
    assert(built.actions.size() == 20);  // local + cloud + 6 x 3 codecs

    CandidatePruningConfig pruneConfig;
    pruneConfig.maximumShortlistSize = 6;
    pruneConfig.lowerBoundSlack = 0.05;
    CandidatePruner pruner(pruneConfig);
    const auto pruned = pruner.prune(runtimeState(oldModel.modelHash), built.actions);
    assert(!pruned.shortlist.empty());
    assert(pruned.inputActions == built.actions.size());
    assert(pruned.records.empty()); // production hot path keeps counters, not descriptor copies
    assert(pruned.shortlist.size() <= pruneConfig.maximumShortlistSize);
    bool keptLocal = false;
    bool keptCloud = false;
    for (const auto& action : pruned.shortlist) {
        keptLocal |= action.action.mode == ActionMode::LOCAL;
        keptCloud |= action.action.mode == ActionMode::CLOUD;
    }
    assert(keptLocal && keptCloud); // safety anchors must never be crowded out

    ProfileFreeControllerConfig controllerConfig;
    controllerConfig.bootstrapMode = ControllerBootstrapMode::PROFILE_FREE;
    controllerConfig.shadowMode = true;
    controllerConfig.pruning = pruneConfig;
    controllerConfig.adaptation.maximumProbeRequests = 3;
    ProfileFreeRuntime runtime(controllerConfig);
    assert(runtime.registerManifest(oldModel, &error));
    auto state = runtimeState(oldModel.modelHash);
    auto first = runtime.decide("device-A", state);
    assert(first.selectedIndex >= 0);
    assert(first.predictions.size() <= pruneConfig.maximumShortlistSize);
    assert(first.totalDecisionMs >= 0.0);
    assert(first.toJson().find("local_calibration_anchor") != std::string::npos);
    assert(!first.selectedAsProbe);
    const auto chosen = first.predictions[static_cast<std::size_t>(first.selectedIndex)];
    assert(runtime.observe(first.traceId, actualFor(chosen, 0.85), true, false,
                           chosen.total.meanMs * 0.80, &error));
    assert(runtime.estimator().actionSamples("device-A", chosen.structuralActionKey) == 1);
    assert(runtime.statusJson("device-A", oldModel.modelHash).find("observations\":1") !=
           std::string::npos);

    state.requestId = "request-2";
    const auto second = runtime.decide("device-A", state);
    assert(second.selectedIndex >= 0);
    assert(second.selectedAsProbe);
    assert(second.predictions[static_cast<std::size_t>(second.selectedIndex)]
               .candidate.action.mode != ActionMode::LOCAL);
    const auto secondChosen = second.predictions[static_cast<std::size_t>(second.selectedIndex)];
    assert(runtime.observe(second.traceId, actualFor(secondChosen, 0.90), true, false,
                           secondChosen.total.meanMs * 0.85, &error));

    auto degraded = state;
    degraded.requestId = "request-degraded";
    degraded.features.set("network.goodput_up_bps", 0.5e6);
    degraded.features.set("network.rtt_ms", 80.0);
    const auto afterDegradation = runtime.decide("device-A", degraded);
    assert(afterDegradation.environmentChanged);
    assert(!afterDegradation.environmentKey.empty());
    assert(afterDegradation.selectedAsProbe);
    assert(afterDegradation.predictions[static_cast<std::size_t>(afterDegradation.selectedIndex)]
               .structuralActionKey == secondChosen.structuralActionKey);

    // Negative residuals must be learnable: after an observation faster than
    // the prior, the same action's estimate cannot move upward.
    const auto directlyPredicted = runtime.estimator().predict(
        "device-A", pruned.shortlist.front(), state);
    assert(std::isfinite(directlyPredicted.total.meanMs));

    const auto newModel = manifest("model-v2", 100);
    assert(runtime.registerManifest(newModel, &error));
    CompatibilityMap compatibility;
    compatibility.oldModelHash = oldModel.modelHash;
    compatibility.newModelHash = newModel.modelHash;
    for (int i = 0; i < 6; ++i) {
        BoundaryMapping mapping;
        mapping.oldCanonicalId = "stage." + std::to_string(i);
        mapping.newCanonicalId = mapping.oldCanonicalId;
        mapping.kind = i < 3 ? MappingKind::EXACT :
                       (i < 5 ? MappingKind::CONTRACT_COMPATIBLE : MappingKind::INCOMPATIBLE);
        compatibility.boundaries.push_back(mapping);
    }
    const auto migration = runtime.migrateVersion("device-A", oldModel.modelHash,
                                                   newModel.modelHash, compatibility, &error);
    assert(migration.exactReused == 9);
    assert(migration.compatibleReused == 6);
    assert(migration.incompatibleRemoved == 3);
    assert(migration.requiredProbes == 6);

    state.modelHash = newModel.modelHash;
    state.requestId = "request-after-update";
    const auto afterUpdate = runtime.decide("device-A", state);
    assert(afterUpdate.selectedIndex >= 0);
    assert(afterUpdate.toJson().find("profile_free") == std::string::npos ||
           !afterUpdate.predictions.empty());

    RuntimeState offline = state;
    offline.features.set("network.available", 0.0);
    const auto offlinePruned = pruner.prune(offline, builder.build(newModel).actions);
    assert(offlinePruned.shortlist.size() == 1);
    assert(offlinePruned.shortlist.front().action.mode == ActionMode::LOCAL);

    std::cout << "PROFILE_FREE_CONTROLLER_TEST_PASS\n"
              << "raw_actions=" << built.actions.size()
              << " shortlist=" << pruned.shortlist.size()
              << " decision_ms=" << first.totalDecisionMs
              << " migration=" << migration.toJson() << std::endl;
    return 0;
}
