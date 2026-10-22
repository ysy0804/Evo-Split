#include "AdaptiveRuntime.h"

#include <cmath>
#include <iostream>
#include <memory>

namespace {

ComponentProfile profile(ActionMode mode, int boundary, CodecKind codec,
                         double edge, double cloud, int64_t bytes, double accuracy = 0.0) {
    ComponentProfile value;
    value.action.mode = mode;
    value.action.boundary = boundary;
    value.action.codec = codec;
    value.action.modelHash = "hash-v1";
    value.action.manifestId = "manifest-v1";
    value.edgePrefixMs = edge;
    value.captureMs = mode == ActionMode::SPLIT ? 0.4 : 0.0;
    value.encodeMs = mode == ActionMode::SPLIT ? 0.8 : 0.0;
    value.decodeMs = mode == ActionMode::SPLIT ? 0.3 : 0.0;
    value.injectMs = mode == ActionMode::SPLIT ? 0.2 : 0.0;
    value.cloudSuffixMs = cloud;
    value.bundleBytes = bytes;
    value.rawBytes = bytes * 4;
    value.encodedBytes = bytes;
    value.offlineP95PaddingMs = 2.0;
    value.accuracyDrop = accuracy;
    value.peakMemoryMb = 300.0;
    value.energyMj = edge * 2.0;
    value.failureProbability = 0.01;
    return value;
}

OptimizationSpec spec() {
    OptimizationSpec value;
    value.id = "runtime_test";
    value.objectives.push_back({"latency.p95_ms", 1.0, 1.0, ObjectiveDirection::MINIMIZE});
    value.constraints.push_back({"safe", "safety.allowed", ConstraintOperator::GREATER_EQUAL,
                                 1.0, 0.0, true, 0.0});
    return value;
}

}  // namespace

int main() {
    PerformanceMetrics metrics{};
    metrics.cpuUsagePercent = 67.0;
    metrics.cpuFrequencyMHz = 1800.0;
    metrics.memoryUsageMB = 120.0;
    metrics.memoryPeakMB = 180.0;
    metrics.socTempCelsius = 48.0;
    metrics.powerConsumptionMw = 2300.0;
    NetworkRuntimeSample network{10.0, 10.0e6, 2.0, 0.01, "wifi", true};
    CloudRuntimeSample cloud{1.5, 30.0, true};
    RuntimeHistorySample history{1.0, 0, "local:mhash-v1"};
    RuntimeState state = RuntimeStateAdapter::build(
        metrics, network, cloud, history, "r1", "MobileNetV2", "hash-v1", "v1");
    double cpu = 0.0, available = 0.0;
    if (!state.features.getNumber("device.cpu.util_pct", cpu) || cpu != 67.0 ||
        !state.features.getNumber("network.available", available) || available != 1.0) return 1;

    ProfileManifest manifest;
    manifest.manifestId = "manifest-v1";
    manifest.modelHash = "hash-v1";
    manifest.modelName = "MobileNetV2";
    manifest.modelVersion = "v1";
    manifest.profiles = {
        profile(ActionMode::LOCAL, -1, CodecKind::RAW32, 70.0, 0.0, 0),
        profile(ActionMode::CLOUD, -1, CodecKind::RAW32, 0.0, 20.0, 150528),
        profile(ActionMode::SPLIT, 11, CodecKind::BLOCK_INT8, 12.0, 8.0, 76566, 0.01),
        profile(ActionMode::SPLIT, 11, CodecKind::BLOCK_INT4, 12.0, 8.0, 42462, 0.08),
        profile(ActionMode::SPLIT, 11, CodecKind::RAW32, 12.0, 8.0, 301154, 0.0),
    };
    // Verify that the explainable profile model really reacts to runtime state,
    // instead of merely replaying a constant offline latency.
    manifest.profiles[2].referenceCpuFrequencyMhz = 1800.0;
    manifest.profiles[2].cpuUtilSensitivity = 0.5;
    manifest.profiles[2].referenceCloudUtilPct = 20.0;
    manifest.profiles[2].cloudUtilSensitivity = 1.0;
    OfflineProfileStore store;
    std::string error;
    if (!store.replace(manifest, &error) || store.profiles("hash-v1").size() != 5) return 2;
    ProfileManifest bad = manifest;
    bad.profiles.push_back(bad.profiles.front());
    if (store.replace(bad, &error)) return 3;

    ComponentCostPredictor base;
    auto candidates = base.predictAll(manifest.profiles, state);
    double transfer = 0.0, total = 0.0, scaledEdge = 0.0, scaledSuffix = 0.0;
    if (!candidates[2].predicted.getNumber("latency.transfer_ms", transfer) ||
        !candidates[2].predicted.getNumber("latency.total_mean_ms", total) ||
        !candidates[2].predicted.getNumber("latency.edge_prefix_ms", scaledEdge) ||
        !candidates[2].predicted.getNumber("latency.cloud_suffix_ms", scaledSuffix) ||
        std::fabs(transfer - (10.0 + 2.0 + 76566.0 / 10000.0 * 1.02)) > 1e-6 ||
        std::fabs(scaledEdge - 12.0 * (1.0 + 0.5 * 0.67)) > 1e-6 ||
        std::fabs(scaledSuffix - 8.0 * 1.1) > 1e-6 || total <= transfer) return 4;

    auto residuals = std::make_shared<OnlineResidualModel>(ResidualConfig{0.15, 0.35, 64, 16, 0.5});
    OnlineCorrectedPredictor online(base, residuals);
    ActionCandidate initial = online.predict(manifest.profiles[2], state);
    double initialP95 = 0.0;
    initial.predicted.getNumber("latency.p95_ms", initialP95);
    for (int i = 0; i < 20; ++i) {
        MetricBag actual;
        for (const std::string component : {"latency.edge_prefix_ms", "latency.capture_ms",
                 "latency.encode_ms", "latency.transfer_ms", "latency.decode_ms",
                 "latency.inject_ms", "latency.cloud_queue_ms", "latency.cloud_suffix_ms"}) {
            double predicted = 0.0;
            initial.predicted.getNumber(component, predicted);
            actual.set(component, predicted + (component == "latency.transfer_ms" ? 8.0 : 1.0));
        }
        online.observe(initial, actual, i == 0);
    }
    ActionCandidate corrected = online.predict(manifest.profiles[2], state);
    double correctedMean = 0.0, correctedP95 = 0.0;
    corrected.predicted.getNumber("latency.total_mean_ms", correctedMean);
    corrected.predicted.getNumber("latency.p95_ms", correctedP95);
    if (correctedMean <= total + 10.0 || correctedP95 < correctedMean ||
        residuals->snapshot(corrected.action.key(), "latency.transfer_ms").samples != 20) return 5;
    SplitAction migratedAction = corrected.action;
    migratedAction.modelHash = "hash-v2";
    residuals->migrateAction(corrected.action.key(), migratedAction.key(), 0.5, 0.0, 1.0);
    if (residuals->snapshot(migratedAction.key(), "latency.transfer_ms").samples != 20 ||
        residuals->snapshot(migratedAction.key(), "latency.encode_ms").samples != 0) return 13;

    SafetyConfig safetyConfig;
    safetyConfig.maxAccuracyDrop = 0.02;
    safetyConfig.maxPeakMemoryMb = 512.0;
    safetyConfig.hardSocTemperatureC = 80.0;
    SafetyFilter safety(safetyConfig);
    safety.annotate(state, candidates);
    double int8Safe = 0.0, int4Safe = 1.0;
    candidates[2].predicted.getNumber("safety.allowed", int8Safe);
    candidates[3].predicted.getNumber("safety.allowed", int4Safe);
    if (int8Safe != 1.0 || int4Safe != 0.0) return 6;

    auto evaluator = std::make_shared<ConfigurableOptimizationEvaluator>(spec());
    HysteresisConfig hysteresisConfig;
    hysteresisConfig.minimumAbsoluteGain = 5.0;
    hysteresisConfig.minimumRelativeGain = 0.0;
    hysteresisConfig.requiredConsecutiveWins = 2;
    hysteresisConfig.minimumDwellRequests = 2;
    hysteresisConfig.cooldownRequests = 0;
    auto hysteresis = std::make_unique<HysteresisPolicy>(hysteresisConfig);
    auto sink = std::make_shared<MemoryDecisionTraceSink>();
    AdaptiveSplitController controller(std::move(hysteresis), evaluator, sink);
    PolicyContext context;
    context.state = state;
    context.candidates = candidates;
    // Force local to be the initial best.
    context.candidates[0].predicted.set("latency.p95_ms", 20.0);
    context.candidates[2].predicted.set("latency.p95_ms", 40.0);
    DecisionTrace first = controller.decide(context, true);
    if (first.selectedIndex != 0) return 7;
    context.candidates[2].predicted.set("latency.p95_ms", 10.0);
    DecisionTrace hold = controller.decide(context, true);
    DecisionTrace switched = controller.decide(context, true);
    if (hold.selectedIndex != 0 || switched.selectedIndex != 2 ||
        switched.reason != "hysteresis_threshold_met") return 8;

    FallbackStateMachine fallback;
    fallback.onEvent(FallbackEvent::ACTION_FAILURE);
    if (fallback.state() != FallbackState::DEGRADED || fallback.selectFallback(candidates, 3) != 2) return 9;
    fallback.onEvent(FallbackEvent::ACTION_FAILURE);
    if (fallback.state() != FallbackState::LOCAL_SAFE || fallback.selectFallback(candidates, 2) != 0) return 10;
    fallback.onEvent(FallbackEvent::NETWORK_DOWN);
    if (fallback.stateName() != "local_safe") return 11;
    fallback.onEvent(FallbackEvent::SUCCESS);
    fallback.onEvent(FallbackEvent::SUCCESS);
    fallback.onEvent(FallbackEvent::SUCCESS);
    if (fallback.state() != FallbackState::NORMAL) return 12;

    std::cout << "ADAPTIVE_RUNTIME_TEST=PASS base_ms=" << total
              << " corrected_mean_ms=" << correctedMean
              << " corrected_p95_ms=" << correctedP95
              << " switch_reason=" << switched.reason << std::endl;
    return 0;
}
