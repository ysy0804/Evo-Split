#include "ProfileFreeController.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

StructuralModelManifest makeManifest(const std::string& version, int physicalOffset) {
    StructuralModelManifest model;
    model.manifestId = "efficientnet-b0-" + version;
    model.modelName = "EfficientNet-B0";
    model.modelHash = "efficientnet-b0-" + version;
    model.modelVersion = version;
    model.structureHash = "efficientnet-b0-structure-" + version;
    model.inputBytes = 224u * 224u * 3u * 4u;
    model.totalFlops = 780.0e6;
    model.localPeakMemoryMb = 220.0;

    const std::vector<uint64_t> bytes = {2'408'448, 1'204'224, 602'112, 301'056, 125'440};
    const std::vector<std::string> ids = {
        "blocks.1.output", "blocks.2.output", "blocks.3.output",
        "blocks.4.output", "head.output"};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        StructuralBoundaryDescriptor boundary;
        boundary.canonicalBoundaryId = ids[i];
        boundary.physicalBoundary = physicalOffset + 20 + static_cast<int>(i) * 12;
        boundary.prefixFlops = model.totalFlops * (0.16 + 0.16 * static_cast<double>(i));
        boundary.suffixFlops = model.totalFlops - boundary.prefixFlops;
        boundary.rawBundleBytes = bytes[i];
        boundary.peakEdgeMemoryMb = 90.0 + 18.0 * static_cast<double>(i);
        boundary.boundaryDepth = static_cast<int>(i);
        boundary.cutSetTensorCount = (i == 2 ? 3 : 1);
        boundary.supportedCodecs = {CodecKind::RAW32, CodecKind::FP16, CodecKind::BLOCK_INT8};
        boundary.compressionRatios[CodecKind::RAW32] = 1.0;
        boundary.compressionRatios[CodecKind::FP16] = 0.5;
        boundary.compressionRatios[CodecKind::BLOCK_INT8] = 0.27;
        boundary.accuracyDrops[CodecKind::RAW32] = 0.0;
        boundary.accuracyDrops[CodecKind::FP16] = 0.001;
        boundary.accuracyDrops[CodecKind::BLOCK_INT8] = 0.006;
        model.boundaries.push_back(boundary);
    }
    return model;
}

RuntimeState makeState(const StructuralModelManifest& model, int request,
                       double goodputMbps, double rttMs) {
    RuntimeState state;
    state.requestId = "request-" + std::to_string(request);
    state.modelName = model.modelName;
    state.modelHash = model.modelHash;
    state.modelVersion = model.modelVersion;
    state.features.set("network.available", 1.0);
    state.features.set("cloud.available", 1.0);
    state.features.set("network.goodput_up_bps", goodputMbps * 1.0e6);
    state.features.set("network.rtt_ms", rttMs);
    state.features.set("network.jitter_ms", static_cast<double>(request % 4));
    state.features.set("device.memory.free_mb", 512.0);
    state.features.set("device.compute.flops_per_ms", 4.8e6);
    state.features.set("cloud.compute.flops_per_ms", 28.0e6);
    return state;
}

double encodedBytes(const StructuralActionDescriptor& action) {
    return static_cast<double>(action.estimatedEncodedBytes > 0
        ? action.estimatedEncodedBytes : action.rawBundleBytes);
}

MetricBag measure(const StructuralActionDescriptor& action,
                  const StructuralModelManifest& model,
                  double goodputMbps, double rttMs, int request) {
    const double edgeRate = 4.5e6 * (1.0 + 0.025 * std::sin(request * 0.7));
    const double cloudRate = 30.0e6;
    const double bps = std::max(0.1, goodputMbps) * 1.0e6;
    double edge = 0.0, capture = 0.0, encode = 0.0, transfer = 0.0;
    double decode = 0.0, inject = 0.0, cloud = 0.0;

    if (action.action.mode == ActionMode::LOCAL) {
        edge = model.totalFlops / edgeRate;
    } else if (action.action.mode == ActionMode::CLOUD) {
        transfer = static_cast<double>(model.inputBytes) * 8.0 * 1000.0 / bps + rttMs;
        cloud = model.totalFlops / cloudRate;
    } else {
        edge = action.prefixFlops / edgeRate;
        capture = static_cast<double>(action.rawBundleBytes) / 35.0e6;
        if (action.action.codec != CodecKind::RAW32) {
            encode = static_cast<double>(action.rawBundleBytes) / 8.0e6;
            decode = encodedBytes(action) / 18.0e6;
        }
        transfer = encodedBytes(action) * 8.0 * 1000.0 / bps + rttMs;
        inject = static_cast<double>(action.rawBundleBytes) / 42.0e6;
        cloud = action.suffixFlops / cloudRate;
    }

    MetricBag actual;
    actual.set("latency.edge_prefix_ms", edge);
    actual.set("latency.capture_ms", capture);
    actual.set("latency.encode_ms", encode);
    actual.set("latency.transfer_ms", transfer);
    actual.set("latency.decode_ms", decode);
    actual.set("latency.inject_ms", inject);
    actual.set("latency.cloud_queue_ms", 0.0);
    actual.set("latency.cloud_suffix_ms", cloud);
    actual.set("latency.total_ms", edge + capture + encode + transfer + decode + inject + cloud);
    return actual;
}

double totalMs(const MetricBag& bag) {
    double value = 0.0;
    bag.getNumber("latency.total_ms", value);
    return value;
}

CompatibilityMap compatibility(const StructuralModelManifest& oldModel,
                               const StructuralModelManifest& newModel) {
    CompatibilityMap result;
    result.oldModelHash = oldModel.modelHash;
    result.newModelHash = newModel.modelHash;
    for (std::size_t i = 0; i < oldModel.boundaries.size(); ++i) {
        BoundaryMapping item;
        item.oldCanonicalId = oldModel.boundaries[i].canonicalBoundaryId;
        item.newCanonicalId = newModel.boundaries[i].canonicalBoundaryId;
        item.oldPhysicalId = std::to_string(oldModel.boundaries[i].physicalBoundary);
        item.newPhysicalId = std::to_string(newModel.boundaries[i].physicalBoundary);
        item.kind = i < 3 ? MappingKind::EXACT
                          : (i == 3 ? MappingKind::CONTRACT_COMPATIBLE
                                    : MappingKind::INCOMPATIBLE);
        item.score = i < 3 ? 1.0 : (i == 3 ? 0.9 : 0.0);
        result.boundaries.push_back(item);
    }
    return result;
}

std::string modeName(const SplitAction& action) {
    if (action.mode == ActionMode::LOCAL) return "local";
    if (action.mode == ActionMode::CLOUD) return "cloud";
    return "split:b" + std::to_string(action.boundary) + ":" + codecKindName(action.codec);
}

}  // namespace

int main(int argc, char** argv) {
    std::string output = "demo_trace.csv";
    if (argc == 3 && std::string(argv[1]) == "--output") output = argv[2];

    const auto modelV1 = makeManifest("v1", 0);
    const auto modelV2 = makeManifest("v2", 100);
    StructuralActionBuilder builder;
    const auto actionsV1 = builder.build(modelV1).actions;
    const auto actionsV2 = builder.build(modelV2).actions;

    ProfileFreeControllerConfig config;
    config.bootstrapMode = ControllerBootstrapMode::PROFILE_FREE;
    config.shadowMode = false;
    config.pruning.maximumShortlistSize = 6;
    config.pruning.lowerBoundSlack = 0.35;
    config.adaptation.maximumProbeRequests = 4;
    config.adaptation.changeProbeRequests = 2;

    auto evo = std::make_unique<ProfileFreeRuntime>(config);
    auto cold = std::make_unique<ProfileFreeRuntime>(config);
    std::string error;
    if (!evo->registerManifest(modelV1, &error) || !cold->registerManifest(modelV1, &error)) {
        std::cerr << "Failed to register demo manifest: " << error << '\n';
        return 1;
    }

    std::ofstream csv(output);
    if (!csv) {
        std::cerr << "Cannot open output file: " << output << '\n';
        return 2;
    }
    csv << "method,request,phase,model_version,goodput_mbps,rtt_ms,action,actual_ms,"
           "oracle_ms,regret_ms,probe,candidates,update_event,reused,reset,removed\n";

    ProfileFreeMigrationResult migration;
    double evoRegret = 0.0, coldRegret = 0.0;
    for (int request = 1; request <= 30; ++request) {
        const bool updated = request >= 21;
        const auto& model = updated ? modelV2 : modelV1;
        const auto& actions = updated ? actionsV2 : actionsV1;
        const double goodput = request <= 10 ? 30.0 : (request <= 20 ? 2.0 : 15.0);
        const double rtt = request <= 10 ? 15.0 : (request <= 20 ? 80.0 : 25.0);
        const std::string phase = request <= 10 ? "normal" : (request <= 20 ? "degraded" : "updated");

        if (request == 21) {
            if (!evo->registerManifest(modelV2, &error)) return 3;
            migration = evo->migrateVersion("demo-device", modelV1.modelHash,
                                            modelV2.modelHash,
                                            compatibility(modelV1, modelV2), &error);
            cold = std::make_unique<ProfileFreeRuntime>(config);
            if (!cold->registerManifest(modelV2, &error)) return 4;
        }

        auto state = makeState(model, request, goodput, rtt);
        double oracle = 1.0e30;
        for (const auto& action : actions) oracle = std::min(oracle, totalMs(measure(action, model, goodput, rtt, request)));

        for (const std::string method : {"Evo-Split", "Cold-Restart"}) {
            ProfileFreeRuntime* runtime = method == "Evo-Split" ? evo.get() : cold.get();
            auto decision = runtime->decide("demo-device", state);
            if (decision.selectedIndex < 0) return 5;
            const auto& selected = decision.predictions[static_cast<std::size_t>(decision.selectedIndex)];
            const auto descriptorIt = std::find_if(actions.begin(), actions.end(), [&](const auto& item) {
                return item.stableKey() == selected.structuralActionKey;
            });
            if (descriptorIt == actions.end()) return 6;
            const auto actual = measure(*descriptorIt, model, goodput, rtt, request);
            const double latency = totalMs(actual);
            const double regret = std::max(0.0, latency - oracle);
            if (method == "Evo-Split") evoRegret += regret; else coldRegret += regret;
            if (!runtime->observe(decision.traceId, actual, true, false, oracle, &error)) return 7;

            csv << method << ',' << request << ',' << phase << ',' << model.modelVersion << ','
                << goodput << ',' << rtt << ',' << modeName(descriptorIt->action) << ','
                << std::fixed << std::setprecision(3) << latency << ',' << oracle << ',' << regret << ','
                << (decision.selectedAsProbe ? 1 : 0) << ',' << decision.predictions.size() << ','
                << (request == 21 ? 1 : 0) << ','
                << (method == "Evo-Split" && request == 21
                        ? migration.exactReused + migration.compatibleReused : 0) << ','
                << (method == "Evo-Split" && request == 21 ? migration.calibrationReset : 0) << ','
                << (method == "Evo-Split" && request == 21 ? migration.incompatibleRemoved : 0) << '\n';
        }
    }

    std::cout << "EVOSPLIT_DEMO=PASS\n"
              << "raw_actions=" << actionsV1.size() << " shortlist_limit="
              << config.pruning.maximumShortlistSize << '\n'
              << "update_reused=" << migration.exactReused + migration.compatibleReused
              << " update_reset=" << migration.calibrationReset
              << " update_removed=" << migration.incompatibleRemoved << '\n'
              << "cumulative_regret_evosplit_ms=" << std::fixed << std::setprecision(3)
              << evoRegret << " cumulative_regret_cold_restart_ms=" << coldRegret << '\n'
              << "trace=" << output << '\n';
    return 0;
}

