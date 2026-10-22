#include "ProfileFreeController.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
double numberOr(const MetricBag& bag, const std::string& key, double fallback) {
    double value = fallback;
    return bag.getNumber(key, value) ? value : fallback;
}

double codecRatio(CodecKind codec) {
    switch (codec) {
        case CodecKind::RAW32: return 1.0;
        case CodecKind::FP16: return 2.0;
        case CodecKind::BLOCK_INT8: return 4.0;
        case CodecKind::BLOCK_INT6: return 16.0 / 3.0;
        case CodecKind::BLOCK_INT4: return 8.0;
        case CodecKind::LZ4: return 1.5;
    }
    return 1.0;
}

const char* profileCodecName(CodecKind codec) {
    switch (codec) {
        case CodecKind::RAW32: return "raw32";
        case CodecKind::FP16: return "fp16";
        case CodecKind::BLOCK_INT8: return "int8";
        case CodecKind::BLOCK_INT6: return "int6";
        case CodecKind::BLOCK_INT4: return "int4";
        case CodecKind::LZ4: return "lz4";
    }
    return "unknown";
}

uint64_t steadyNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

double elapsedMs(uint64_t start) {
    return (steadyNs() - start) / 1.0e6;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else out << c;
    }
    return out.str();
}

double totalActual(const MetricBag& actual) {
    double total = 0.0;
    if (actual.getNumber("latency.total_ms", total) ||
        actual.getNumber("latency.total_mean_ms", total)) return total;
    total = 0.0;
    for (const char* component : {"latency.edge_prefix_ms", "latency.capture_ms",
             "latency.encode_ms", "latency.transfer_ms", "latency.decode_ms",
             "latency.inject_ms", "latency.cloud_queue_ms", "latency.cloud_suffix_ms"}) {
        total += numberOr(actual, component, 0.0);
    }
    return total;
}

bool remoteAvailable(const RuntimeState& state) {
    return numberOr(state.features, "network.available", 0.0) >= 0.5 &&
           numberOr(state.features, "cloud.available", 0.0) >= 0.5;
}

std::string residualContextKey(const StructuralActionDescriptor& action,
                               const RuntimeState& state) {
    const std::string base = action.stableKey();
    if (action.action.mode == ActionMode::LOCAL) return base;
    const double goodput = std::max(1.0, numberOr(state.features, "network.goodput_up_bps", 1.0));
    const double rtt = std::max(0.0, numberOr(state.features, "network.rtt_ms", 0.0));
    const double loss = std::max(0.0, numberOr(state.features, "network.loss_rate", 0.0));
    std::ostringstream out;
    out << base << "|ctx:" << static_cast<long long>(std::llround(std::log2(goodput)))
        << ':' << static_cast<long long>(std::llround(rtt / 10.0))
        << ':' << static_cast<long long>(std::llround(loss * 1000.0));
    return out.str();
}

std::string runtimeEnvironmentKey(const RuntimeState& state) {
    const double goodput = std::max(1.0, numberOr(
        state.features, "network.goodput_up_bps", 1.0));
    const double rtt = std::max(0.0, numberOr(state.features, "network.rtt_ms", 0.0));
    const double loss = std::max(0.0, numberOr(state.features, "network.loss_rate", 0.0));
    std::ostringstream out;
    out << static_cast<long long>(std::llround(std::log2(goodput))) << ':'
        << static_cast<long long>(std::llround(rtt / 10.0)) << ':'
        << static_cast<long long>(std::llround(loss * 1000.0));
    return out.str();
}

double runtimeEnvironmentQuality(const RuntimeState& state) {
    const double goodput = std::max(1.0, numberOr(
        state.features, "network.goodput_up_bps", 1.0));
    const double rtt = std::max(0.0, numberOr(state.features, "network.rtt_ms", 0.0));
    const double loss = std::max(0.0, numberOr(state.features, "network.loss_rate", 0.0));
    return std::log2(goodput) - rtt / 50.0 - 50.0 * loss;
}
}

const char* controllerBootstrapModeName(ControllerBootstrapMode value) {
    switch (value) {
        case ControllerBootstrapMode::OFFLINE_PROFILE: return "offline_profile";
        case ControllerBootstrapMode::PROFILE_FREE: return "profile_free";
        case ControllerBootstrapMode::HYBRID: return "hybrid";
    }
    return "unknown";
}

const char* profileSourceName(ProfileSource value) {
    switch (value) {
        case ProfileSource::NONE: return "none";
        case ProfileSource::OPTIONAL_OFFLINE: return "optional_offline";
        case ProfileSource::INHERITED_VERSION: return "inherited_version";
        case ProfileSource::ONLINE_OBSERVED: return "online_observed";
    }
    return "unknown";
}

const char* adaptationPhaseName(AdaptationPhase value) {
    switch (value) {
        case AdaptationPhase::UNINITIALIZED: return "uninitialized";
        case AdaptationPhase::ANCHORING: return "anchoring";
        case AdaptationPhase::SAFE_EXPLORING: return "safe_exploring";
        case AdaptationPhase::ADAPTING: return "adapting";
        case AdaptationPhase::STABLE: return "stable";
        case AdaptationPhase::DRIFTED: return "drifted";
    }
    return "unknown";
}

bool parseControllerBootstrapMode(const std::string& text, ControllerBootstrapMode& value) {
    if (text == "offline_profile") value = ControllerBootstrapMode::OFFLINE_PROFILE;
    else if (text == "profile_free") value = ControllerBootstrapMode::PROFILE_FREE;
    else if (text == "hybrid") value = ControllerBootstrapMode::HYBRID;
    else return false;
    return true;
}

bool StructuralModelManifest::valid(std::string* error) const {
    if (manifestId.empty() || modelHash.empty() || structureHash.empty()) {
        if (error) *error = "manifest identity is incomplete";
        return false;
    }
    if (totalFlops <= 0.0 || inputBytes == 0) {
        if (error) *error = "model FLOPs and input bytes must be positive";
        return false;
    }
    std::set<std::string> ids;
    std::set<int> physical;
    for (const auto& boundary : boundaries) {
        if (boundary.canonicalBoundaryId.empty() || boundary.physicalBoundary < 0 ||
            boundary.prefixFlops < 0.0 || boundary.suffixFlops < 0.0 ||
            boundary.rawBundleBytes == 0 || boundary.supportedCodecs.empty()) {
            if (error) *error = "invalid structural boundary";
            return false;
        }
        if (!ids.insert(boundary.canonicalBoundaryId).second ||
            !physical.insert(boundary.physicalBoundary).second) {
            if (error) *error = "duplicate boundary identity";
            return false;
        }
    }
    return true;
}

std::string StructuralActionDescriptor::stableKey() const {
    std::ostringstream out;
    out << action.modelHash << '|';
    if (action.mode == ActionMode::LOCAL) out << "local";
    else if (action.mode == ActionMode::CLOUD) out << "cloud";
    else out << canonicalBoundaryId << '|' << static_cast<uint32_t>(action.codec);
    return out.str();
}

ActionBuildResult StructuralActionBuilder::build(const StructuralModelManifest& manifest) const {
    const uint64_t start = steadyNs();
    ActionBuildResult result;
    StructuralActionDescriptor local;
    local.action.mode = ActionMode::LOCAL;
    local.action.modelHash = manifest.modelHash;
    local.action.manifestId = manifest.manifestId;
    local.prefixFlops = manifest.totalFlops;
    local.peakEdgeMemoryMb = manifest.localPeakMemoryMb;
    local.versionQualified = true;
    result.actions.push_back(local);

    StructuralActionDescriptor cloud;
    cloud.action.mode = ActionMode::CLOUD;
    cloud.action.modelHash = manifest.modelHash;
    cloud.action.manifestId = manifest.manifestId;
    cloud.suffixFlops = manifest.totalFlops;
    cloud.rawBundleBytes = manifest.inputBytes;
    cloud.estimatedEncodedBytes = manifest.inputBytes;
    cloud.versionQualified = true;
    result.actions.push_back(cloud);

    result.rawActions = 2;
    for (const auto& boundary : manifest.boundaries) {
        result.rawActions += boundary.supportedCodecs.size();
        if (!boundary.versionQualified || !boundary.structurallyCompatible) {
            result.rejectedVersion += boundary.supportedCodecs.size();
            continue;
        }
        for (CodecKind codec : boundary.supportedCodecs) {
            StructuralActionDescriptor action;
            action.action.mode = ActionMode::SPLIT;
            action.action.boundary = boundary.physicalBoundary;
            action.action.codec = codec;
            action.action.modelHash = manifest.modelHash;
            action.action.manifestId = manifest.manifestId;
            action.canonicalBoundaryId = boundary.canonicalBoundaryId;
            action.prefixFlops = boundary.prefixFlops;
            action.suffixFlops = boundary.suffixFlops;
            action.rawBundleBytes = boundary.rawBundleBytes;
            double ratio = codecRatio(codec);
            const auto ratioIt = boundary.compressionRatios.find(codec);
            if (ratioIt != boundary.compressionRatios.end() && ratioIt->second > 0.0) ratio = ratioIt->second;
            action.estimatedEncodedBytes = static_cast<uint64_t>(std::ceil(boundary.rawBundleBytes / ratio));
            action.peakEdgeMemoryMb = boundary.peakEdgeMemoryMb;
            action.boundaryDepth = boundary.boundaryDepth;
            action.cutSetTensorCount = boundary.cutSetTensorCount;
            const auto accuracy = boundary.accuracyDrops.find(codec);
            action.estimatedAccuracyDrop = accuracy == boundary.accuracyDrops.end()
                ? (codec == CodecKind::RAW32 || codec == CodecKind::LZ4 ? 0.0 : 0.005)
                : accuracy->second;
            action.versionQualified = boundary.versionQualified;
            action.structurallyCompatible = boundary.structurallyCompatible;
            action.codecQualified = true;
            result.actions.push_back(action);
        }
    }
    result.buildMs = elapsedMs(start);
    return result;
}

CandidatePruningResult CandidatePruner::prune(
    const RuntimeState& state, const std::vector<StructuralActionDescriptor>& actions,
    const std::string& previousSafeActionKey) const {
    const uint64_t start = steadyNs();
    CandidatePruningResult result;
    result.inputActions = actions.size();
    const double memoryFree = numberOr(state.features, "device.memory.free_mb", 1.0e12);
    const double goodput = numberOr(state.features, "network.goodput_up_bps", 0.0);
    const double rtt = numberOr(state.features, "network.rtt_ms", 0.0);
    const double edgeRate = std::max(1.0, numberOr(state.features, "device.compute.flops_per_ms", 1.0e6));
    const double cloudRate = std::max(1.0, numberOr(state.features, "cloud.compute.flops_per_ms", 1.0e7));
    const bool remote = remoteAvailable(state);

    struct ViableAction { std::size_t index; double lowerBoundMs; bool forced; };
    std::vector<ViableAction> viable;
    viable.reserve(actions.size());
    for (std::size_t index = 0; index < actions.size(); ++index) {
        const auto& action = actions[index];
        const char* rejection = nullptr;
        if (!action.versionQualified || !action.structurallyCompatible) rejection = "version_or_structure";
        else if (!action.codecQualified) rejection = "codec_unqualified";
        else if (action.estimatedAccuracyDrop > config_.maximumAccuracyDrop) rejection = "accuracy_limit";
        else if (action.peakEdgeMemoryMb > memoryFree) rejection = "memory_limit";
        else if (action.action.mode != ActionMode::LOCAL && !remote) rejection = "remote_unavailable";
        if (rejection) {
            ++result.hardRejected;
            if (config_.retainAuditRecords) {
                PrunedActionRecord record;
                record.descriptor = action;
                record.reason = rejection;
                result.records.push_back(std::move(record));
            }
            continue;
        }
        double lower = action.prefixFlops / edgeRate + action.suffixFlops / cloudRate;
        if (action.action.mode != ActionMode::LOCAL) {
            lower += rtt;
            lower += goodput > 0.0 ? action.estimatedEncodedBytes * 1000.0 / goodput : 1.0e12;
        }
        bool forced = action.action.mode != ActionMode::SPLIT;
        if (!previousSafeActionKey.empty() &&
            (action.action.key() == previousSafeActionKey ||
             action.stableKey() == previousSafeActionKey)) forced = true;
        viable.push_back({index, lower, forced});
    }

    double best = std::numeric_limits<double>::infinity();
    for (const auto& record : viable) best = std::min(best, record.lowerBoundMs);
    int minimumDepth = std::numeric_limits<int>::max(), maximumDepth = std::numeric_limits<int>::min();
    for (const auto& record : viable) {
        const auto& action = actions[record.index];
        if (action.action.mode == ActionMode::SPLIT) {
            minimumDepth = std::min(minimumDepth, action.boundaryDepth);
            maximumDepth = std::max(maximumDepth, action.boundaryDepth);
        }
    }
    if (config_.structuralPruningEnabled && config_.keepRepresentativeBoundaries &&
        minimumDepth <= maximumDepth) {
        const int middle = minimumDepth + (maximumDepth - minimumDepth) / 2;
        for (int target : {minimumDepth, middle, maximumDepth}) {
            ViableAction* nearest = nullptr;
            for (auto& record : viable) {
                const auto& action = actions[record.index];
                if (action.action.mode != ActionMode::SPLIT) continue;
                if (!nearest || std::abs(action.boundaryDepth - target) <
                    std::abs(actions[nearest->index].boundaryDepth - target)) nearest = &record;
            }
            if (nearest) nearest->forced = true;
        }
    }
    std::sort(viable.begin(), viable.end(), [](const ViableAction& a, const ViableAction& b) {
        if (a.forced != b.forced) return a.forced;
        return a.lowerBoundMs < b.lowerBoundMs;
    });
    for (const auto& candidate : viable) {
        const bool mustKeep = candidate.forced;
        const bool withinBound = !config_.structuralPruningEnabled ||
            candidate.lowerBoundMs <= best * (1.0 + config_.lowerBoundSlack);
        bool kept = false;
        const char* reason = nullptr;
        if ((mustKeep || withinBound) &&
            (!config_.structuralPruningEnabled || mustKeep ||
             result.shortlist.size() < config_.maximumShortlistSize)) {
            kept = true;
            reason = !config_.structuralPruningEnabled ? "structural_pruning_disabled" :
                (mustKeep ? "forced_safe_or_representative" : "competitive_lower_bound");
            result.shortlist.push_back(actions[candidate.index]);
        } else {
            reason = result.shortlist.size() >= config_.maximumShortlistSize
                ? "shortlist_limit" : "dominated_lower_bound";
            ++result.lowerBoundRejected;
        }
        if (config_.retainAuditRecords) {
            PrunedActionRecord record;
            record.descriptor = actions[candidate.index];
            record.kept = kept;
            record.lowerBoundMs = candidate.lowerBoundMs;
            record.reason = reason;
            result.records.push_back(std::move(record));
        }
    }
    result.pruneMs = elapsedMs(start);
    return result;
}

void OnlineComponentEstimator::setConfig(const OnlineEstimatorConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

void OnlineComponentEstimator::updateSeries(ScalarSeries& series, double value, double alpha) {
    if (!std::isfinite(value) || value <= 0.0) return;
    if (series.samples == 0) {
        series.mean = value;
        series.variance = 0.0;
    } else {
        const double delta = value - series.mean;
        series.mean += alpha * delta;
        series.variance = (1.0 - alpha) * (series.variance + alpha * delta * delta);
    }
    ++series.samples;
}

void OnlineComponentEstimator::updateSignedSeries(ScalarSeries& series, double value, double alpha) {
    if (!std::isfinite(value)) return;
    if (series.samples == 0) {
        series.mean = value;
        series.variance = 0.0;
    } else {
        const double delta = value - series.mean;
        series.mean += alpha * delta;
        series.variance = (1.0 - alpha) * (series.variance + alpha * delta * delta);
    }
    ++series.samples;
}

double OnlineComponentEstimator::seriesMean(const ScalarSeries& series, double fallback) {
    return series.samples ? series.mean : fallback;
}

ComponentEstimate OnlineComponentEstimator::estimate(double mean, const ScalarSeries& series) const {
    ComponentEstimate result;
    result.meanMs = std::max(0.0, mean);
    result.samples = series.samples;
    result.coldStart = series.samples == 0;
    const double learned = series.samples > 1 ? std::sqrt(std::max(0.0, series.variance)) : 0.0;
    result.uncertaintyMs = std::max(config_.minimumUncertaintyMs,
        result.coldStart ? result.meanMs * config_.coldStartRelativeUncertainty : learned);
    result.p95Ms = result.meanMs + config_.p95SigmaMultiplier * result.uncertaintyMs;
    return result;
}

ProfileFreePrediction OnlineComponentEstimator::predict(
    const std::string& device, const StructuralActionDescriptor& action,
    const RuntimeState& state) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return predictUnlocked(device, action, state);
}

std::vector<ProfileFreePrediction> OnlineComponentEstimator::predictBatch(
    const std::string& device, const std::vector<StructuralActionDescriptor>& actions,
    const RuntimeState& state) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ProfileFreePrediction> predictions;
    predictions.reserve(actions.size());
    for (const auto& action : actions)
        predictions.push_back(predictUnlocked(device, action, state));
    return predictions;
}

ProfileFreePrediction OnlineComponentEstimator::predictUnlocked(
    const std::string& device, const StructuralActionDescriptor& action,
    const RuntimeState& state) const {
    const auto deviceIt = devices_.find(device);
    DeviceState empty;
    const DeviceState& learned = deviceIt == devices_.end() ? empty : deviceIt->second;
    ProfileFreePrediction result;
    result.structuralActionKey = action.stableKey();
    result.candidate.action = action.action;
    result.source = learned.actionSamples.count(action.stableKey())
        ? ProfileSource::ONLINE_OBSERVED : action.source;
    const double cpuUtil = std::max(0.0, numberOr(state.features, "device.cpu.util_pct", 0.0));
    const double thermal = std::max(0.0, numberOr(state.features, "device.thermal.soc_c", 25.0) - 60.0);
    const double edgeScale = (1.0 + 0.5 * cpuUtil / 100.0) * (1.0 + 0.01 * thermal);
    const double cloudUtil = std::max(0.0, numberOr(state.features, "cloud.util_pct", 0.0));
    const double edgeRate = seriesMean(learned.edgeFlopsPerMs, config_.initialEdgeFlopsPerMs);
    const double cloudRate = seriesMean(learned.cloudFlopsPerMs, config_.initialCloudFlopsPerMs);
    const double rawBytes = static_cast<double>(action.rawBundleBytes);
    const double encodedBytes = static_cast<double>(action.estimatedEncodedBytes);
    ScalarSeries emptySeries;
    auto codecSeries = [&](const std::map<CodecKind, ScalarSeries>& values) -> const ScalarSeries& {
        const auto found = values.find(action.action.codec);
        return found == values.end() ? emptySeries : found->second;
    };
    const ScalarSeries& encodeRateSeries = codecSeries(learned.encodeBytesPerMs);
    const ScalarSeries& decodeRateSeries = codecSeries(learned.decodeBytesPerMs);
    const double encodeRate = seriesMean(encodeRateSeries, config_.initialCodecBytesPerMs);
    const double decodeRate = seriesMean(decodeRateSeries, config_.initialCodecBytesPerMs);
    const double captureRate = seriesMean(learned.captureBytesPerMs, config_.initialCaptureBytesPerMs);
    const double injectRate = seriesMean(learned.injectBytesPerMs, config_.initialInjectBytesPerMs);
    const bool split = action.action.mode == ActionMode::SPLIT;
    const bool local = action.action.mode == ActionMode::LOCAL;
    const double edge = action.prefixFlops > 0.0 ? action.prefixFlops / edgeRate * edgeScale : 0.0;
    const double capture = split ? rawBytes / captureRate * edgeScale : 0.0;
    const double encode = split ? rawBytes / encodeRate * edgeScale : 0.0;
    double transfer = 0.0;
    if (!local) {
        const double goodput = numberOr(state.features, "network.goodput_up_bps", 0.0);
        const double rtt = numberOr(state.features, "network.rtt_ms", 0.0);
        const double jitter = numberOr(state.features, "network.jitter_ms", 0.0);
        const double loss = std::max(0.0, numberOr(state.features, "network.loss_rate", 0.0));
        transfer = remoteAvailable(state) && goodput > 0.0
            ? rtt + jitter + encodedBytes * 1000.0 / goodput * (1.0 + 2.0 * loss) : 1.0e12;
    }
    const double decode = split ? encodedBytes / decodeRate * (1.0 + cloudUtil / 200.0) : 0.0;
    const double inject = split ? rawBytes / injectRate * (1.0 + cloudUtil / 200.0) : 0.0;
    const double queue = local ? 0.0 : std::max(0.0, numberOr(state.features, "cloud.queue_ms", 0.0));
    const double cloud = action.suffixFlops > 0.0
        ? action.suffixFlops / cloudRate * (1.0 + cloudUtil / 100.0) : 0.0;

    double componentVariance = 0.0;
    auto put = [&](const std::string& key, double value, const ScalarSeries& series) {
        const ComponentEstimate component = estimate(value, series);
        componentVariance += component.uncertaintyMs * component.uncertaintyMs;
        if (config_.retainComponentDetails) {
            result.components[key] = component;
            result.candidate.predicted.set(key, value);
        }
    };
    put("latency.edge_prefix_ms", edge, learned.edgeFlopsPerMs);
    put("latency.capture_ms", capture, learned.captureBytesPerMs);
    put("latency.encode_ms", encode, encodeRateSeries);
    ScalarSeries networkSeries;
    networkSeries.samples = remoteAvailable(state) ? 1 : 0;
    networkSeries.mean = transfer;
    networkSeries.variance = std::pow(numberOr(state.features, "network.jitter_ms", 0.0), 2);
    put("latency.transfer_ms", transfer, networkSeries);
    put("latency.decode_ms", decode, decodeRateSeries);
    put("latency.inject_ms", inject, learned.injectBytesPerMs);
    put("latency.cloud_queue_ms", queue, networkSeries);
    put("latency.cloud_suffix_ms", cloud, learned.cloudFlopsPerMs);
    double total = edge + capture + encode + transfer + decode + inject + queue + cloud;
    ScalarSeries residual;
    const auto residualIt = learned.actionResidualMs.find(residualContextKey(action, state));
    if (residualIt != learned.actionResidualMs.end()) residual = residualIt->second;
    total = std::max(0.0, total + (residual.samples ? residual.mean : 0.0));
    result.total = estimate(total, residual);
    result.total.uncertaintyMs = std::max(result.total.uncertaintyMs, std::sqrt(componentVariance));
    result.total.p95Ms = result.total.meanMs + config_.p95SigmaMultiplier * result.total.uncertaintyMs;
    // A remote observation is valid only in its coarse network context.  The
    // device/model components remain shared, but a new network regime must not
    // look "already sampled" merely because this action ran under another RTT.
    result.historicalSamples = learned.actionSamples.count(action.stableKey())
        ? learned.actionSamples.at(action.stableKey()) : 0;
    result.total.samples = local ? result.historicalSamples : residual.samples;
    result.total.coldStart = result.total.samples == 0;
    result.candidate.predicted.set("latency.total_mean_ms", result.total.meanMs);
    result.candidate.predicted.set("latency.p95_ms", result.total.p95Ms);
    result.candidate.predicted.set("latency.uncertainty_ms", result.total.uncertaintyMs);
    result.candidate.predicted.set("payload.raw_bytes", static_cast<int64_t>(action.rawBundleBytes));
    result.candidate.predicted.set("payload.encoded_bytes", static_cast<int64_t>(action.estimatedEncodedBytes));
    result.candidate.predicted.set("payload.bundle_bytes", static_cast<int64_t>(action.estimatedEncodedBytes));
    result.candidate.predicted.set("accuracy.drop", action.estimatedAccuracyDrop);
    result.candidate.predicted.set("memory.peak_mb", action.peakEdgeMemoryMb);
    result.candidate.predicted.set("reliability.failure_probability", 0.0);
    result.candidate.predicted.set("safety.allowed", 1.0);
    result.candidate.annotations.set("profile.source", profileSourceName(result.source));
    result.candidate.annotations.set("profile.free", true);
    result.candidate.annotations.set("boundary.canonical_id", action.canonicalBoundaryId);
    return result;
}

void OnlineComponentEstimator::observe(const EstimatorObservation& observation) {
    if (observation.deviceCapabilityHash.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceState& state = devices_[observation.deviceCapabilityHash];
    const double alpha = observation.abrupt ? std::min(0.75, config_.ewmaAlpha * 2.0) : config_.ewmaAlpha;
    const bool split = observation.action.action.mode == ActionMode::SPLIT;
    const bool local = observation.action.action.mode == ActionMode::LOCAL;
    const double cpuUtil = std::max(0.0, numberOr(
        observation.state.features, "device.cpu.util_pct", 0.0));
    const double thermal = std::max(0.0, numberOr(
        observation.state.features, "device.thermal.soc_c", 25.0) - 60.0);
    const double edgeScale = (1.0 + 0.5 * cpuUtil / 100.0) * (1.0 + 0.01 * thermal);
    const double cloudUtil = std::max(0.0, numberOr(
        observation.state.features, "cloud.util_pct", 0.0));
    const double rawBytes = static_cast<double>(observation.action.rawBundleBytes);
    const double encodedBytes = static_cast<double>(observation.action.estimatedEncodedBytes);
    const auto updateRate = [&](ScalarSeries& series, double work, const char* metric) {
        const double duration = numberOr(observation.actual, metric, 0.0);
        if (work > 0.0 && duration > 0.0) updateSeries(series, work / duration, alpha);
    };
    if (config_.componentSharingEnabled) {
        if (local)
            updateRate(state.edgeFlopsPerMs, observation.action.prefixFlops,
                       "latency.edge_prefix_ms");
        if (!local && !split)
            updateRate(state.cloudFlopsPerMs, observation.action.suffixFlops,
                       "latency.cloud_suffix_ms");
        updateRate(state.captureBytesPerMs, observation.action.rawBundleBytes, "latency.capture_ms");
        updateRate(state.injectBytesPerMs, observation.action.rawBundleBytes, "latency.inject_ms");
        updateRate(state.encodeBytesPerMs[observation.action.action.codec],
                   observation.action.rawBundleBytes, "latency.encode_ms");
        updateRate(state.decodeBytesPerMs[observation.action.action.codec],
                   observation.action.estimatedEncodedBytes, "latency.decode_ms");
    }
    const double actualTotal = totalActual(observation.actual);
    if (actualTotal > 0.0) {
        double calibrated = observation.action.prefixFlops /
            seriesMean(state.edgeFlopsPerMs, config_.initialEdgeFlopsPerMs) * edgeScale;
        calibrated += observation.action.suffixFlops /
            seriesMean(state.cloudFlopsPerMs, config_.initialCloudFlopsPerMs) *
            (1.0 + cloudUtil / 100.0);
        if (split) {
            calibrated += rawBytes / seriesMean(state.captureBytesPerMs,
                                                 config_.initialCaptureBytesPerMs) * edgeScale;
            calibrated += rawBytes / seriesMean(
                state.encodeBytesPerMs[observation.action.action.codec],
                config_.initialCodecBytesPerMs) * edgeScale;
            calibrated += encodedBytes / seriesMean(
                state.decodeBytesPerMs[observation.action.action.codec],
                config_.initialCodecBytesPerMs) * (1.0 + cloudUtil / 200.0);
            calibrated += rawBytes / seriesMean(state.injectBytesPerMs,
                                                 config_.initialInjectBytesPerMs) *
                          (1.0 + cloudUtil / 200.0);
        }
        if (!local) {
            const double goodput = numberOr(
                observation.state.features, "network.goodput_up_bps", 0.0);
            const double rtt = numberOr(observation.state.features, "network.rtt_ms", 0.0);
            const double jitter = numberOr(observation.state.features, "network.jitter_ms", 0.0);
            const double loss = std::max(0.0, numberOr(
                observation.state.features, "network.loss_rate", 0.0));
            calibrated += rtt + jitter;
            if (goodput > 0.0)
                calibrated += encodedBytes * 1000.0 / goodput * (1.0 + 2.0 * loss);
            calibrated += std::max(0.0, numberOr(
                observation.state.features, "cloud.queue_ms", 0.0));
        }
        updateSignedSeries(state.actionResidualMs[residualContextKey(observation.action,
                                                                     observation.state)],
                           actualTotal - calibrated, alpha);
    }
    ++state.actionSamples[observation.action.stableKey()];
}

void OnlineComponentEstimator::migrateAction(const std::string& device,
        const std::string& oldKey, const std::string& newKey, double factor) {
    if (factor <= 0.0 || oldKey.empty() || newKey.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceState& state = devices_[device];
    std::vector<std::pair<std::string, ScalarSeries>> migrated;
    for (const auto& entry : state.actionResidualMs) {
        if (entry.first != oldKey && entry.first.rfind(oldKey + "|ctx:", 0) != 0) continue;
        ScalarSeries copy = entry.second;
        copy.mean *= factor;
        copy.variance /= std::max(0.01, factor);
        copy.samples = static_cast<std::size_t>(copy.samples * factor);
        migrated.emplace_back(newKey + entry.first.substr(oldKey.size()), copy);
    }
    for (const auto& entry : migrated) state.actionResidualMs[entry.first] = entry.second;
    const auto samples = state.actionSamples.find(oldKey);
    if (samples != state.actionSamples.end()) state.actionSamples[newKey] = static_cast<std::size_t>(samples->second * factor);
}

void OnlineComponentEstimator::eraseAction(const std::string& device, const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceState& state = devices_[device];
    for (auto it = state.actionResidualMs.begin(); it != state.actionResidualMs.end();) {
        if (it->first == key || it->first.rfind(key + "|ctx:", 0) == 0)
            it = state.actionResidualMs.erase(it);
        else ++it;
    }
    state.actionSamples.erase(key);
}

std::size_t OnlineComponentEstimator::actionSamples(const std::string& device,
                                                     const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto d = devices_.find(device);
    if (d == devices_.end()) return 0;
    const auto a = d->second.actionSamples.find(key);
    return a == d->second.actionSamples.end() ? 0 : a->second;
}

std::string AdaptationManager::key(const std::string& device, const std::string& model) const {
    return device + "\n" + model;
}

void AdaptationManager::setConfig(const AdaptationConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

AdaptationSnapshot AdaptationManager::snapshot(const std::string& device,
                                                const std::string& model) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = states_.find(key(device, model));
    return found == states_.end() ? AdaptationSnapshot() : found->second;
}

int AdaptationManager::selectProbe(const std::string& device, const std::string& model,
        const std::vector<ProfileFreePrediction>& predictions, int normal) const {
    AdaptationSnapshot state;
    AdaptationConfig config;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = states_.find(key(device, model));
        if (found != states_.end()) state = found->second;
        config = config_;
    }
    if (state.probesUsed >= config.maximumProbeRequests ||
        state.consecutiveProbes >= config.maximumConsecutiveProbes || predictions.empty()) return -1;
    const double normalMean = normal >= 0 && static_cast<std::size_t>(normal) < predictions.size()
        ? predictions[static_cast<std::size_t>(normal)].total.meanMs
        : std::numeric_limits<double>::infinity();
    auto coldSafe = [&](int index) {
        if (index < 0 || static_cast<std::size_t>(index) >= predictions.size()) return false;
        const auto& candidate = predictions[static_cast<std::size_t>(index)];
        if (candidate.total.meanMs > config.maximumAllowedP95Ms) return false;
        const double allowance = std::max(config.maximumProbeOverheadMs, normalMean);
        return candidate.total.meanMs <= normalMean + allowance;
    };
    if (state.phase == AdaptationPhase::UNINITIALIZED || state.phase == AdaptationPhase::ANCHORING) {
        // A local execution is the only network-independent safety anchor.  It
        // must remain probeable even when the uncalibrated FLOPs prior gives it
        // a pessimistic p95; otherwise cold start can incorrectly exclude the
        // very measurement needed to calibrate the edge rate.
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i].candidate.action.mode == ActionMode::LOCAL &&
                predictions[i].total.samples == 0) return static_cast<int>(i);
        }
        for (ActionMode mode : {ActionMode::CLOUD, ActionMode::SPLIT}) {
            for (std::size_t i = 0; i < predictions.size(); ++i) {
                if (predictions[i].candidate.action.mode == mode && predictions[i].total.samples == 0 && coldSafe(static_cast<int>(i)))
                    return static_cast<int>(i);
            }
        }
    }
    int experiencedBest = -1;
    std::size_t experiencedSamples = 0;
    double experiencedMean = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; config.historyGuidedCalibrationEnabled &&
         i < predictions.size(); ++i) {
        const auto& prediction = predictions[i];
        if (prediction.candidate.action.mode == ActionMode::LOCAL ||
            prediction.total.samples > 0 || !coldSafe(static_cast<int>(i)) ||
            prediction.historicalSamples == 0) continue;
        if (prediction.historicalSamples > experiencedSamples ||
            (prediction.historicalSamples == experiencedSamples &&
             prediction.total.meanMs < experiencedMean)) {
            experiencedSamples = prediction.historicalSamples;
            experiencedMean = prediction.total.meanMs;
            experiencedBest = static_cast<int>(i);
        }
    }
    if (experiencedBest >= 0) return experiencedBest;
    // Only after historical calibration has had a chance to choose its
    // evidence-rich action may a cold default remote action win the probe.
    if (normal >= 0 && static_cast<std::size_t>(normal) < predictions.size() &&
        predictions[static_cast<std::size_t>(normal)].candidate.action.mode != ActionMode::LOCAL &&
        predictions[static_cast<std::size_t>(normal)].total.samples == 0 && coldSafe(normal))
        return normal;
    int contextualBest = -1;
    double contextualMean = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].candidate.action.mode == ActionMode::LOCAL ||
            predictions[i].total.samples > 0 || !coldSafe(static_cast<int>(i))) continue;
        if (predictions[i].total.meanMs < contextualMean) {
            contextualMean = predictions[i].total.meanMs;
            contextualBest = static_cast<int>(i);
        }
    }
    if (contextualBest >= 0) return contextualBest;
    int best = -1;
    double information = -1.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (!coldSafe(static_cast<int>(i)) || static_cast<int>(i) == normal || predictions[i].total.samples > 0) continue;
        const double value = predictions[i].total.uncertaintyMs /
                             std::max(1.0, predictions[i].total.p95Ms);
        if (value > information) { information = value; best = static_cast<int>(i); }
    }
    return best;
}

void AdaptationManager::observe(const std::string& device, const std::string& model,
        const ProfileFreePrediction& prediction, const MetricBag& actual,
        bool success, bool wasProbe, double oracleMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    AdaptationSnapshot& state = states_[key(device, model)];
    if (state.phase == AdaptationPhase::UNINITIALIZED) state.phase = AdaptationPhase::ANCHORING;
    ++state.observations;
    if (wasProbe) { ++state.probesUsed; ++state.consecutiveProbes; }
    else state.consecutiveProbes = 0;
    const double actualMs = totalActual(actual);
    if (!success || actualMs <= 0.0) {
        ++state.failures;
        state.stableObservations = 0;
        if (state.failures >= config_.driftFailures) state.phase = AdaptationPhase::DRIFTED;
        return;
    }
    state.failures = 0;
    state.lastSafeActionKey = prediction.candidate.action.key();
    const double relative = std::fabs(actualMs - prediction.total.meanMs) / std::max(1.0, actualMs);
    if (relative <= config_.convergenceRelativeError) ++state.stableObservations;
    else state.stableObservations = 0;
    if (oracleMs >= 0.0) state.cumulativeRegretMs += std::max(0.0, actualMs - oracleMs);
    if (state.stableObservations >= config_.stableWindowRequests) state.phase = AdaptationPhase::STABLE;
    else if (state.probesUsed < config_.maximumProbeRequests) state.phase = AdaptationPhase::SAFE_EXPLORING;
    else state.phase = AdaptationPhase::ADAPTING;
}

void AdaptationManager::markDrift(const std::string& device, const std::string& model) {
    std::lock_guard<std::mutex> lock(mutex_);
    AdaptationSnapshot& state = states_[key(device, model)];
    state.phase = AdaptationPhase::DRIFTED;
    state.stableObservations = 0;
    state.consecutiveProbes = 0;
}

void AdaptationManager::markEnvironmentChange(const std::string& device,
                                               const std::string& model,
                                               int probeRequests) {
    std::lock_guard<std::mutex> lock(mutex_);
    AdaptationSnapshot& state = states_[key(device, model)];
    state.phase = AdaptationPhase::DRIFTED;
    state.stableObservations = 0;
    state.consecutiveProbes = 0;
    const int budget = probeRequests < 0 ? config_.changeProbeRequests : probeRequests;
    state.probesUsed = std::max(0, config_.maximumProbeRequests - std::max(0, budget));
    ++state.environmentChanges;
}

void AdaptationManager::migrateModel(const std::string& device, const std::string& oldModel,
                                     const std::string& newModel, double factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto old = states_.find(key(device, oldModel));
    AdaptationSnapshot next;
    if (old != states_.end() && factor > 0.0) {
        next = old->second;
        next.probesUsed = static_cast<int>(next.probesUsed * factor);
        next.stableObservations = static_cast<int>(next.stableObservations * factor);
        next.phase = factor >= 0.99 ? AdaptationPhase::ADAPTING : AdaptationPhase::DRIFTED;
        next.consecutiveProbes = 0;
    }
    states_[key(device, newModel)] = next;
}

ProfileFreeRuntime::ProfileFreeRuntime(ProfileFreeControllerConfig config)
    : config_(config), pruner_(config.pruning), estimator_(config.estimator), adaptation_(config.adaptation) {}

void ProfileFreeRuntime::setConfig(const ProfileFreeControllerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    pruner_ = CandidatePruner(config.pruning);
    estimator_.setConfig(config.estimator);
    adaptation_.setConfig(config.adaptation);
}

bool ProfileFreeRuntime::registerManifest(const StructuralModelManifest& manifest, std::string* error) {
    if (!manifest.valid(error)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    manifests_[manifest.modelHash] = manifest;
    actionCache_[manifest.modelHash] = std::make_shared<const ActionBuildResult>(
        builder_.build(manifest));
    return true;
}

bool ProfileFreeRuntime::hasManifest(const std::string& modelHash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return manifests_.count(modelHash) != 0;
}

ProfileFreeDecision ProfileFreeRuntime::decide(const std::string& device, const RuntimeState& state) {
    const uint64_t totalStart = steadyNs();
    static std::atomic<uint64_t> sequence{0};
    ProfileFreeDecision decision;
    decision.traceId = "pfhoc-" + std::to_string(steadyNs()) + "-" + std::to_string(++sequence);
    decision.requestId = state.requestId;
    decision.deviceCapabilityHash = device;
    decision.modelHash = state.modelHash;
    std::shared_ptr<const ActionBuildResult> built;
    ProfileFreeControllerConfig config;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = manifests_.find(state.modelHash);
        if (found == manifests_.end()) {
            decision.reason = "structural_manifest_missing";
            decision.totalDecisionMs = elapsedMs(totalStart);
            return decision;
        }
        const auto cached = actionCache_.find(state.modelHash);
        if (cached == actionCache_.end()) {
            decision.reason = "structural_action_cache_missing";
            decision.totalDecisionMs = elapsedMs(totalStart);
            return decision;
        }
        built = cached->second;
        knownDevices_.insert(device);
        config = config_;
    }
    decision.shadowMode = config.shadowMode;
    decision.bootstrapMode = config.bootstrapMode;
    if (!config.enabled) {
        decision.reason = "profile_free_controller_disabled";
        decision.totalDecisionMs = elapsedMs(totalStart);
        return decision;
    }
    if (config.bootstrapMode == ControllerBootstrapMode::OFFLINE_PROFILE) {
        decision.reason = "delegated_to_offline_profile_controller";
        decision.totalDecisionMs = elapsedMs(totalStart);
        return decision;
    }
    // Building structural actions is a model-registration cost, not a request
    // hot-path cost.  The cached vector is immutable for this model hash.
    decision.actionBuildMs = 0.0;
    decision.environmentKey = runtimeEnvironmentKey(state);
    bool environmentImproved = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string key = device + "\n" + state.modelHash;
        const auto previous = environmentKeys_.find(key);
        decision.environmentChanged = previous != environmentKeys_.end() &&
                                      previous->second != decision.environmentKey;
        const double quality = runtimeEnvironmentQuality(state);
        const auto previousQuality = environmentQuality_.find(key);
        environmentImproved = previousQuality != environmentQuality_.end() &&
                              quality > previousQuality->second;
        environmentKeys_[key] = decision.environmentKey;
        environmentQuality_[key] = quality;
    }
    // Degradation needs no risky re-probe: the analytic transfer term becomes
    // worse immediately and local remains the safety anchor.  Re-open a tiny
    // probe budget only when connectivity improves enough that split execution
    // can plausibly become competitive again.
    if (decision.environmentChanged)
        adaptation_.markEnvironmentChange(device, state.modelHash,
            environmentImproved ? config.adaptation.changeProbeRequests : 1);
    const AdaptationSnapshot before = adaptation_.snapshot(device, state.modelHash);
    decision.pruning = pruner_.prune(state, built->actions, before.lastSafeActionKey);
    const uint64_t predictionStart = steadyNs();
    decision.predictions = estimator_.predictBatch(device, decision.pruning.shortlist, state);
    decision.predictionMs = elapsedMs(predictionStart);
    const uint64_t selectionStart = steadyNs();
    int normal = -1;
    double best = std::numeric_limits<double>::infinity();
    const bool hasSampledCandidate = std::any_of(
        decision.predictions.begin(), decision.predictions.end(),
        [](const ProfileFreePrediction& prediction) {
            return prediction.total.samples > 0;
        });
    for (std::size_t i = 0; i < decision.predictions.size(); ++i) {
        if (decision.predictions[i].total.p95Ms >
            config.adaptation.maximumAllowedP95Ms) continue;
        // Once a measured safe path exists, an uncalibrated remote path may be
        // entered only through AdaptationManager's bounded probe budget.  This
        // prevents optimistic structural priors from becoming unbounded probes.
        if (hasSampledCandidate && decision.predictions[i].total.samples == 0 &&
            decision.predictions[i].candidate.action.mode != ActionMode::LOCAL)
            continue;
        double safe = decision.predictions[i].total.meanMs +
            config.riskMultiplier * decision.predictions[i].total.uncertaintyMs;
        if (safe < best) { best = safe; normal = static_cast<int>(i); }
    }
    int selected = normal;
    if (config.explorationEnabled) {
        const int probe = adaptation_.selectProbe(device, state.modelHash, decision.predictions, normal);
        if (probe >= 0) {
            selected = probe;
            const bool firstLocalAnchor = before.observations == 0 &&
                decision.predictions[static_cast<std::size_t>(probe)].candidate.action.mode ==
                    ActionMode::LOCAL;
            decision.selectedAsProbe = !firstLocalAnchor;
        }
    }
    decision.selectedIndex = selected;
    const bool localAnchor = selected >= 0 && before.observations == 0 &&
        decision.predictions[static_cast<std::size_t>(selected)].candidate.action.mode ==
            ActionMode::LOCAL;
    decision.reason = selected < 0 ? "no_safe_candidate" :
        (localAnchor ? "local_calibration_anchor" :
         (decision.selectedAsProbe ? "bounded_safe_probe" :
          "minimum_uncertainty_adjusted_cost"));
    decision.selectionMs = elapsedMs(selectionStart);
    decision.adaptation = before;
    decision.totalDecisionMs = elapsedMs(totalStart);
    if (selected >= 0) {
        PendingDecision pending;
        pending.deviceCapabilityHash = device;
        pending.modelHash = state.modelHash;
        pending.prediction = decision.predictions[selected];
        pending.descriptor = decision.pruning.shortlist[static_cast<std::size_t>(selected)];
        pending.state = state;
        pending.wasProbe = decision.selectedAsProbe;
        std::lock_guard<std::mutex> lock(mutex_);
        pending_[decision.traceId] = pending;
    }
    return decision;
}

bool ProfileFreeRuntime::observe(const std::string& traceId, const MetricBag& actual,
        bool success, bool abrupt, double oracleMs, std::string* error) {
    PendingDecision pending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = pending_.find(traceId);
        if (found == pending_.end()) { if (error) *error = "trace not found"; return false; }
        pending = found->second;
        pending_.erase(found);
    }
    EstimatorObservation observation;
    observation.deviceCapabilityHash = pending.deviceCapabilityHash;
    observation.action = pending.descriptor;
    observation.state = pending.state;
    observation.actual = actual;
    observation.success = success;
    observation.abrupt = abrupt;
    if (success) estimator_.observe(observation);
    adaptation_.observe(pending.deviceCapabilityHash, pending.modelHash,
                        pending.prediction, actual, success, pending.wasProbe, oracleMs);
    return true;
}

std::string ProfileFreeRuntime::statusJson(const std::string& device,
                                           const std::string& model) const {
    const AdaptationSnapshot state = adaptation_.snapshot(device, model);
    std::ostringstream out;
    out << "{\"schemaVersion\":1,\"deviceCapabilityHash\":\"" << jsonEscape(device)
        << "\",\"modelHash\":\"" << jsonEscape(model) << "\",\"phase\":\""
        << adaptationPhaseName(state.phase) << "\",\"probesUsed\":" << state.probesUsed
        << ",\"observations\":" << state.observations << ",\"stableObservations\":"
        << state.stableObservations << ",\"failures\":" << state.failures
        << ",\"cumulativeRegretMs\":" << state.cumulativeRegretMs
        << ",\"lastSafeActionKey\":\"" << jsonEscape(state.lastSafeActionKey) << "\"}";
    return out.str();
}

std::string ProfileFreeDecision::toJson() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\"schemaVersion\":1,\"traceId\":\"" << jsonEscape(traceId)
        << "\",\"requestId\":\"" << jsonEscape(requestId)
        << "\",\"deviceCapabilityHash\":\"" << jsonEscape(deviceCapabilityHash)
        << "\",\"modelHash\":\"" << jsonEscape(modelHash)
        << "\",\"selectedIndex\":" << selectedIndex << ",\"selectedAsProbe\":"
        << (selectedAsProbe ? "true" : "false") << ",\"shadowMode\":"
        << (shadowMode ? "true" : "false") << ",\"bootstrapMode\":\""
        << controllerBootstrapModeName(bootstrapMode) << "\",\"reason\":\"" << jsonEscape(reason)
        << "\",\"adaptationPhase\":\"" << adaptationPhaseName(adaptation.phase)
        << "\",\"environmentChanged\":" << (environmentChanged ? "true" : "false")
        << ",\"environmentKey\":\"" << jsonEscape(environmentKey)
        << "\",\"timing\":{\"buildMs\":" << actionBuildMs << ",\"pruneMs\":"
        << pruning.pruneMs << ",\"predictionMs\":" << predictionMs
        << ",\"selectionMs\":" << selectionMs << ",\"totalMs\":" << totalDecisionMs
        << "},\"pruning\":{\"input\":" << pruning.inputActions
        << ",\"shortlist\":" << pruning.shortlist.size() << ",\"hardRejected\":"
        << pruning.hardRejected << ",\"lowerBoundRejected\":" << pruning.lowerBoundRejected
        << "},\"selectedAction\":";
    if (selectedIndex >= 0 && static_cast<std::size_t>(selectedIndex) < predictions.size()) {
        const auto& action = predictions[static_cast<std::size_t>(selectedIndex)].candidate.action;
        out << "{\"key\":\"" << jsonEscape(action.key()) << "\",\"mode\":\""
            << (action.mode == ActionMode::LOCAL ? "local" :
                (action.mode == ActionMode::CLOUD ? "cloud" : "split"))
            << "\",\"boundary\":" << action.boundary << ",\"codec\":\""
            << profileCodecName(action.codec) << "\"}";
    } else out << "null";
    out << ",\"predictions\":[";
    for (std::size_t i = 0; i < predictions.size(); ++i) {
        if (i) out << ',';
        out << "{\"actionKey\":\"" << jsonEscape(predictions[i].candidate.action.key())
            << "\",\"structuralKey\":\"" << jsonEscape(predictions[i].structuralActionKey)
            << "\",\"meanMs\":" << predictions[i].total.meanMs
            << ",\"p95Ms\":" << predictions[i].total.p95Ms
            << ",\"uncertaintyMs\":" << predictions[i].total.uncertaintyMs
            << ",\"samples\":" << predictions[i].total.samples
            << ",\"historicalSamples\":" << predictions[i].historicalSamples
            << ",\"source\":\"" << profileSourceName(predictions[i].source) << "\"}";
    }
    out << "]}";
    return out.str();
}

ProfileFreeMigrationResult OnlineStateMigrationManager::migrate(
    const std::string& device, const StructuralModelManifest& oldManifest,
    const StructuralModelManifest& newManifest, const CompatibilityMap& compatibility,
    OnlineComponentEstimator& estimator, AdaptationManager& adaptation) {
    const uint64_t start = steadyNs();
    ProfileFreeMigrationResult result;
    StructuralActionBuilder builder;
    const auto oldActions = builder.build(oldManifest).actions;
    const auto newActions = builder.build(newManifest).actions;
    auto actionsFor = [](const std::vector<StructuralActionDescriptor>& actions, const std::string& canonical) {
        std::vector<StructuralActionDescriptor> found;
        for (const auto& action : actions) if (action.canonicalBoundaryId == canonical) found.push_back(action);
        return found;
    };
    for (const auto& mapping : compatibility.boundaries) {
        const auto oldSet = actionsFor(oldActions, mapping.oldCanonicalId);
        const auto newSet = actionsFor(newActions, mapping.newCanonicalId);
        for (const auto& oldAction : oldSet) {
            const auto match = std::find_if(newSet.begin(), newSet.end(), [&](const StructuralActionDescriptor& action) {
                return action.action.codec == oldAction.action.codec;
            });
            if (mapping.kind == MappingKind::INCOMPATIBLE || match == newSet.end()) {
                estimator.eraseAction(device, oldAction.stableKey());
                ++result.incompatibleRemoved;
            } else if (mapping.kind == MappingKind::EXACT) {
                estimator.migrateAction(device, oldAction.stableKey(), match->stableKey(), 1.0);
                ++result.exactReused;
            } else if (mapping.kind == MappingKind::CONTRACT_COMPATIBLE) {
                estimator.migrateAction(device, oldAction.stableKey(), match->stableKey(), 0.65);
                ++result.compatibleReused;
                ++result.requiredProbes;
            } else {
                estimator.eraseAction(device, match->stableKey());
                ++result.calibrationReset;
                ++result.requiredProbes;
            }
        }
    }
    const double reusable = result.exactReused + 0.65 * result.compatibleReused;
    const double total = result.exactReused + result.compatibleReused +
        result.calibrationReset + result.incompatibleRemoved;
    adaptation.migrateModel(device, oldManifest.modelHash, newManifest.modelHash,
                            total > 0.0 ? reusable / total : 0.0);
    result.migrationMs = elapsedMs(start);
    return result;
}

ProfileFreeMigrationResult ProfileFreeRuntime::migrateVersion(
    const std::string& device, const std::string& oldModelHash,
    const std::string& newModelHash, const CompatibilityMap& compatibility,
    std::string* error) {
    StructuralModelManifest oldManifest;
    StructuralModelManifest newManifest;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto oldIt = manifests_.find(oldModelHash);
        const auto newIt = manifests_.find(newModelHash);
        if (oldIt == manifests_.end() || newIt == manifests_.end()) {
            if (error) *error = "old or new structural manifest missing";
            return {};
        }
        oldManifest = oldIt->second;
        newManifest = newIt->second;
    }
    return OnlineStateMigrationManager::migrate(
        device, oldManifest, newManifest, compatibility, estimator_, adaptation_);
}

ProfileFreeMigrationResult ProfileFreeRuntime::migrateKnownDevices(
    const std::string& oldModelHash, const std::string& newModelHash,
    const CompatibilityMap& compatibility, std::string* error) {
    std::vector<std::string> devices;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!manifests_.count(oldModelHash) || !manifests_.count(newModelHash)) {
            if (error) *error = "old or new structural manifest missing";
            return {};
        }
        devices.assign(knownDevices_.begin(), knownDevices_.end());
    }
    ProfileFreeMigrationResult aggregate;
    const uint64_t start = steadyNs();
    for (const auto& device : devices) {
        const auto result = migrateVersion(device, oldModelHash, newModelHash,
                                           compatibility, error);
        aggregate.exactReused += result.exactReused;
        aggregate.compatibleReused += result.compatibleReused;
        aggregate.calibrationReset += result.calibrationReset;
        aggregate.incompatibleRemoved += result.incompatibleRemoved;
        aggregate.requiredProbes += result.requiredProbes;
    }
    aggregate.migrationMs = elapsedMs(start);
    return aggregate;
}

std::string ProfileFreeMigrationResult::toJson() const {
    std::ostringstream out;
    out << "{\"schemaVersion\":1,\"exactReused\":" << exactReused
        << ",\"compatibleReused\":" << compatibleReused
        << ",\"calibrationReset\":" << calibrationReset
        << ",\"incompatibleRemoved\":" << incompatibleRemoved
        << ",\"requiredProbes\":" << requiredProbes
        << ",\"migrationMs\":" << migrationMs << "}";
    return out.str();
}
