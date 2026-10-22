#include "AdaptiveRuntime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

namespace {

uint64_t unixMs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

double numberOr(const MetricBag& bag, const std::string& name, double fallback) {
    double value = fallback;
    return bag.getNumber(name, value) ? value : fallback;
}

const std::vector<std::string>& latencyComponents() {
    static const std::vector<std::string> names = {
        "latency.edge_prefix_ms", "latency.capture_ms", "latency.encode_ms",
        "latency.transfer_ms", "latency.decode_ms", "latency.inject_ms",
        "latency.cloud_queue_ms", "latency.cloud_suffix_ms"};
    return names;
}

int codecSafetyRank(CodecKind codec) {
    switch (codec) {
        case CodecKind::BLOCK_INT4: return 0;
        case CodecKind::BLOCK_INT6: return 1;
        case CodecKind::BLOCK_INT8: return 2;
        case CodecKind::FP16: return 3;
        case CodecKind::LZ4: return 4;
        case CodecKind::RAW32: return 5;
    }
    return -1;
}

}  // namespace

RuntimeState RuntimeStateAdapter::build(
    const PerformanceMetrics& device, const NetworkRuntimeSample& network,
    const CloudRuntimeSample& cloud, const RuntimeHistorySample& history,
    const std::string& requestId, const std::string& modelName,
    const std::string& modelHash, const std::string& modelVersion) {
    RuntimeState state;
    state.schemaVersion = buildDefaultRuntimeStateSchema().version();
    state.timestampUnixMs = unixMs();
    state.requestId = requestId;
    state.modelName = modelName;
    state.modelHash = modelHash;
    state.modelVersion = modelVersion;
    state.features.set("device.cpu.util_pct", device.cpuUsagePercent);
    state.features.set("device.cpu.freq_mhz", device.cpuFrequencyMHz);
    state.features.set("device.memory.used_mb", device.memoryUsageMB);
    state.features.set("device.memory.peak_mb", device.memoryPeakMB);
    state.features.set("device.thermal.soc_c", device.socTempCelsius);
    state.features.set("device.thermal.battery_c", device.batteryTempCelsius);
    state.features.set("device.battery.power_mw", device.powerConsumptionMw);
    state.features.set("network.rtt_ms", network.rttMs);
    state.features.set("network.goodput_up_bps", network.goodputUpBps);
    state.features.set("network.jitter_ms", network.jitterMs);
    state.features.set("network.loss_rate", network.lossRate);
    state.features.set("network.interface", network.interfaceName);
    state.features.set("network.available", network.available ? 1.0 : 0.0);
    state.features.set("cloud.queue_ms", cloud.queueMs);
    state.features.set("cloud.util_pct", cloud.utilizationPct);
    state.features.set("cloud.available", cloud.available ? 1.0 : 0.0);
    state.features.set("history.residual_ms", history.latestResidualMs);
    state.features.set("history.fallback_count", history.consecutiveFallbacks);
    state.features.set("history.previous_action", history.previousActionKey);
    return state;
}

bool OfflineProfileStore::replace(const ProfileManifest& manifest, std::string* error) {
    if (manifest.manifestId.empty() || manifest.modelHash.empty() || manifest.profiles.empty()) {
        if (error) *error = "profile manifest fields missing";
        return false;
    }
    std::set<std::string> actions;
    for (const auto& profile : manifest.profiles) {
        if (!profile.action.modelHash.empty() && profile.action.modelHash != manifest.modelHash) {
            if (error) *error = "profile model hash mismatch";
            return false;
        }
        const bool invalidTiming = !std::isfinite(profile.edgePrefixMs) || profile.edgePrefixMs < 0.0 ||
            !std::isfinite(profile.captureMs) || profile.captureMs < 0.0 ||
            !std::isfinite(profile.encodeMs) || profile.encodeMs < 0.0 ||
            !std::isfinite(profile.decodeMs) || profile.decodeMs < 0.0 ||
            !std::isfinite(profile.injectMs) || profile.injectMs < 0.0 ||
            !std::isfinite(profile.cloudSuffixMs) || profile.cloudSuffixMs < 0.0;
        if (!actions.insert(profile.action.key()).second || invalidTiming || profile.bundleBytes < 0 ||
            profile.accuracyDrop < 0.0 || profile.failureProbability < 0.0 ||
            profile.failureProbability > 1.0) {
            if (error) *error = "invalid or duplicate component profile";
            return false;
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    manifests_[manifest.modelHash] = manifest;
    return true;
}

bool OfflineProfileStore::getManifest(const std::string& modelHash, ProfileManifest& manifest) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = manifests_.find(modelHash);
    if (found == manifests_.end()) return false;
    manifest = found->second;
    return true;
}

std::vector<ComponentProfile> OfflineProfileStore::profiles(const std::string& modelHash) const {
    ProfileManifest manifest;
    return getManifest(modelHash, manifest) ? manifest.profiles : std::vector<ComponentProfile>();
}

bool OfflineProfileStore::invalidate(const std::string& modelHash) {
    std::lock_guard<std::mutex> lock(mutex_);
    return manifests_.erase(modelHash) != 0;
}

ActionCandidate ComponentCostPredictor::predict(
    const ComponentProfile& profile, const RuntimeState& state) const {
    ActionCandidate candidate;
    candidate.action = profile.action;
    candidate.annotations = profile.extensions;
    double transferMs = profile.fixedTransferMs;
    if (profile.action.mode != ActionMode::LOCAL) {
        const double available = numberOr(state.features, "network.available", 0.0);
        const double goodput = numberOr(state.features, "network.goodput_up_bps", 0.0);
        if (available < 0.5 || goodput < config_.minimumGoodputBps) {
            transferMs = config_.unavailableNetworkPenaltyMs;
        } else {
            const double rtt = numberOr(state.features, "network.rtt_ms", 0.0);
            const double jitter = numberOr(state.features, "network.jitter_ms", 0.0);
            const double loss = std::max(0.0, numberOr(state.features, "network.loss_rate", 0.0));
            transferMs += rtt * config_.controlRttFactor + jitter +
                static_cast<double>(profile.bundleBytes) * 1000.0 / goodput *
                (1.0 + loss * config_.lossPenaltyFactor);
        }
    }
    double edgeScale = 1.0;
    const double cpuFrequency = numberOr(state.features, "device.cpu.freq_mhz", 0.0);
    const double cpuUtil = numberOr(state.features, "device.cpu.util_pct", 0.0);
    if (profile.referenceCpuFrequencyMhz > 0.0 && cpuFrequency > 0.0) {
        edgeScale *= profile.referenceCpuFrequencyMhz / cpuFrequency;
    }
    edgeScale *= 1.0 + profile.cpuUtilSensitivity * std::max(0.0, cpuUtil) / 100.0;
    edgeScale = std::max(0.1, edgeScale);
    double cloudScale = 1.0;
    const double cloudUtil = numberOr(state.features, "cloud.util_pct", 0.0);
    cloudScale *= 1.0 + profile.cloudUtilSensitivity *
        std::max(0.0, cloudUtil - profile.referenceCloudUtilPct) / 100.0;
    cloudScale = std::max(0.1, cloudScale);
    const double edgePrefix = profile.edgePrefixMs * edgeScale;
    const double capture = profile.captureMs * edgeScale;
    const double encode = profile.encodeMs * edgeScale;
    const double decode = profile.decodeMs * cloudScale;
    const double inject = profile.injectMs * cloudScale;
    const double suffix = profile.cloudSuffixMs * cloudScale;
    const double cloudQueue = profile.action.mode == ActionMode::LOCAL
        ? 0.0 : std::max(0.0, numberOr(state.features, "cloud.queue_ms", 0.0));
    candidate.predicted.set("latency.edge_prefix_ms", edgePrefix);
    candidate.predicted.set("latency.capture_ms", capture);
    candidate.predicted.set("latency.encode_ms", encode);
    candidate.predicted.set("latency.transfer_ms", transferMs);
    candidate.predicted.set("latency.decode_ms", decode);
    candidate.predicted.set("latency.inject_ms", inject);
    candidate.predicted.set("latency.cloud_queue_ms", cloudQueue);
    candidate.predicted.set("latency.cloud_suffix_ms", suffix);
    const double total = edgePrefix + capture + encode + transferMs + decode + inject + cloudQueue + suffix;
    candidate.predicted.set("latency.total_mean_ms", total);
    candidate.predicted.set("latency.p95_ms", total + profile.offlineP95PaddingMs);
    candidate.predicted.set("payload.raw_bytes", profile.rawBytes);
    candidate.predicted.set("payload.encoded_bytes", profile.encodedBytes);
    candidate.predicted.set("payload.bundle_bytes", profile.bundleBytes);
    candidate.predicted.set("accuracy.drop", profile.accuracyDrop);
    candidate.predicted.set("memory.peak_mb", profile.peakMemoryMb);
    candidate.predicted.set("energy.mj", profile.energyMj);
    candidate.predicted.set("reliability.failure_probability", profile.failureProbability);
    return candidate;
}

std::vector<ActionCandidate> ComponentCostPredictor::predictAll(
    const std::vector<ComponentProfile>& profiles, const RuntimeState& state) const {
    std::vector<ActionCandidate> result;
    result.reserve(profiles.size());
    for (const auto& profile : profiles) result.push_back(predict(profile, state));
    return result;
}

std::string OnlineResidualModel::key(const std::string& actionKey,
                                     const std::string& component) const {
    return actionKey + "\n" + component;
}

void OnlineResidualModel::update(const std::string& actionKey, const std::string& component,
                                 double predictedMs, double actualMs, bool abrupt) {
    if (!std::isfinite(predictedMs) || !std::isfinite(actualMs)) return;
    const double residual = actualMs - predictedMs;
    std::lock_guard<std::mutex> lock(mutex_);
    Series& series = series_[key(actionKey, component)];
    const double alpha = abrupt ? config_.abruptAlpha : config_.alpha;
    series.ewmaMs = series.samples == 0 ? residual : (1.0 - alpha) * series.ewmaMs + alpha * residual;
    ++series.samples;
    series.positiveResiduals.push_back(std::max(0.0, residual - series.ewmaMs));
    while (series.positiveResiduals.size() > config_.windowSize) series.positiveResiduals.pop_front();
}

ResidualSnapshot OnlineResidualModel::snapshot(
    const std::string& actionKey, const std::string& component) const {
    std::lock_guard<std::mutex> lock(mutex_);
    ResidualSnapshot result;
    const auto found = series_.find(key(actionKey, component));
    if (found == series_.end()) return result;
    result.samples = found->second.samples;
    result.ewmaMs = found->second.ewmaMs;
    if (!found->second.positiveResiduals.empty()) {
        std::vector<double> sorted(found->second.positiveResiduals.begin(),
                                   found->second.positiveResiduals.end());
        std::sort(sorted.begin(), sorted.end());
        const size_t index = static_cast<size_t>(std::ceil(0.95 * sorted.size())) - 1;
        result.positiveP95Ms = sorted[std::min(index, sorted.size() - 1)];
    }
    return result;
}

double OnlineResidualModel::correctedMean(
    const std::string& actionKey, const std::string& component, double offlineMs) const {
    const auto value = snapshot(actionKey, component);
    return std::max(0.0, offlineMs + value.ewmaMs);
}

double OnlineResidualModel::correctedP95(
    const std::string& actionKey, const std::string& component,
    double offlineMs, double offlinePaddingMs) const {
    const auto value = snapshot(actionKey, component);
    const double onlinePadding = value.samples >= config_.minimumOnlineSamples
        ? value.positiveP95Ms : std::max(config_.coldStartPaddingMs, offlinePaddingMs);
    return std::max(0.0, offlineMs + value.ewmaMs + onlinePadding);
}

void OnlineResidualModel::updateFromObservation(
    const SplitAction& action, const MetricBag& predicted, const MetricBag& actual,
    bool networkAbrupt) {
    for (const std::string& component : latencyComponents()) {
        double predictedValue = 0.0, actualValue = 0.0;
        if (predicted.getNumber(component, predictedValue) && actual.getNumber(component, actualValue)) {
            update(action.key(), component, predictedValue, actualValue,
                   networkAbrupt && component == "latency.transfer_ms");
        }
    }
}

void OnlineResidualModel::migrateAction(const std::string& oldActionKey,
                                        const std::string& newActionKey,
                                        double computeFactor, double codecFactor,
                                        double networkFactor) {
    if (oldActionKey.empty() || newActionKey.empty() || oldActionKey == newActionKey) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string prefix = oldActionKey + "\n";
    std::vector<std::pair<std::string, Series>> migrated;
    for (const auto& entry : series_) {
        if (entry.first.compare(0, prefix.size(), prefix) != 0) continue;
        const std::string component = entry.first.substr(prefix.size());
        double factor = computeFactor;
        if (component == "latency.transfer_ms" || component == "latency.cloud_queue_ms") {
            factor = networkFactor;
        } else if (component == "latency.encode_ms" || component == "latency.decode_ms") {
            factor = codecFactor;
        }
        if (factor <= 0.0) continue;
        Series copy = entry.second;
        copy.ewmaMs *= factor;
        for (double& residual : copy.positiveResiduals) residual *= factor;
        migrated.emplace_back(newActionKey + "\n" + component, std::move(copy));
    }
    for (auto& entry : migrated) series_[entry.first] = std::move(entry.second);
}

void OnlineResidualModel::eraseAction(const std::string& actionKey) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string prefix = actionKey + "\n";
    for (auto it = series_.begin(); it != series_.end();) {
        if (it->first.compare(0, prefix.size(), prefix) == 0) it = series_.erase(it);
        else ++it;
    }
}

ActionCandidate OnlineCorrectedPredictor::predict(
    const ComponentProfile& profile, const RuntimeState& state) const {
    ActionCandidate candidate = base_.predict(profile, state);
    if (!residuals_) return candidate;
    double correctedTotal = 0.0;
    double correctedP95 = 0.0;
    for (const std::string& component : latencyComponents()) {
        double offline = 0.0;
        candidate.predicted.getNumber(component, offline);
        const double mean = residuals_->correctedMean(profile.action.key(), component, offline);
        const double p95 = residuals_->correctedP95(
            profile.action.key(), component, offline, profile.offlineP95PaddingMs / latencyComponents().size());
        candidate.predicted.set(component, mean);
        correctedTotal += mean;
        correctedP95 += p95;
    }
    candidate.predicted.set("latency.total_mean_ms", correctedTotal);
    candidate.predicted.set("latency.p95_ms", correctedP95);
    return candidate;
}

void OnlineCorrectedPredictor::observe(
    const ActionCandidate& prediction, const MetricBag& actual, bool networkAbrupt) {
    if (residuals_) residuals_->updateFromObservation(
        prediction.action, prediction.predicted, actual, networkAbrupt);
}

SafetyAssessment SafetyFilter::assess(
    const RuntimeState& state, const ActionCandidate& candidate) const {
    SafetyAssessment result;
    const auto reject = [&](const std::string& reason) {
        result.allowed = false;
        result.reasons.push_back(reason);
    };
    if (config_.requireModelHash &&
        (candidate.action.modelHash.empty() || candidate.action.modelHash != state.modelHash)) {
        reject("model_hash_mismatch");
    }
    if (numberOr(candidate.predicted, "accuracy.drop", 0.0) > config_.maxAccuracyDrop) reject("accuracy_limit");
    if (numberOr(candidate.predicted, "memory.peak_mb", 0.0) > config_.maxPeakMemoryMb) reject("memory_limit");
    if (numberOr(candidate.predicted, "reliability.failure_probability", 0.0) >
        config_.maxFailureProbability) reject("failure_risk");
    const double temperature = numberOr(state.features, "device.thermal.soc_c", 0.0);
    if (temperature >= config_.hardSocTemperatureC && candidate.action.mode != ActionMode::CLOUD) {
        reject("thermal_hard");
    }
    const bool networkAvailable = numberOr(state.features, "network.available", 0.0) >= 0.5;
    const bool cloudAvailable = numberOr(state.features, "cloud.available", 0.0) >= 0.5;
    if (candidate.action.mode != ActionMode::LOCAL && (!networkAvailable || !cloudAvailable)) {
        reject("remote_unavailable");
    }
    return result;
}

void SafetyFilter::annotate(
    const RuntimeState& state, std::vector<ActionCandidate>& candidates) const {
    for (auto& candidate : candidates) {
        const SafetyAssessment assessment = assess(state, candidate);
        candidate.predicted.set("safety.allowed", assessment.allowed ? 1.0 : 0.0);
        candidate.annotations.set("safety.allowed", assessment.allowed);
        std::string reasons;
        for (size_t i = 0; i < assessment.reasons.size(); ++i) {
            if (i) reasons += '|';
            reasons += assessment.reasons[i];
        }
        candidate.annotations.set("safety.reasons", reasons);
    }
}

PolicyDecision HysteresisPolicy::select(
    const PolicyContext& context, const std::vector<CandidateEvaluation>& evaluations) {
    PolicyDecision result;
    int best = -1;
    int current = -1;
    double bestScore = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < evaluations.size() && i < context.candidates.size(); ++i) {
        if (context.candidates[i].action.key() == currentActionKey_) current = static_cast<int>(i);
        if (evaluations[i].feasible && evaluations[i].score < bestScore) {
            bestScore = evaluations[i].score;
            best = static_cast<int>(i);
        }
    }
    if (best < 0) {
        result.reason = "no_feasible_action";
        return result;
    }
    if (currentActionKey_.empty() && !context.previousActionKey.empty()) {
        currentActionKey_ = context.previousActionKey;
        for (size_t i = 0; i < context.candidates.size(); ++i) {
            if (context.candidates[i].action.key() == currentActionKey_) current = static_cast<int>(i);
        }
    }
    if (current < 0 || static_cast<size_t>(current) >= evaluations.size() || !evaluations[current].feasible) {
        currentActionKey_ = context.candidates[best].action.key();
        challengerKey_.clear(); challengerWins_ = 0; dwellRequests_ = 1;
        cooldownRemaining_ = config_.cooldownRequests;
        result.selectedIndex = best;
        result.reason = current < 0 ? "initialize_or_missing_current" : "unsafe_current_immediate_switch";
        result.diagnostics.set("switched", true);
        return result;
    }
    ++dwellRequests_;
    if (cooldownRemaining_ > 0) --cooldownRemaining_;
    if (best == current) {
        challengerKey_.clear(); challengerWins_ = 0;
        result.selectedIndex = current;
        result.reason = "current_still_best";
        result.diagnostics.set("switched", false);
        return result;
    }
    double currentMetric = 0.0, bestMetric = 0.0;
    const bool comparable = context.candidates[current].predicted.getNumber(config_.decisionMetric, currentMetric) &&
                            context.candidates[best].predicted.getNumber(config_.decisionMetric, bestMetric);
    const double gain = comparable ? currentMetric - bestMetric : 0.0;
    const double threshold = comparable
        ? std::max(config_.minimumAbsoluteGain, currentMetric * config_.minimumRelativeGain)
        : std::numeric_limits<double>::infinity();
    const bool eligible = gain > threshold && dwellRequests_ >= config_.minimumDwellRequests &&
                          cooldownRemaining_ == 0;
    const std::string bestKey = context.candidates[best].action.key();
    if (eligible) {
        if (challengerKey_ == bestKey) ++challengerWins_;
        else { challengerKey_ = bestKey; challengerWins_ = 1; }
    } else {
        challengerKey_.clear(); challengerWins_ = 0;
    }
    if (challengerWins_ >= config_.requiredConsecutiveWins) {
        currentActionKey_ = bestKey;
        challengerKey_.clear(); challengerWins_ = 0; dwellRequests_ = 0;
        cooldownRemaining_ = config_.cooldownRequests;
        result.selectedIndex = best;
        result.reason = "hysteresis_threshold_met";
        result.diagnostics.set("switched", true);
    } else {
        result.selectedIndex = current;
        result.reason = eligible ? "waiting_consecutive_wins" : "hysteresis_hold";
        result.diagnostics.set("switched", false);
    }
    result.diagnostics.set("gain", gain);
    result.diagnostics.set("threshold", threshold);
    result.diagnostics.set("challenger_wins", static_cast<int64_t>(challengerWins_));
    result.diagnostics.set("dwell_requests", static_cast<int64_t>(dwellRequests_));
    result.diagnostics.set("cooldown_remaining", static_cast<int64_t>(cooldownRemaining_));
    return result;
}

void HysteresisPolicy::reset() {
    currentActionKey_.clear(); challengerKey_.clear(); challengerWins_ = 0;
    dwellRequests_ = 0; cooldownRemaining_ = 0;
}

void HysteresisPolicy::forceCurrent(const std::string& actionKey) {
    currentActionKey_ = actionKey;
    challengerKey_.clear();
    challengerWins_ = 0;
    dwellRequests_ = 0;
    cooldownRemaining_ = config_.cooldownRequests;
}

void FallbackStateMachine::onEvent(FallbackEvent event) {
    if (event == FallbackEvent::SUCCESS) {
        consecutiveFailures_ = 0;
        if (++consecutiveSuccesses_ >= 3) state_ = FallbackState::NORMAL;
        return;
    }
    consecutiveSuccesses_ = 0;
    if (event == FallbackEvent::NETWORK_DOWN || event == FallbackEvent::VERSION_MISMATCH) {
        state_ = FallbackState::LOCAL_SAFE;
        ++consecutiveFailures_;
    } else if (event == FallbackEvent::THERMAL_HARD) {
        state_ = FallbackState::CLOUD_SAFE;
        ++consecutiveFailures_;
    } else if (event == FallbackEvent::ACTION_FAILURE) {
        ++consecutiveFailures_;
        state_ = consecutiveFailures_ >= 2 ? FallbackState::LOCAL_SAFE : FallbackState::DEGRADED;
    }
}

int FallbackStateMachine::selectFallback(
    const std::vector<ActionCandidate>& candidates, int failedIndex) const {
    if (state_ == FallbackState::LOCAL_SAFE || state_ == FallbackState::CLOUD_SAFE) {
        const ActionMode wanted = state_ == FallbackState::LOCAL_SAFE ? ActionMode::LOCAL : ActionMode::CLOUD;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].action.mode == wanted &&
                numberOr(candidates[i].predicted, "safety.allowed", 1.0) >= 0.5) return static_cast<int>(i);
        }
    }
    if (state_ == FallbackState::DEGRADED && failedIndex >= 0 &&
        static_cast<size_t>(failedIndex) < candidates.size()) {
        const auto& failed = candidates[failedIndex].action;
        int best = -1;
        const int failedRank = codecSafetyRank(failed.codec);
        int bestRank = std::numeric_limits<int>::max();
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& action = candidates[i].action;
            const int rank = codecSafetyRank(action.codec);
            if (action.mode == ActionMode::SPLIT && action.boundary == failed.boundary &&
                rank > failedRank && rank < bestRank &&
                numberOr(candidates[i].predicted, "safety.allowed", 1.0) >= 0.5) {
                best = static_cast<int>(i); bestRank = rank;
            }
        }
        if (best >= 0) return best;
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].action.mode == ActionMode::LOCAL) return static_cast<int>(i);
    }
    return -1;
}

std::string FallbackStateMachine::stateName() const {
    switch (state_) {
        case FallbackState::NORMAL: return "normal";
        case FallbackState::DEGRADED: return "degraded";
        case FallbackState::LOCAL_SAFE: return "local_safe";
        case FallbackState::CLOUD_SAFE: return "cloud_safe";
    }
    return "unknown";
}

void registerRuntimeDecisionPolicies() {
    DecisionPolicyRegistry::instance().registerFactory(
        "hysteresis_profile_v1", [] { return std::make_unique<HysteresisPolicy>(); });
}
