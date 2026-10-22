#ifndef EDGE_AI_ADAPTIVE_RUNTIME_H
#define EDGE_AI_ADAPTIVE_RUNTIME_H

#include "AdaptiveController.h"
#include "RuntimeMetrics.h"

#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct NetworkRuntimeSample {
    double rttMs{0.0};
    double goodputUpBps{0.0};
    double jitterMs{0.0};
    double lossRate{0.0};
    std::string interfaceName{"unknown"};
    bool available{false};
};

struct CloudRuntimeSample {
    double queueMs{0.0};
    double utilizationPct{0.0};
    bool available{false};
};

struct RuntimeHistorySample {
    double latestResidualMs{0.0};
    int64_t consecutiveFallbacks{0};
    std::string previousActionKey;
};

class RuntimeStateAdapter {
public:
    static RuntimeState build(const PerformanceMetrics& device,
                              const NetworkRuntimeSample& network,
                              const CloudRuntimeSample& cloud,
                              const RuntimeHistorySample& history,
                              const std::string& requestId,
                              const std::string& modelName,
                              const std::string& modelHash,
                              const std::string& modelVersion);
};

struct ComponentProfile {
    SplitAction action;
    double edgePrefixMs{0.0};
    double captureMs{0.0};
    double encodeMs{0.0};
    double decodeMs{0.0};
    double injectMs{0.0};
    double cloudSuffixMs{0.0};
    double fixedTransferMs{0.0};
    double offlineP95PaddingMs{0.0};
    int64_t rawBytes{0};
    int64_t encodedBytes{0};
    int64_t bundleBytes{0};
    double accuracyDrop{0.0};
    double peakMemoryMb{0.0};
    double energyMj{0.0};
    double failureProbability{0.0};
    double referenceCpuFrequencyMhz{0.0};
    double cpuUtilSensitivity{0.0};
    double referenceCloudUtilPct{0.0};
    double cloudUtilSensitivity{0.0};
    MetricBag extensions;
};

struct ProfileManifest {
    uint32_t schemaVersion{1};
    std::string manifestId;
    std::string modelName;
    std::string modelHash;
    std::string modelVersion;
    std::vector<ComponentProfile> profiles;
};

class OfflineProfileStore {
public:
    bool replace(const ProfileManifest& manifest, std::string* error = nullptr);
    bool getManifest(const std::string& modelHash, ProfileManifest& manifest) const;
    std::vector<ComponentProfile> profiles(const std::string& modelHash) const;
    bool invalidate(const std::string& modelHash);

private:
    mutable std::mutex mutex_;
    std::map<std::string, ProfileManifest> manifests_;
};

struct ComponentPredictorConfig {
    double controlRttFactor{1.0};
    double lossPenaltyFactor{2.0};
    double minimumGoodputBps{1024.0};
    double unavailableNetworkPenaltyMs{1.0e9};
};

class ComponentCostPredictor {
public:
    explicit ComponentCostPredictor(ComponentPredictorConfig config = {}) : config_(config) {}
    ActionCandidate predict(const ComponentProfile& profile, const RuntimeState& state) const;
    std::vector<ActionCandidate> predictAll(const std::vector<ComponentProfile>& profiles,
                                            const RuntimeState& state) const;

private:
    ComponentPredictorConfig config_;
};

struct ResidualConfig {
    double alpha{0.15};
    double abruptAlpha{0.35};
    std::size_t windowSize{64};
    std::size_t minimumOnlineSamples{16};
    double coldStartPaddingMs{3.0};
};

struct ResidualSnapshot {
    std::size_t samples{0};
    double ewmaMs{0.0};
    double positiveP95Ms{0.0};
};

class OnlineResidualModel {
public:
    explicit OnlineResidualModel(ResidualConfig config = {}) : config_(config) {}
    void update(const std::string& actionKey, const std::string& component,
                double predictedMs, double actualMs, bool abrupt = false);
    ResidualSnapshot snapshot(const std::string& actionKey, const std::string& component) const;
    double correctedMean(const std::string& actionKey, const std::string& component,
                         double offlineMs) const;
    double correctedP95(const std::string& actionKey, const std::string& component,
                        double offlineMs, double offlinePaddingMs = 0.0) const;
    void updateFromObservation(const SplitAction& action, const MetricBag& predicted,
                               const MetricBag& actual, bool networkAbrupt = false);
    void migrateAction(const std::string& oldActionKey, const std::string& newActionKey,
                       double computeFactor, double codecFactor, double networkFactor = 1.0);
    void eraseAction(const std::string& actionKey);

private:
    struct Series {
        std::size_t samples{0};
        double ewmaMs{0.0};
        std::deque<double> positiveResiduals;
    };
    std::string key(const std::string& actionKey, const std::string& component) const;
    ResidualConfig config_;
    mutable std::mutex mutex_;
    std::map<std::string, Series> series_;
};

class OnlineCorrectedPredictor {
public:
    OnlineCorrectedPredictor(ComponentCostPredictor base,
                             std::shared_ptr<OnlineResidualModel> residuals)
        : base_(std::move(base)), residuals_(std::move(residuals)) {}
    ActionCandidate predict(const ComponentProfile& profile, const RuntimeState& state) const;
    void observe(const ActionCandidate& prediction, const MetricBag& actual,
                 bool networkAbrupt = false);

private:
    ComponentCostPredictor base_;
    std::shared_ptr<OnlineResidualModel> residuals_;
};

struct SafetyConfig {
    double maxAccuracyDrop{0.01};
    double maxPeakMemoryMb{1.0e12};
    double hardSocTemperatureC{85.0};
    double maxFailureProbability{0.25};
    bool requireModelHash{true};
};

struct SafetyAssessment {
    bool allowed{true};
    std::vector<std::string> reasons;
};

class SafetyFilter {
public:
    explicit SafetyFilter(SafetyConfig config = {}) : config_(config) {}
    SafetyAssessment assess(const RuntimeState& state, const ActionCandidate& candidate) const;
    void annotate(const RuntimeState& state, std::vector<ActionCandidate>& candidates) const;

private:
    SafetyConfig config_;
};

struct HysteresisConfig {
    std::string decisionMetric{"latency.p95_ms"};
    double minimumAbsoluteGain{5.0};
    double minimumRelativeGain{0.08};
    int requiredConsecutiveWins{3};
    int minimumDwellRequests{5};
    int cooldownRequests{3};
};

class HysteresisPolicy final : public IDecisionPolicy {
public:
    explicit HysteresisPolicy(HysteresisConfig config = {}) : config_(std::move(config)) {}
    std::string name() const override { return "hysteresis_profile_v1"; }
    PolicyDecision select(const PolicyContext& context,
                          const std::vector<CandidateEvaluation>& evaluations) override;
    void reset();
    void forceCurrent(const std::string& actionKey);

private:
    HysteresisConfig config_;
    std::string currentActionKey_;
    std::string challengerKey_;
    int challengerWins_{0};
    int dwellRequests_{0};
    int cooldownRemaining_{0};
};

enum class FallbackState { NORMAL, DEGRADED, LOCAL_SAFE, CLOUD_SAFE };
enum class FallbackEvent { SUCCESS, ACTION_FAILURE, NETWORK_DOWN, THERMAL_HARD, VERSION_MISMATCH };

class FallbackStateMachine {
public:
    void onEvent(FallbackEvent event);
    FallbackState state() const { return state_; }
    int consecutiveFailures() const { return consecutiveFailures_; }
    int selectFallback(const std::vector<ActionCandidate>& candidates,
                       int failedIndex = -1) const;
    std::string stateName() const;

private:
    FallbackState state_{FallbackState::NORMAL};
    int consecutiveFailures_{0};
    int consecutiveSuccesses_{0};
};

void registerRuntimeDecisionPolicies();

#endif
