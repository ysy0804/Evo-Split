#ifndef EDGE_AI_PROFILE_FREE_CONTROLLER_H
#define EDGE_AI_PROFILE_FREE_CONTROLLER_H

#include "AdaptiveController.h"
#include "VersionedUpdateManager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

enum class ControllerBootstrapMode { OFFLINE_PROFILE, PROFILE_FREE, HYBRID };
enum class ProfileSource { NONE, OPTIONAL_OFFLINE, INHERITED_VERSION, ONLINE_OBSERVED };
enum class AdaptationPhase { UNINITIALIZED, ANCHORING, SAFE_EXPLORING, ADAPTING, STABLE, DRIFTED };

const char* controllerBootstrapModeName(ControllerBootstrapMode value);
const char* profileSourceName(ProfileSource value);
const char* adaptationPhaseName(AdaptationPhase value);
bool parseControllerBootstrapMode(const std::string& text, ControllerBootstrapMode& value);

struct StructuralBoundaryDescriptor {
    std::string canonicalBoundaryId;
    int physicalBoundary{-1};
    double prefixFlops{0.0};
    double suffixFlops{0.0};
    uint64_t rawBundleBytes{0};
    double peakEdgeMemoryMb{0.0};
    int boundaryDepth{0};
    int cutSetTensorCount{1};
    bool versionQualified{true};
    bool structurallyCompatible{true};
    std::set<CodecKind> supportedCodecs;
    std::map<CodecKind, double> compressionRatios;
    std::map<CodecKind, double> accuracyDrops;
};

struct StructuralModelManifest {
    uint32_t schemaVersion{1};
    std::string manifestId;
    std::string modelName;
    std::string modelHash;
    std::string modelVersion;
    std::string structureHash;
    uint64_t inputBytes{0};
    double totalFlops{0.0};
    double localPeakMemoryMb{0.0};
    std::vector<StructuralBoundaryDescriptor> boundaries;

    bool valid(std::string* error = nullptr) const;
};

struct StructuralActionDescriptor {
    SplitAction action;
    std::string canonicalBoundaryId;
    double prefixFlops{0.0};
    double suffixFlops{0.0};
    uint64_t rawBundleBytes{0};
    uint64_t estimatedEncodedBytes{0};
    double peakEdgeMemoryMb{0.0};
    int boundaryDepth{0};
    int cutSetTensorCount{0};
    double estimatedAccuracyDrop{0.0};
    bool versionQualified{true};
    bool codecQualified{true};
    bool structurallyCompatible{true};
    ProfileSource source{ProfileSource::NONE};

    std::string stableKey() const;
};

struct ActionBuildResult {
    std::vector<StructuralActionDescriptor> actions;
    std::size_t rawActions{0};
    std::size_t rejectedVersion{0};
    std::size_t rejectedCodec{0};
    double buildMs{0.0};
};

class StructuralActionBuilder {
public:
    ActionBuildResult build(const StructuralModelManifest& manifest) const;
};

struct CandidatePruningConfig {
    bool structuralPruningEnabled{true};
    std::size_t maximumShortlistSize{8};
    double lowerBoundSlack{0.30};
    double maximumAccuracyDrop{0.01};
    double maximumFailureProbability{0.25};
    bool keepRepresentativeBoundaries{true};
    bool keepPreviousSafeAction{true};
    bool retainAuditRecords{false};
};

struct PrunedActionRecord {
    StructuralActionDescriptor descriptor;
    bool kept{false};
    double lowerBoundMs{0.0};
    std::string reason;
};

struct CandidatePruningResult {
    std::size_t inputActions{0};
    std::vector<StructuralActionDescriptor> shortlist;
    std::vector<PrunedActionRecord> records;
    std::size_t hardRejected{0};
    std::size_t lowerBoundRejected{0};
    double pruneMs{0.0};
};

class CandidatePruner {
public:
    explicit CandidatePruner(CandidatePruningConfig config = {}) : config_(config) {}
    CandidatePruningResult prune(const RuntimeState& state,
                                 const std::vector<StructuralActionDescriptor>& actions,
                                 const std::string& previousSafeActionKey = "") const;

private:
    CandidatePruningConfig config_;
};

struct OnlineEstimatorConfig {
    bool componentSharingEnabled{true};
    double initialEdgeFlopsPerMs{1.0e6};
    double initialCloudFlopsPerMs{1.0e7};
    double initialCodecBytesPerMs{2.0e6};
    double initialCaptureBytesPerMs{5.0e6};
    double initialInjectBytesPerMs{5.0e6};
    double ewmaAlpha{0.25};
    double coldStartRelativeUncertainty{0.50};
    double minimumUncertaintyMs{1.0};
    double p95SigmaMultiplier{1.645};
    bool retainComponentDetails{false};
};

struct ComponentEstimate {
    double meanMs{0.0};
    double p95Ms{0.0};
    double uncertaintyMs{0.0};
    std::size_t samples{0};
    bool coldStart{true};
};

struct ProfileFreePrediction {
    ActionCandidate candidate;
    std::string structuralActionKey;
    ComponentEstimate total;
    // Cross-context evidence used only to rank the bounded recalibration
    // probe; it does not make a new network context "sampled".
    std::size_t historicalSamples{0};
    std::map<std::string, ComponentEstimate> components;
    ProfileSource source{ProfileSource::NONE};
};

struct EstimatorObservation {
    std::string deviceCapabilityHash;
    StructuralActionDescriptor action;
    RuntimeState state;
    MetricBag actual;
    bool success{true};
    bool abrupt{false};
};

class OnlineComponentEstimator {
public:
    explicit OnlineComponentEstimator(OnlineEstimatorConfig config = {}) : config_(config) {}
    void setConfig(const OnlineEstimatorConfig& config);
    ProfileFreePrediction predict(const std::string& deviceCapabilityHash,
                                  const StructuralActionDescriptor& action,
                                  const RuntimeState& state) const;
    std::vector<ProfileFreePrediction> predictBatch(
        const std::string& deviceCapabilityHash,
        const std::vector<StructuralActionDescriptor>& actions,
        const RuntimeState& state) const;
    void observe(const EstimatorObservation& observation);
    void migrateAction(const std::string& deviceCapabilityHash,
                       const std::string& oldStableKey,
                       const std::string& newStableKey,
                       double reuseFactor);
    void eraseAction(const std::string& deviceCapabilityHash, const std::string& stableKey);
    std::size_t actionSamples(const std::string& deviceCapabilityHash,
                              const std::string& stableKey) const;

private:
    struct ScalarSeries {
        std::size_t samples{0};
        double mean{0.0};
        double variance{0.0};
    };
    struct DeviceState {
        ScalarSeries edgeFlopsPerMs;
        ScalarSeries cloudFlopsPerMs;
        std::map<CodecKind, ScalarSeries> encodeBytesPerMs;
        std::map<CodecKind, ScalarSeries> decodeBytesPerMs;
        ScalarSeries captureBytesPerMs;
        ScalarSeries injectBytesPerMs;
        std::map<std::string, ScalarSeries> actionResidualMs;
        std::map<std::string, std::size_t> actionSamples;
    };
    static void updateSeries(ScalarSeries& series, double value, double alpha);
    static void updateSignedSeries(ScalarSeries& series, double value, double alpha);
    static double seriesMean(const ScalarSeries& series, double fallback);
    ComponentEstimate estimate(double mean, const ScalarSeries& series) const;
    ProfileFreePrediction predictUnlocked(const std::string& deviceCapabilityHash,
                                          const StructuralActionDescriptor& action,
                                          const RuntimeState& state) const;
    OnlineEstimatorConfig config_;
    mutable std::mutex mutex_;
    std::map<std::string, DeviceState> devices_;
};

struct AdaptationConfig {
    bool historyGuidedCalibrationEnabled{true};
    int maximumProbeRequests{5};
    int maximumConsecutiveProbes{1};
    double maximumProbeOverheadMs{100.0};
    double maximumAllowedP95Ms{1.0e12};
    double convergenceRelativeError{0.10};
    int stableWindowRequests{5};
    int driftFailures{2};
    int changeProbeRequests{2};
};

struct AdaptationSnapshot {
    AdaptationPhase phase{AdaptationPhase::UNINITIALIZED};
    int probesUsed{0};
    int consecutiveProbes{0};
    int observations{0};
    int stableObservations{0};
    int failures{0};
    int environmentChanges{0};
    double cumulativeRegretMs{0.0};
    std::string lastSafeActionKey;
};

class AdaptationManager {
public:
    explicit AdaptationManager(AdaptationConfig config = {}) : config_(config) {}
    void setConfig(const AdaptationConfig& config);
    AdaptationSnapshot snapshot(const std::string& deviceCapabilityHash,
                                const std::string& modelHash) const;
    int selectProbe(const std::string& deviceCapabilityHash, const std::string& modelHash,
                    const std::vector<ProfileFreePrediction>& predictions,
                    int normalSelection) const;
    void observe(const std::string& deviceCapabilityHash, const std::string& modelHash,
                 const ProfileFreePrediction& prediction, const MetricBag& actual,
                 bool success, bool wasProbe, double oracleMs = -1.0);
    void markDrift(const std::string& deviceCapabilityHash, const std::string& modelHash);
    void markEnvironmentChange(const std::string& deviceCapabilityHash,
                               const std::string& modelHash,
                               int probeRequests = -1);
    void migrateModel(const std::string& deviceCapabilityHash,
                      const std::string& oldModelHash, const std::string& newModelHash,
                      double reuseFactor);

private:
    std::string key(const std::string& device, const std::string& model) const;
    AdaptationConfig config_;
    mutable std::mutex mutex_;
    std::map<std::string, AdaptationSnapshot> states_;
};

struct ProfileFreeControllerConfig {
    bool enabled{true};
    ControllerBootstrapMode bootstrapMode{ControllerBootstrapMode::HYBRID};
    bool shadowMode{true};
    bool explorationEnabled{true};
    double riskMultiplier{1.0};
    CandidatePruningConfig pruning;
    OnlineEstimatorConfig estimator;
    AdaptationConfig adaptation;
};

struct ProfileFreeDecision {
    uint32_t schemaVersion{1};
    std::string traceId;
    std::string requestId;
    std::string deviceCapabilityHash;
    std::string modelHash;
    std::vector<ProfileFreePrediction> predictions;
    CandidatePruningResult pruning;
    int selectedIndex{-1};
    bool selectedAsProbe{false};
    bool environmentChanged{false};
    std::string environmentKey;
    bool shadowMode{true};
    ControllerBootstrapMode bootstrapMode{ControllerBootstrapMode::PROFILE_FREE};
    std::string reason;
    AdaptationSnapshot adaptation;
    double actionBuildMs{0.0};
    double predictionMs{0.0};
    double selectionMs{0.0};
    double totalDecisionMs{0.0};
    std::string toJson() const;
};

struct ProfileFreeMigrationResult;

class ProfileFreeRuntime {
public:
    explicit ProfileFreeRuntime(ProfileFreeControllerConfig config = {});
    void setConfig(const ProfileFreeControllerConfig& config);
    bool registerManifest(const StructuralModelManifest& manifest, std::string* error = nullptr);
    bool hasManifest(const std::string& modelHash) const;
    ProfileFreeDecision decide(const std::string& deviceCapabilityHash,
                               const RuntimeState& state);
    bool observe(const std::string& traceId, const MetricBag& actual, bool success,
                 bool abrupt = false, double oracleMs = -1.0,
                 std::string* error = nullptr);
    std::string statusJson(const std::string& deviceCapabilityHash,
                           const std::string& modelHash) const;
    ProfileFreeMigrationResult migrateVersion(const std::string& deviceCapabilityHash,
                                              const std::string& oldModelHash,
                                              const std::string& newModelHash,
                                              const CompatibilityMap& compatibility,
                                              std::string* error = nullptr);
    ProfileFreeMigrationResult migrateKnownDevices(const std::string& oldModelHash,
                                                   const std::string& newModelHash,
                                                   const CompatibilityMap& compatibility,
                                                   std::string* error = nullptr);
    OnlineComponentEstimator& estimator() { return estimator_; }
    AdaptationManager& adaptation() { return adaptation_; }

private:
    struct PendingDecision {
        std::string deviceCapabilityHash;
        std::string modelHash;
        ProfileFreePrediction prediction;
        StructuralActionDescriptor descriptor;
        RuntimeState state;
        bool wasProbe{false};
    };
    ProfileFreeControllerConfig config_;
    StructuralActionBuilder builder_;
    CandidatePruner pruner_;
    OnlineComponentEstimator estimator_;
    AdaptationManager adaptation_;
    mutable std::mutex mutex_;
    std::map<std::string, StructuralModelManifest> manifests_;
    std::map<std::string, std::shared_ptr<const ActionBuildResult>> actionCache_;
    std::map<std::string, std::string> environmentKeys_;
    std::map<std::string, double> environmentQuality_;
    std::map<std::string, PendingDecision> pending_;
    std::set<std::string> knownDevices_;
};

struct ProfileFreeMigrationResult {
    std::size_t exactReused{0};
    std::size_t compatibleReused{0};
    std::size_t calibrationReset{0};
    std::size_t incompatibleRemoved{0};
    std::size_t requiredProbes{0};
    double migrationMs{0.0};
    std::string toJson() const;
};

class OnlineStateMigrationManager {
public:
    static ProfileFreeMigrationResult migrate(
        const std::string& deviceCapabilityHash,
        const StructuralModelManifest& oldManifest,
        const StructuralModelManifest& newManifest,
        const CompatibilityMap& compatibility,
        OnlineComponentEstimator& estimator,
        AdaptationManager& adaptation);
};

#endif
