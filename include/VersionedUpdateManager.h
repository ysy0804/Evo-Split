#ifndef EDGE_AI_VERSIONED_UPDATE_MANAGER_H
#define EDGE_AI_VERSIONED_UPDATE_MANAGER_H

#include "CodecTypes.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

enum class UpdateMode { OFF, AUDIT, SHADOW, ENFORCE, SAFE_ONLY };
enum class ImpactLevel { I0_NONE, I1_PARAMETERS, I2_LOCAL_COMPUTE, I3_INTERFACE, I4_GLOBAL };
enum class MappingKind { EXACT, CONTRACT_COMPATIBLE, CALIBRATION_REQUIRED, INCOMPATIBLE };
enum class QualificationState { UNAVAILABLE, PROFILING, SHADOW_READY, ACTIVE, QUARANTINED };
enum class UpdateTransactionState { IDLE, PREPARING, PREPARED, COMMITTED, ABORTED, ROLLED_BACK };

const char* updateModeName(UpdateMode value);
const char* impactLevelName(ImpactLevel value);
const char* mappingKindName(MappingKind value);
const char* qualificationStateName(QualificationState value);
const char* updateTransactionStateName(UpdateTransactionState value);
bool parseUpdateMode(const std::string& text, UpdateMode& value);

struct UpdateConfig {
    UpdateMode mode{UpdateMode::SHADOW};
    bool diffEnabled{true};
    bool minimalInvalidationEnabled{true};
    bool compatibilityMappingEnabled{true};
    bool stateMigrationEnabled{true};
    bool incrementalQualificationEnabled{true};
    bool autoCommitEnabled{false};
    bool autoRollbackEnabled{true};
    double exactMappingThreshold{0.999};
    double compatibleMappingThreshold{0.85};
    std::size_t qualificationMinimumSamples{16};
    double qualificationMaxMaeMs{10.0};
    double qualificationMaxAccuracyDrop{0.01};
};

struct FileMerkleIdentity {
    uint32_t schemaVersion{1};
    std::string algorithm{"sha256-merkle-v1"};
    uint64_t fileBytes{0};
    uint64_t chunkBytes{4u * 1024u * 1024u};
    std::vector<std::string> chunkHashes;
    std::string rootHash;
};

class VersionHasher {
public:
    static std::string sha256(const void* data, std::size_t size);
    static std::string sha256(const std::string& text);
    static bool hashFileMerkle(const std::string& path, FileMerkleIdentity& result,
                               std::string* error = nullptr,
                               std::size_t chunkBytes = 4u * 1024u * 1024u);
};

struct VersionedTensorContract {
    std::string stableId;
    std::string tensorName;
    std::vector<int64_t> shape;
    std::string dtype{"float32"};
    std::string layout{"unknown"};

    std::string signature() const;
};

struct VersionedBoundary {
    std::string canonicalId;
    std::string physicalId;
    int splitAfterOp{-1};
    int totalOps{0};
    std::vector<VersionedTensorContract> cutSet;
    std::string upstreamNeighborhoodHash;
    std::string downstreamNeighborhoodHash;
    std::set<CodecKind> supportedCodecs;
    std::set<std::string> dependentProfileIds;

    std::string contractSignature() const;
    std::string semanticFingerprint() const;
};

struct ModelVersionManifestV4 {
    uint32_t schemaVersion{1};
    std::string modelName;
    std::string modelVersion;
    std::string modelPath;
    std::string structureHash;
    std::string contentHash;
    std::string manifestId;
    uint32_t tensorPacketProtocolVersion{3};
    std::map<std::string, std::string> artifactHashes;
    std::vector<VersionedBoundary> boundaries;

    bool valid(std::string* error = nullptr) const;
    const VersionedBoundary* findBoundary(const std::string& canonicalId) const;
};

struct ModelChangeSet {
    uint32_t schemaVersion{1};
    std::string oldContentHash;
    std::string newContentHash;
    bool contentChanged{false};
    bool structureChanged{false};
    bool protocolChanged{false};
    bool configChanged{false};
    std::set<std::string> changedArtifacts;
    std::set<std::string> addedBoundaries;
    std::set<std::string> removedBoundaries;
    std::set<std::string> changedBoundaryContracts;
    std::set<std::string> changedBoundaryNeighborhoods;
    ImpactLevel preliminaryLevel{ImpactLevel::I0_NONE};
    double diffMs{0.0};

    std::set<std::string> dependencySeeds() const;
    std::string toJson() const;
};

class ModelDiffEngine {
public:
    static ModelChangeSet compare(const ModelVersionManifestV4& oldModel,
                                  const ModelVersionManifestV4& newModel);
};

enum class DecisionArtifactKind {
    ACTION, PREFIX_PROFILE, SUFFIX_PROFILE, CODEC_PROFILE,
    ACCURACY_PROFILE, RESIDUAL, HYSTERESIS_STATE
};

struct DecisionArtifactDependency {
    std::string artifactId;
    DecisionArtifactKind kind{DecisionArtifactKind::ACTION};
    std::string actionKey;
    std::string canonicalBoundaryId;
    std::set<std::string> dependencies;
};

struct ImpactPlan {
    uint32_t schemaVersion{1};
    ImpactLevel level{ImpactLevel::I0_NONE};
    std::set<std::string> reusableArtifacts;
    std::set<std::string> invalidatedArtifacts;
    std::set<std::string> requiredProbes;
    std::set<std::string> reusableActions;
    std::set<std::string> invalidatedActions;
    double closureRatio{0.0};
    double analysisMs{0.0};
    std::string reason;
    std::string toJson() const;
};

class DecisionDependencyGraph {
public:
    bool add(const DecisionArtifactDependency& artifact);
    void clear();
    std::size_t size() const;
    ImpactPlan analyze(const ModelChangeSet& changes, bool minimalInvalidation = true) const;

private:
    std::map<std::string, DecisionArtifactDependency> artifacts_;
};

struct BoundaryMapping {
    std::string oldCanonicalId;
    std::string oldPhysicalId;
    std::string newCanonicalId;
    std::string newPhysicalId;
    MappingKind kind{MappingKind::INCOMPATIBLE};
    double score{0.0};
    bool adapterRequired{false};
    std::string reason;
};

struct CompatibilityMap {
    uint32_t schemaVersion{1};
    std::string oldModelHash;
    std::string newModelHash;
    std::vector<BoundaryMapping> boundaries;
    double mappingMs{0.0};
    std::string toJson() const;
};

class BoundaryCompatibilityMapper {
public:
    static CompatibilityMap build(const ModelVersionManifestV4& oldModel,
                                  const ModelVersionManifestV4& newModel,
                                  const UpdateConfig& config = {});
};

struct StateMigrationEntry {
    std::string stateKey;
    std::string oldActionKey;
    std::string newActionKey;
    double reuseFactor{0.0};
    bool reset{true};
    std::string reason;
};

struct StateMigrationPlan {
    uint32_t schemaVersion{1};
    std::vector<StateMigrationEntry> entries;
    std::size_t reusedCount{0};
    std::size_t resetCount{0};
    double averageReuseFactor{0.0};
    bool hysteresisMigrated{false};
    double migrationMs{0.0};
    std::string toJson() const;
};

class ControllerStateMigrator {
public:
    static StateMigrationPlan plan(const CompatibilityMap& mappings,
                                   const ModelChangeSet& changes,
                                   bool enabled = true);
};

struct QualificationRecord {
    QualificationState state{QualificationState::UNAVAILABLE};
    std::size_t samples{0};
    std::size_t failures{0};
    double absoluteErrorSumMs{0.0};
    double maximumAccuracyDrop{0.0};
};

class IncrementalQualificationManager {
public:
    explicit IncrementalQualificationManager(UpdateConfig config = {}) : config_(config) {}
    void setConfig(const UpdateConfig& config);
    void seed(const std::string& actionKey, MappingKind mapping);
    QualificationState observe(const std::string& actionKey, bool success,
                               double absolutePredictionErrorMs,
                               double accuracyDrop);
    QualificationRecord get(const std::string& actionKey) const;
    bool activate(const std::string& actionKey);
    void quarantine(const std::string& actionKey);
    void clearModel(const std::string& modelHashFragment);

private:
    UpdateConfig config_;
    mutable std::mutex mutex_;
    std::map<std::string, QualificationRecord> records_;
};

struct UpdateTransaction {
    uint32_t schemaVersion{1};
    std::string transactionId;
    UpdateTransactionState state{UpdateTransactionState::IDLE};
    ModelVersionManifestV4 stagedManifest;
    ModelChangeSet changes;
    ImpactPlan impact;
    CompatibilityMap compatibility;
    StateMigrationPlan migration;
    bool edgeReady{false};
    bool cloudReady{false};
    std::string failureReason;
    uint64_t startedUnixMs{0};
    uint64_t updatedUnixMs{0};
};

class VersionedUpdateCoordinator {
public:
    explicit VersionedUpdateCoordinator(UpdateConfig config = {}) : config_(config) {}
    void setConfig(const UpdateConfig& config);
    UpdateConfig config() const;
    bool bootstrap(const ModelVersionManifestV4& active, std::string* error = nullptr);
    bool prepare(const std::string& transactionId,
                 const ModelVersionManifestV4& staged,
                 const ModelChangeSet& changes,
                 const ImpactPlan& impact,
                 const CompatibilityMap& compatibility,
                 const StateMigrationPlan& migration,
                 std::string* error = nullptr);
    bool markReady(const std::string& transactionId, bool edge, bool cloud,
                   std::string* error = nullptr);
    bool commit(const std::string& transactionId, std::string* error = nullptr);
    bool abort(const std::string& transactionId, const std::string& reason,
               std::string* error = nullptr);
    bool rollback(const std::string& reason, std::string* error = nullptr);
    bool splitCompatible(const std::string& edgeContentHash,
                         const std::string& cloudContentHash,
                         const std::string& manifestId,
                         uint32_t packetProtocolVersion,
                         std::string* reason = nullptr) const;
    bool splitAllowed() const;
    bool hasActive() const;
    ModelVersionManifestV4 active() const;
    ModelVersionManifestV4 staged() const;
    ModelVersionManifestV4 rollbackSlot() const;
    UpdateTransaction transaction() const;
    std::string statusJson() const;

private:
    mutable std::mutex mutex_;
    UpdateConfig config_;
    bool hasActive_{false};
    bool hasStaged_{false};
    bool hasRollback_{false};
    ModelVersionManifestV4 active_;
    ModelVersionManifestV4 staged_;
    ModelVersionManifestV4 rollback_;
    UpdateTransaction transaction_;
};

struct UpdateAuditRecord {
    uint32_t schemaVersion{1};
    uint64_t timestampUnixMs{0};
    std::string transactionId;
    std::string event;
    std::string oldModelHash;
    std::string newModelHash;
    std::string detailJson{"{}"};
    double durationMs{0.0};
    std::string toJson() const;
};

class IUpdateAuditSink {
public:
    virtual ~IUpdateAuditSink() = default;
    virtual bool append(const UpdateAuditRecord& record) = 0;
};

class JsonlUpdateAuditSink final : public IUpdateAuditSink {
public:
    explicit JsonlUpdateAuditSink(std::string path) : path_(std::move(path)) {}
    bool append(const UpdateAuditRecord& record) override;
private:
    std::mutex mutex_;
    std::string path_;
};

#endif
