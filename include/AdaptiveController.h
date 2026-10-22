#ifndef EDGE_AI_ADAPTIVE_CONTROLLER_H
#define EDGE_AI_ADAPTIVE_CONTROLLER_H

#include "CodecTypes.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

// ============================================================
// Metric values: integer, floating point, Boolean, or string.
// ============================================================
using MetricValue = std::variant<int64_t, double, bool, std::string>;

enum class MetricType { INTEGER, NUMBER, BOOLEAN, STRING };

// ============================================================
// Metric definition.
// ============================================================
struct MetricDefinition {
    std::string name;
    MetricType type{MetricType::NUMBER};
    std::string unit;
    std::string description;
    bool required{false};
};

// ============================================================
// Metric bag.
// ============================================================
class MetricBag {
public:
    void set(const std::string& name, int64_t value);
    void set(const std::string& name, double value);
    void set(const std::string& name, bool value);
    void set(const std::string& name, const std::string& value);
    void set(const std::string& name, const char* value);
    bool contains(const std::string& name) const;
    bool getNumber(const std::string& name, double& value) const;
    bool getString(const std::string& name, std::string& value) const;
    const std::map<std::string, MetricValue>& values() const { return values_; }
    std::string toJson() const;

private:
    std::map<std::string, MetricValue> values_;
};


// ============================================================
// Metric schema.
// ============================================================
class MetricSchema {
public:
    explicit MetricSchema(uint32_t version = 1) : version_(version) {}
    bool add(const MetricDefinition& definition);
    bool validate(const MetricBag& bag, std::vector<std::string>& errors) const;
    uint32_t version() const { return version_; }
    const std::map<std::string, MetricDefinition>& definitions() const { return definitions_; }

private:
    uint32_t version_{1};
    std::map<std::string, MetricDefinition> definitions_;
};


MetricSchema buildDefaultRuntimeStateSchema();
MetricSchema buildDefaultPredictionSchema();

enum class ActionMode { LOCAL, CLOUD, SPLIT };

struct SplitAction {
    ActionMode mode{ActionMode::LOCAL};
    int boundary{-1};
    CodecKind codec{CodecKind::RAW32};
    uint32_t blockSize{0};
    std::string modelHash;
    std::string manifestId;

    std::string key() const;
};

struct RuntimeState {
    uint32_t schemaVersion{1};
    uint64_t timestampUnixMs{0};
    std::string requestId;
    std::string modelName;
    std::string modelHash;
    std::string modelVersion;
    MetricBag features;
};

struct ActionCandidate {
    SplitAction action;
    MetricBag predicted;
    MetricBag annotations;
};

struct CandidateManifest {
    uint32_t schemaVersion{1};
    std::string manifestId;
    std::string modelName;
    std::string modelHash;
    std::string modelVersion;
    std::vector<ActionCandidate> candidates;
    MetricBag metadata;
};

class CandidateRegistry {
public:
    bool upsert(const CandidateManifest& manifest, std::string* error = nullptr);
    bool get(const std::string& modelHash, CandidateManifest& manifest) const;
    bool invalidate(const std::string& modelHash);
    std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, CandidateManifest> manifests_;
};

enum class ObjectiveDirection { MINIMIZE, MAXIMIZE };

struct ObjectiveTerm {
    std::string metric;
    double weight{1.0};
    double scale{1.0};
    ObjectiveDirection direction{ObjectiveDirection::MINIMIZE};
};

enum class ConstraintOperator { LESS_EQUAL, GREATER_EQUAL, EQUAL };

struct ConstraintSpec {
    std::string id;
    std::string metric;
    ConstraintOperator op{ConstraintOperator::LESS_EQUAL};
    double threshold{0.0};
    double tolerance{0.0};
    bool hard{true};
    double penalty{0.0};
};

struct OptimizationSpec {
    std::string id{"p95_latency_v1"};
    uint32_t version{1};
    std::vector<ObjectiveTerm> objectives;
    std::vector<ConstraintSpec> constraints;
};

struct CandidateEvaluation {
    bool feasible{false};
    double score{0.0};
    std::vector<std::string> violations;
    MetricBag termContributions;
};


// ============================================================
// Optimization evaluator interface.
// ============================================================
class IOptimizationEvaluator {
public:
    virtual ~IOptimizationEvaluator() = default;
    virtual CandidateEvaluation evaluate(const RuntimeState& state,
                                         const ActionCandidate& candidate) const = 0;
    virtual std::string id() const = 0;
    virtual uint32_t version() const = 0;
    virtual std::string describeJson() const = 0;
};


// ============================================================
// Configurable optimization evaluator.
// ============================================================
class ConfigurableOptimizationEvaluator final : public IOptimizationEvaluator {
public:
    explicit ConfigurableOptimizationEvaluator(OptimizationSpec spec);
    CandidateEvaluation evaluate(const RuntimeState& state,
                                 const ActionCandidate& candidate) const override;
    std::string id() const override { return spec_.id; }
    uint32_t version() const override { return spec_.version; }
    std::string describeJson() const override;
    const OptimizationSpec& spec() const { return spec_; }
    void setSpec(OptimizationSpec spec) { spec_ = std::move(spec); }

private:
    OptimizationSpec spec_;
};

struct PolicyContext {
    RuntimeState state;
    std::vector<ActionCandidate> candidates;
    std::string previousActionKey;
};

struct PolicyDecision {
    int selectedIndex{-1};
    std::string reason;
    MetricBag diagnostics;
};

class IDecisionPolicy {
public:
    virtual ~IDecisionPolicy() = default;
    virtual std::string name() const = 0;
    virtual PolicyDecision select(const PolicyContext& context,
                                  const std::vector<CandidateEvaluation>& evaluations) = 0;
};

class GreedyProfilePolicy final : public IDecisionPolicy {
public:
    std::string name() const override { return "greedy_profile_v1"; }
    PolicyDecision select(const PolicyContext& context,
                          const std::vector<CandidateEvaluation>& evaluations) override;
};

class DecisionPolicyRegistry {
public:
    using Factory = std::function<std::unique_ptr<IDecisionPolicy>()>;
    static DecisionPolicyRegistry& instance();
    bool registerFactory(const std::string& name, Factory factory);
    std::unique_ptr<IDecisionPolicy> create(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, Factory> factories_;
};

struct DecisionTiming {
    uint64_t startSteadyNs{0};
    uint64_t endSteadyNs{0};
    double evaluationMs{0.0};
    double policyMs{0.0};
    double traceSerializationMs{0.0};
    double tracePersistMs{0.0};
    double totalMs{0.0};
};

struct DecisionTrace {
    uint32_t schemaVersion{1};
    std::string traceId;
    std::string requestId;
    std::string algorithm;
    std::string optimizationSpecId;
    uint32_t optimizationSpecVersion{0};
    std::string optimizationSpecJson;
    bool shadowMode{true};
    std::string previousActionKey;
    RuntimeState state;
    std::vector<ActionCandidate> candidates;
    std::vector<CandidateEvaluation> evaluations;
    int selectedIndex{-1};
    std::string selectedActionKey;
    std::string reason;
    MetricBag policyDiagnostics;
    DecisionTiming timing;

    std::string toJson() const;
};

class IDecisionTraceSink {
public:
    virtual ~IDecisionTraceSink() = default;
    virtual bool append(const std::string& jsonLine) = 0;
};

class MemoryDecisionTraceSink final : public IDecisionTraceSink {
public:
    bool append(const std::string& jsonLine) override;
    std::vector<std::string> records() const;

private:
    mutable std::mutex mutex_;
    std::vector<std::string> records_;
};

class JsonlDecisionTraceSink final : public IDecisionTraceSink {
public:
    explicit JsonlDecisionTraceSink(std::string path) : path_(std::move(path)) {}
    bool append(const std::string& jsonLine) override;

private:
    std::mutex mutex_;
    std::string path_;
};

struct DecisionObservation {
    uint32_t schemaVersion{1};
    std::string traceId;
    uint64_t timestampUnixMs{0};
    bool success{false};
    std::string failureReason;
    MetricBag actual;
    std::string toJson() const;
};

class AdaptiveSplitController {
public:
    AdaptiveSplitController(std::unique_ptr<IDecisionPolicy> policy,
                            std::shared_ptr<IOptimizationEvaluator> evaluator,
                            std::shared_ptr<IDecisionTraceSink> traceSink);

    DecisionTrace decide(const PolicyContext& context, bool shadowMode = true);

private:
    std::unique_ptr<IDecisionPolicy> policy_;
    std::shared_ptr<IOptimizationEvaluator> evaluator_;
    std::shared_ptr<IDecisionTraceSink> traceSink_;
};

void registerBuiltInDecisionPolicies();

#endif
