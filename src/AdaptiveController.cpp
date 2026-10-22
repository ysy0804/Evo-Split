#include "AdaptiveController.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

using Clock = std::chrono::steady_clock;

// ============================================================
// Monotonic timestamp.
// ============================================================
uint64_t steadyNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}


// ============================================================
// Compute an elapsed duration.
// ============================================================
double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}


// ============================================================
// Escape a JSON string.
// ============================================================
std::string escapeJson(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
                } else out << static_cast<char>(c);
        }
    }
    return out.str();
}

// ============================================================
// Codec name.
// ============================================================
const char* codecName(CodecKind codec) {
    switch (codec) {
        case CodecKind::RAW32: return "RAW32";
        case CodecKind::FP16: return "FP16";
        case CodecKind::BLOCK_INT8: return "INT8";
        case CodecKind::BLOCK_INT6: return "INT6";
        case CodecKind::BLOCK_INT4: return "INT4";
        case CodecKind::LZ4: return "LZ4";
    }
    return "UNKNOWN";
}

// ============================================================
// Execution mode name.
// ============================================================
const char* modeName(ActionMode mode) {
    switch (mode) {
        case ActionMode::LOCAL: return "local";
        case ActionMode::CLOUD: return "cloud";
        case ActionMode::SPLIT: return "split";
    }
    return "unknown";
}

// ============================================================
// Metric type.
// ============================================================
MetricType typeOf(const MetricValue& value) {
    if (std::holds_alternative<int64_t>(value)) return MetricType::INTEGER;
    if (std::holds_alternative<double>(value)) return MetricType::NUMBER;
    if (std::holds_alternative<bool>(value)) return MetricType::BOOLEAN;
    return MetricType::STRING;
}

// ============================================================
// JSON representation of a metric value.
// ============================================================
std::string metricValueJson(const MetricValue& value) {
    std::ostringstream out;
    if (const auto* number = std::get_if<int64_t>(&value)) out << *number;
    else if (const auto* number = std::get_if<double>(&value)) {
        if (std::isfinite(*number)) out << std::setprecision(15) << *number;
        else out << "null";
    } else if (const auto* boolean = std::get_if<bool>(&value)) out << (*boolean ? "true" : "false");
    else out << '"' << escapeJson(std::get<std::string>(value)) << '"';
    return out.str();
}

// ============================================================
// JSON representation of an action.
// ============================================================
std::string actionJson(const SplitAction& action) {
    std::ostringstream out;
    out << "{\"key\":\"" << escapeJson(action.key()) << "\",\"mode\":\""
        << modeName(action.mode) << "\",\"boundary\":" << action.boundary
        << ",\"codec\":\"" << codecName(action.codec) << "\",\"blockSize\":"
        << action.blockSize << ",\"modelHash\":\"" << escapeJson(action.modelHash)
        << "\",\"manifestId\":\"" << escapeJson(action.manifestId) << "\"}";
    return out.str();
}

// ============================================================
// Generate the next trace identifier.
// ============================================================
std::string nextTraceId(const std::string& requestId) {
    static std::atomic<uint64_t> sequence{0};
    std::ostringstream out;
    out << (requestId.empty() ? "request" : requestId) << '-' << steadyNs() << '-'
        << sequence.fetch_add(1, std::memory_order_relaxed);
    return out.str();
}

// ============================================================
// Evaluate an optimization constraint.
// ============================================================
bool constraintSatisfied(double value, const ConstraintSpec& constraint) {
    switch (constraint.op) {
        case ConstraintOperator::LESS_EQUAL:
            return value <= constraint.threshold + constraint.tolerance;
        case ConstraintOperator::GREATER_EQUAL:
            return value + constraint.tolerance >= constraint.threshold;
        case ConstraintOperator::EQUAL:
            return std::fabs(value - constraint.threshold) <= constraint.tolerance;
    }
    return false;
}

// ============================================================
// Parse a metric value.
// ============================================================
bool resolveMetric(const RuntimeState& state, const ActionCandidate& candidate,
                   const std::string& metric, double& value) {
    constexpr const char* STATE_PREFIX = "state.";
    constexpr const char* PREDICTED_PREFIX = "predicted.";
    if (metric.compare(0, 6, STATE_PREFIX) == 0) {
        return state.features.getNumber(metric.substr(6), value);
    }
    if (metric.compare(0, 10, PREDICTED_PREFIX) == 0) {
        return candidate.predicted.getNumber(metric.substr(10), value);
    }
    return candidate.predicted.getNumber(metric, value) || state.features.getNumber(metric, value);
}

}  // namespace


void MetricBag::set(const std::string& name, int64_t value) { values_[name] = value; }
void MetricBag::set(const std::string& name, double value) { values_[name] = value; }
void MetricBag::set(const std::string& name, bool value) { values_[name] = value; }
void MetricBag::set(const std::string& name, const std::string& value) { values_[name] = value; }
void MetricBag::set(const std::string& name, const char* value) {
    values_[name] = std::string(value ? value : "");
}


// ============================================================
// Check whether a metric exists.
// ============================================================
bool MetricBag::contains(const std::string& name) const { return values_.count(name) != 0; }

bool MetricBag::getNumber(const std::string& name, double& value) const {
    const auto found = values_.find(name);
    if (found == values_.end()) return false;
    if (const auto* integer = std::get_if<int64_t>(&found->second)) {
        value = static_cast<double>(*integer);
        return true;
    }
    if (const auto* number = std::get_if<double>(&found->second)) {
        value = *number;
        return std::isfinite(value);
    }
    if (const auto* boolean = std::get_if<bool>(&found->second)) {
        value = *boolean ? 1.0 : 0.0;
        return true;
    }
    return false;
}

bool MetricBag::getString(const std::string& name, std::string& value) const {
    const auto found = values_.find(name);
    if (found == values_.end()) return false;
    const auto* text = std::get_if<std::string>(&found->second);
    if (!text) return false;
    value = *text;
    return true;
}

// ============================================================
// Serialize a metric bag as JSON.
// ============================================================
std::string MetricBag::toJson() const {
    std::ostringstream out;
    out << '{';
    bool first = true;
    for (const auto& entry : values_) {
        if (!first) out << ',';
        first = false;
        out << '"' << escapeJson(entry.first) << "\":" << metricValueJson(entry.second);
    }
    out << '}';
    return out.str();
}

bool MetricSchema::add(const MetricDefinition& definition) {
    if (definition.name.empty() || definitions_.count(definition.name)) return false;
    definitions_[definition.name] = definition;
    return true;
}

bool MetricSchema::validate(const MetricBag& bag, std::vector<std::string>& errors) const {
    errors.clear();
    for (const auto& entry : definitions_) {
        const auto value = bag.values().find(entry.first);
        if (value == bag.values().end()) {
            if (entry.second.required) errors.push_back("missing:" + entry.first);
            continue;
        }
        const MetricType actual = typeOf(value->second);
        const bool numericCompatible = entry.second.type == MetricType::NUMBER && actual == MetricType::INTEGER;
        if (actual != entry.second.type && !numericCompatible) errors.push_back("type:" + entry.first);
    }
    return errors.empty();
}

// ============================================================
// Build the default runtime-state metric schema.
// ============================================================
MetricSchema buildDefaultRuntimeStateSchema() {
    MetricSchema schema(1);
    const std::vector<MetricDefinition> definitions = {
        {"device.cpu.util_pct", MetricType::NUMBER, "%", "whole-device CPU utilization", false},
        {"device.cpu.little_util_pct", MetricType::NUMBER, "%", "little-cluster utilization", false},
        {"device.cpu.big_util_pct", MetricType::NUMBER, "%", "big-cluster utilization", false},
        {"device.cpu.freq_mhz", MetricType::NUMBER, "MHz", "effective CPU frequency", false},
        {"device.memory.available_mb", MetricType::NUMBER, "MiB", "system available memory", false},
        {"device.memory.used_mb", MetricType::NUMBER, "MiB", "daemon resident memory", false},
        {"device.memory.peak_mb", MetricType::NUMBER, "MiB", "daemon peak resident memory", false},
        {"device.thermal.soc_c", MetricType::NUMBER, "C", "SoC temperature", false},
        {"device.battery.power_mw", MetricType::NUMBER, "mW", "battery power", false},
        {"network.rtt_ms", MetricType::NUMBER, "ms", "recent round-trip time", false},
        {"network.goodput_up_bps", MetricType::NUMBER, "B/s", "uplink application goodput", false},
        {"network.jitter_ms", MetricType::NUMBER, "ms", "recent RTT jitter", false},
        {"network.loss_rate", MetricType::NUMBER, "ratio", "packet/request loss rate", false},
        {"network.available", MetricType::NUMBER, "0/1", "remote network availability", false},
        {"network.interface", MetricType::STRING, "", "wifi, cellular, hotspot or usb", false},
        {"cloud.queue_ms", MetricType::NUMBER, "ms", "reported cloud queue delay", false},
        {"cloud.util_pct", MetricType::NUMBER, "%", "reported cloud utilization", false},
        {"cloud.available", MetricType::NUMBER, "0/1", "cloud service availability", false},
        {"history.residual_ms", MetricType::NUMBER, "ms", "latest prediction residual", false},
        {"history.fallback_count", MetricType::INTEGER, "count", "consecutive fallbacks", false},
    };
    for (const auto& definition : definitions) schema.add(definition);
    return schema;
}

// ============================================================
// Build the default prediction metric schema.
// ============================================================
MetricSchema buildDefaultPredictionSchema() {
    MetricSchema schema(1);
    const std::vector<MetricDefinition> definitions = {
        {"latency.edge_prefix_ms", MetricType::NUMBER, "ms", "edge prefix", false},
        {"latency.capture_ms", MetricType::NUMBER, "ms", "cut-set capture", false},
        {"latency.encode_ms", MetricType::NUMBER, "ms", "edge codec", false},
        {"latency.transfer_ms", MetricType::NUMBER, "ms", "network transfer", false},
        {"latency.decode_ms", MetricType::NUMBER, "ms", "cloud codec", false},
        {"latency.inject_ms", MetricType::NUMBER, "ms", "cloud injection", false},
        {"latency.cloud_suffix_ms", MetricType::NUMBER, "ms", "cloud suffix", false},
        {"latency.cloud_queue_ms", MetricType::NUMBER, "ms", "cloud queue", false},
        {"latency.total_mean_ms", MetricType::NUMBER, "ms", "predicted mean total", false},
        {"latency.p95_ms", MetricType::NUMBER, "ms", "predicted p95 total", false},
        {"payload.raw_bytes", MetricType::INTEGER, "B", "raw cut-set payload", false},
        {"payload.encoded_bytes", MetricType::INTEGER, "B", "encoded tensor payload", false},
        {"payload.bundle_bytes", MetricType::INTEGER, "B", "wire bundle", false},
        {"accuracy.drop", MetricType::NUMBER, "task metric", "quality loss versus RAW32", false},
        {"memory.peak_mb", MetricType::NUMBER, "MiB", "predicted peak memory", false},
        {"energy.mj", MetricType::NUMBER, "mJ", "predicted request energy", false},
        {"reliability.failure_probability", MetricType::NUMBER, "ratio", "failure risk", false},
    };
    for (const auto& definition : definitions) schema.add(definition);
    return schema;
}

// ============================================================
// Generate a stable action key.
// ============================================================
std::string SplitAction::key() const {
    std::ostringstream out;
    out << modeName(mode);
    if (mode == ActionMode::SPLIT) out << ":b" << boundary << ':' << codecName(codec) << ":g" << blockSize;
    if (!modelHash.empty()) out << ":m" << modelHash;
    return out.str();
}

// ============================================================
// Insert or update a candidate manifest.
// ============================================================
bool CandidateRegistry::upsert(const CandidateManifest& manifest, std::string* error) {
    if (manifest.manifestId.empty() || manifest.modelHash.empty() || manifest.candidates.empty()) {
        if (error) *error = "manifest id, model hash and candidates are required";
        return false;
    }
    std::map<std::string, bool> unique;
    for (const auto& candidate : manifest.candidates) {
        if (!candidate.action.modelHash.empty() && candidate.action.modelHash != manifest.modelHash) {
            if (error) *error = "candidate model hash mismatch";
            return false;
        }
        const std::string key = candidate.action.key();
        if (unique.count(key)) {
            if (error) *error = "duplicate candidate action";
            return false;
        }
        unique[key] = true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    manifests_[manifest.modelHash] = manifest;
    return true;
}

// ============================================================
// Retrieve a candidate manifest.
// ============================================================
bool CandidateRegistry::get(const std::string& modelHash, CandidateManifest& manifest) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = manifests_.find(modelHash);
    if (found == manifests_.end()) return false;
    manifest = found->second;
    return true;
}

bool CandidateRegistry::invalidate(const std::string& modelHash) {
    std::lock_guard<std::mutex> lock(mutex_);
    return manifests_.erase(modelHash) != 0;
}

std::size_t CandidateRegistry::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return manifests_.size();
}

// ============================================================
// Configure the optimization evaluator.
// ============================================================
ConfigurableOptimizationEvaluator::ConfigurableOptimizationEvaluator(OptimizationSpec spec)
    : spec_(std::move(spec)) {}

CandidateEvaluation ConfigurableOptimizationEvaluator::evaluate(
    const RuntimeState& state, const ActionCandidate& candidate) const {
    CandidateEvaluation result;
    result.feasible = true;
    for (const ObjectiveTerm& term : spec_.objectives) {
        double value = 0.0;
        if (!resolveMetric(state, candidate, term.metric, value) || term.scale <= 0.0) {
            result.feasible = false;
            result.violations.push_back("missing_objective:" + term.metric);
            continue;
        }
        const double sign = term.direction == ObjectiveDirection::MINIMIZE ? 1.0 : -1.0;
        const double contribution = sign * term.weight * value / term.scale;
        result.score += contribution;
        result.termContributions.set(term.metric, contribution);
    }
    for (const ConstraintSpec& constraint : spec_.constraints) {
        double value = 0.0;
        if (!resolveMetric(state, candidate, constraint.metric, value)) {
            if (constraint.hard) result.feasible = false;
            result.score += constraint.penalty;
            result.violations.push_back("missing_constraint:" + constraint.id);
            continue;
        }
        if (!constraintSatisfied(value, constraint)) {
            if (constraint.hard) result.feasible = false;
            result.score += constraint.penalty;
            result.violations.push_back("violated:" + constraint.id);
        }
    }
    return result;
}

// ============================================================
// Return the evaluator description as JSON.
// ============================================================
std::string ConfigurableOptimizationEvaluator::describeJson() const {
    std::ostringstream out;
    out << "{\"id\":\"" << escapeJson(spec_.id) << "\",\"version\":" << spec_.version
        << ",\"objectives\":[";
    for (size_t i = 0; i < spec_.objectives.size(); ++i) {
        if (i) out << ',';
        const auto& term = spec_.objectives[i];
        out << "{\"metric\":\"" << escapeJson(term.metric) << "\",\"weight\":" << term.weight
            << ",\"scale\":" << term.scale << ",\"direction\":\""
            << (term.direction == ObjectiveDirection::MINIMIZE ? "minimize" : "maximize") << "\"}";
    }
    out << "],\"constraints\":[";
    for (size_t i = 0; i < spec_.constraints.size(); ++i) {
        if (i) out << ',';
        const auto& constraint = spec_.constraints[i];
        const char* op = constraint.op == ConstraintOperator::LESS_EQUAL ? "le" :
                         constraint.op == ConstraintOperator::GREATER_EQUAL ? "ge" : "eq";
        out << "{\"id\":\"" << escapeJson(constraint.id) << "\",\"metric\":\""
            << escapeJson(constraint.metric) << "\",\"op\":\"" << op
            << "\",\"threshold\":" << constraint.threshold << ",\"tolerance\":"
            << constraint.tolerance << ",\"hard\":" << (constraint.hard ? "true" : "false")
            << ",\"penalty\":" << constraint.penalty << '}';
    }
    out << "]}";
    return out.str();
}

// ============================================================
// Select the best feasible action.
// ============================================================
PolicyDecision GreedyProfilePolicy::select(
    const PolicyContext&, const std::vector<CandidateEvaluation>& evaluations) {
    PolicyDecision decision;
    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < evaluations.size(); ++i) {
        if (evaluations[i].feasible && evaluations[i].score < best) {
            best = evaluations[i].score;
            decision.selectedIndex = static_cast<int>(i);
        }
    }
    decision.reason = decision.selectedIndex >= 0 ? "minimum_feasible_score" : "no_feasible_action";
    decision.diagnostics.set("best_score", best);
    decision.diagnostics.set("evaluated_candidates", static_cast<int64_t>(evaluations.size()));
    return decision;
}


// ============================================================
// Return the decision-policy registry.
// ============================================================
DecisionPolicyRegistry& DecisionPolicyRegistry::instance() {
    static DecisionPolicyRegistry registry;
    return registry;
}

// ============================================================
// Register a decision-policy factory.
// ============================================================
bool DecisionPolicyRegistry::registerFactory(const std::string& name, Factory factory) {
    if (name.empty() || !factory) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return factories_.emplace(name, std::move(factory)).second;
}

// ============================================================
// Create a decision policy.
// ============================================================
std::unique_ptr<IDecisionPolicy> DecisionPolicyRegistry::create(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = factories_.find(name);
    return found == factories_.end() ? nullptr : found->second();
}


// ============================================================
// List registered decision-policy names.
// ============================================================
std::vector<std::string> DecisionPolicyRegistry::names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    for (const auto& item : factories_) result.push_back(item.first);
    return result;
}


// ============================================================
// Serialize a decision trace as JSON.
// ============================================================
std::string DecisionTrace::toJson() const {
    std::ostringstream out;
    out << "{\"schemaVersion\":" << schemaVersion << ",\"traceId\":\"" << escapeJson(traceId)
        << "\",\"requestId\":\"" << escapeJson(requestId) << "\",\"algorithm\":\""
        << escapeJson(algorithm) << "\",\"optimizationSpecId\":\""
        << escapeJson(optimizationSpecId) << "\",\"optimizationSpecVersion\":"
        << optimizationSpecVersion << ",\"optimizationSpec\":"
        << (optimizationSpecJson.empty() ? "null" : optimizationSpecJson)
        << ",\"shadowMode\":" << (shadowMode ? "true" : "false")
        << ",\"previousActionKey\":\"" << escapeJson(previousActionKey)
        << "\",\"state\":{\"schemaVersion\":"
        << state.schemaVersion << ",\"timestampUnixMs\":" << state.timestampUnixMs
        << ",\"modelName\":\"" << escapeJson(state.modelName) << "\",\"modelHash\":\""
        << escapeJson(state.modelHash) << "\",\"modelVersion\":\""
        << escapeJson(state.modelVersion) << "\",\"features\":" << state.features.toJson()
        << "},\"candidates\":[";
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i) out << ',';
        out << "{\"action\":" << actionJson(candidates[i].action)
            << ",\"predicted\":" << candidates[i].predicted.toJson()
            << ",\"annotations\":" << candidates[i].annotations.toJson();
        if (i < evaluations.size()) {
            out << ",\"feasible\":" << (evaluations[i].feasible ? "true" : "false")
                << ",\"score\":" << evaluations[i].score << ",\"violations\":[";
            for (size_t v = 0; v < evaluations[i].violations.size(); ++v) {
                if (v) out << ',';
                out << '"' << escapeJson(evaluations[i].violations[v]) << '"';
            }
            out << "],\"termContributions\":" << evaluations[i].termContributions.toJson();
        }
        out << '}';
    }
    out << "],\"selectedIndex\":" << selectedIndex << ",\"selectedActionKey\":\""
        << escapeJson(selectedActionKey) << "\",\"reason\":\"" << escapeJson(reason)
        << "\",\"policyDiagnostics\":" << policyDiagnostics.toJson()
        << ",\"timing\":{\"startSteadyNs\":" << timing.startSteadyNs
        << ",\"endSteadyNs\":" << timing.endSteadyNs << ",\"evaluationMs\":"
        << timing.evaluationMs << ",\"policyMs\":" << timing.policyMs
        << ",\"traceSerializationMs\":" << timing.traceSerializationMs
        << ",\"tracePersistMs\":" << timing.tracePersistMs << ",\"totalMs\":"
        << timing.totalMs << "}}";
    return out.str();
}


// ============================================================
// In-memory decision trace sink.
// ============================================================
bool MemoryDecisionTraceSink::append(const std::string& jsonLine) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back(jsonLine);
    return true;
}


// ============================================================
// Return all recorded decision traces.
// ============================================================
std::vector<std::string> MemoryDecisionTraceSink::records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

// ============================================================
// JSONL decision trace sink.
// ============================================================
bool JsonlDecisionTraceSink::append(const std::string& jsonLine) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream output(path_, std::ios::app);
    if (!output) return false;
    output << jsonLine << '\n';
    return output.good();
}



// ============================================================
// Serialize a decision observation as JSON.
// ============================================================
std::string DecisionObservation::toJson() const {
    std::ostringstream out;
    out << "{\"schemaVersion\":" << schemaVersion << ",\"traceId\":\""
        << escapeJson(traceId) << "\",\"timestampUnixMs\":" << timestampUnixMs
        << ",\"success\":" << (success ? "true" : "false")
        << ",\"failureReason\":\"" << escapeJson(failureReason)
        << "\",\"actual\":" << actual.toJson() << '}';
    return out.str();
}

// ============================================================
// Split decision controller.
// ============================================================
AdaptiveSplitController::AdaptiveSplitController(
    std::unique_ptr<IDecisionPolicy> policy,
    std::shared_ptr<IOptimizationEvaluator> evaluator,
    std::shared_ptr<IDecisionTraceSink> traceSink)
    : policy_(std::move(policy)), evaluator_(std::move(evaluator)), traceSink_(std::move(traceSink)) {}



DecisionTrace AdaptiveSplitController::decide(const PolicyContext& context, bool shadowMode) {
    const auto totalStart = Clock::now();
    DecisionTrace trace;
    trace.traceId = nextTraceId(context.state.requestId);
    trace.requestId = context.state.requestId;
    trace.algorithm = policy_ ? policy_->name() : "missing_policy";
    trace.optimizationSpecId = evaluator_ ? evaluator_->id() : "missing_evaluator";
    trace.optimizationSpecVersion = evaluator_ ? evaluator_->version() : 0;
    trace.optimizationSpecJson = evaluator_ ? evaluator_->describeJson() : "";
    trace.shadowMode = shadowMode;
    trace.previousActionKey = context.previousActionKey;
    trace.state = context.state;
    trace.candidates = context.candidates;
    trace.timing.startSteadyNs = steadyNs();

    const auto evaluationStart = Clock::now();
    if (evaluator_) {
        trace.evaluations.reserve(context.candidates.size());
        for (const auto& candidate : context.candidates) {
            trace.evaluations.push_back(evaluator_->evaluate(context.state, candidate));
        }
    }
    const auto evaluationEnd = Clock::now();
    trace.timing.evaluationMs = elapsedMs(evaluationStart, evaluationEnd);

    const auto policyStart = Clock::now();
    PolicyDecision decision;
    if (policy_) decision = policy_->select(context, trace.evaluations);
    else decision.reason = "policy_not_configured";
    const auto policyEnd = Clock::now();
    trace.timing.policyMs = elapsedMs(policyStart, policyEnd);
    trace.selectedIndex = decision.selectedIndex;
    trace.reason = decision.reason;
    trace.policyDiagnostics = decision.diagnostics;
    if (decision.selectedIndex >= 0 &&
        static_cast<size_t>(decision.selectedIndex) < context.candidates.size()) {
        trace.selectedActionKey = context.candidates[decision.selectedIndex].action.key();
    }

    const auto serializationStart = Clock::now();
    (void)trace.toJson();
    const auto serializationEnd = Clock::now();
    trace.timing.traceSerializationMs = elapsedMs(serializationStart, serializationEnd);

    // ============================================================
    // Decision trace sink.
    // ============================================================
    
    // The persisted trace contains complete evaluation/policy/serialization
    // overhead. Persistence itself is measured in the returned trace because
    // its exact duration is only known after the record has been written.
    trace.timing.endSteadyNs = steadyNs();
    trace.timing.totalMs = elapsedMs(totalStart, serializationEnd);
    const std::string jsonLine = trace.toJson();

    const auto persistStart = Clock::now();
    if (traceSink_) traceSink_->append(jsonLine);
    const auto persistEnd = Clock::now();
    trace.timing.tracePersistMs = elapsedMs(persistStart, persistEnd);
    trace.timing.endSteadyNs = steadyNs();
    trace.timing.totalMs = elapsedMs(totalStart, persistEnd);
    return trace;
}

void registerBuiltInDecisionPolicies() {
    DecisionPolicyRegistry::instance().registerFactory(
        "greedy_profile_v1", [] { return std::make_unique<GreedyProfilePolicy>(); });
}
