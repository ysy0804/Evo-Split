#include "AdaptiveController.h"

#include <cmath>
#include <iostream>
#include <memory>

namespace {

ActionCandidate candidate(ActionMode mode, int boundary, CodecKind codec,
                          double p95, double accuracyDrop, double memoryMb) {
    ActionCandidate value;
    value.action.mode = mode;
    value.action.boundary = boundary;
    value.action.codec = codec;
    value.action.modelHash = "graph-abc";
    value.action.manifestId = "manifest-1";
    value.predicted.set("latency.p95_ms", p95);
    value.predicted.set("accuracy.drop", accuracyDrop);
    value.predicted.set("memory.peak_mb", memoryMb);
    return value;
}

OptimizationSpec latencySpec(double accuracyLimit) {
    OptimizationSpec spec;
    spec.id = "test_latency_p95";
    spec.objectives.push_back({"latency.p95_ms", 1.0, 1.0, ObjectiveDirection::MINIMIZE});
    spec.constraints.push_back(
        {"accuracy_budget", "accuracy.drop", ConstraintOperator::LESS_EQUAL,
         accuracyLimit, 1e-12, true, 0.0});
    spec.constraints.push_back(
        {"memory_budget", "memory.peak_mb", ConstraintOperator::LESS_EQUAL,
         512.0, 0.0, true, 0.0});
    spec.constraints.push_back(
        {"thermal_budget", "state.device.temp_c", ConstraintOperator::LESS_EQUAL,
         50.0, 0.0, true, 0.0});
    return spec;
}

class LastFeasiblePolicy final : public IDecisionPolicy {
public:
    std::string name() const override { return "last_feasible_test"; }
    PolicyDecision select(const PolicyContext&,
                          const std::vector<CandidateEvaluation>& evaluations) override {
        PolicyDecision result;
        for (size_t i = 0; i < evaluations.size(); ++i) {
            if (evaluations[i].feasible) result.selectedIndex = static_cast<int>(i);
        }
        result.reason = "test_last_feasible";
        return result;
    }
};

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    if (buildDefaultRuntimeStateSchema().definitions().size() < 15 ||
        buildDefaultPredictionSchema().definitions().size() < 15) return 1;
    MetricSchema stateSchema(1);
    if (!stateSchema.add({"device.cpu.util", MetricType::NUMBER, "%", "CPU utilization", true}) ||
        !stateSchema.add({"network.goodput_bps", MetricType::NUMBER, "B/s", "uplink", true}) ||
        stateSchema.add({"network.goodput_bps", MetricType::NUMBER, "B/s", "duplicate", false})) return 1;
    MetricBag stateMetrics;
    stateMetrics.set("device.cpu.util", 72.5);
    std::vector<std::string> schemaErrors;
    if (stateSchema.validate(stateMetrics, schemaErrors) ||
        schemaErrors.size() != 1 || schemaErrors[0] != "missing:network.goodput_bps") return 2;
    stateMetrics.set("network.goodput_bps", 12.0e6);
    stateMetrics.set("device.temp_c", 43.0);
    stateMetrics.set("network.interface", "wifi");
    stateMetrics.set("future.optional.metric", 42.0);  // unknown extensions are preserved and accepted.
    if (!stateSchema.validate(stateMetrics, schemaErrors)) return 3;

    PolicyContext context;
    context.state.requestId = "request-7";
    context.state.modelName = "MobileNetV2";
    context.state.modelHash = "graph-abc";
    context.state.features = stateMetrics;
    context.candidates = {
        candidate(ActionMode::LOCAL, -1, CodecKind::RAW32, 80.0, 0.0, 220.0),
        candidate(ActionMode::SPLIT, 11, CodecKind::BLOCK_INT8, 35.0, 0.01, 260.0),
        candidate(ActionMode::SPLIT, 20, CodecKind::BLOCK_INT4, 22.0, 0.08, 250.0),
        candidate(ActionMode::CLOUD, -1, CodecKind::RAW32, 60.0, 0.0, 600.0),
    };
    CandidateManifest manifest;
    manifest.manifestId = "manifest-1";
    manifest.modelName = "MobileNetV2";
    manifest.modelHash = "graph-abc";
    manifest.modelVersion = "v1";
    manifest.candidates = context.candidates;
    CandidateRegistry candidateRegistry;
    std::string registryError;
    if (!candidateRegistry.upsert(manifest, &registryError) || candidateRegistry.size() != 1) return 4;
    CandidateManifest restored;
    if (!candidateRegistry.get("graph-abc", restored) || restored.candidates.size() != 4) return 4;

    registerBuiltInDecisionPolicies();
    auto greedy = DecisionPolicyRegistry::instance().create("greedy_profile_v1");
    if (!greedy || DecisionPolicyRegistry::instance().create("missing")) return 5;
    auto evaluator = std::make_shared<ConfigurableOptimizationEvaluator>(latencySpec(0.02));
    auto sink = std::make_shared<MemoryDecisionTraceSink>();
    AdaptiveSplitController controller(std::move(greedy), evaluator, sink);
    DecisionTrace first = controller.decide(context, true);
    if (first.selectedIndex != 1 || first.selectedActionKey.find("b11:INT8") == std::string::npos ||
        first.evaluations[2].feasible || first.evaluations[3].feasible ||
        first.timing.totalMs < 0.0 || first.timing.endSteadyNs < first.timing.startSteadyNs ||
        sink->records().size() != 1) return 6;
    const std::string traceJson = sink->records()[0];
    if (!contains(traceJson, "\"algorithm\":\"greedy_profile_v1\"") ||
        !contains(traceJson, "violated:accuracy_budget") ||
        !contains(traceJson, "violated:memory_budget") ||
        !contains(traceJson, "future.optional.metric") ||
        !contains(traceJson, "\"optimizationSpecVersion\":1") ||
        !contains(traceJson, "\"objectives\":[") ||
        !contains(traceJson, "\"evaluationMs\":")) return 7;

    // Changing only the formula configuration should alter the decision;
    // controller/policy code remains untouched.
    OptimizationSpec energySpec;
    energySpec.id = "test_energy";
    energySpec.objectives.push_back({"energy.mj", 1.0, 1.0, ObjectiveDirection::MINIMIZE});
    context.candidates[0].predicted.set("energy.mj", 8.0);
    context.candidates[1].predicted.set("energy.mj", 20.0);
    context.candidates[2].predicted.set("energy.mj", 12.0);
    context.candidates[3].predicted.set("energy.mj", 30.0);
    evaluator->setSpec(energySpec);
    auto secondGreedy = DecisionPolicyRegistry::instance().create("greedy_profile_v1");
    AdaptiveSplitController energyController(std::move(secondGreedy), evaluator, sink);
    DecisionTrace second = energyController.decide(context, true);
    if (second.selectedIndex != 0 || second.traceId == first.traceId) return 8;

    // A new action-selection algorithm is registered without editing the controller.
    if (!DecisionPolicyRegistry::instance().registerFactory(
            "last_feasible_test", [] { return std::make_unique<LastFeasiblePolicy>(); })) return 9;
    auto plugin = DecisionPolicyRegistry::instance().create("last_feasible_test");
    AdaptiveSplitController pluginController(std::move(plugin), evaluator, sink);
    DecisionTrace third = pluginController.decide(context, true);
    if (third.selectedIndex != 3 || third.algorithm != "last_feasible_test") return 10;

    DecisionObservation observation;
    observation.traceId = first.traceId;
    observation.timestampUnixMs = 1234;
    observation.success = true;
    observation.actual.set("latency.total_ms", 37.2);
    if (!contains(observation.toJson(), first.traceId) ||
        !contains(observation.toJson(), "latency.total_ms")) return 11;
    if (!candidateRegistry.invalidate("graph-abc") || candidateRegistry.size() != 0) return 12;

    std::cout << "ADAPTIVE_CONTROLLER_TEST=PASS selected=" << first.selectedActionKey
              << " decision_ms=" << first.timing.totalMs
              << " trace_records=" << sink->records().size() << std::endl;
    if (argc > 1 && std::string(argv[1]) == "--dump-json") {
        std::cout << "DECISION_TRACE_JSON=" << traceJson << std::endl;
    }
    return 0;
}
