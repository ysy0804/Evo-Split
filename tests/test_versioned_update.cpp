#include "VersionedUpdateManager.h"

#include <cmath>
#include <iostream>

namespace {

VersionedBoundary boundary(const std::string& canonical, const std::string& physical,
                           int split, const std::vector<int64_t>& shape,
                           const std::string& upstream = "up",
                           const std::string& downstream = "down") {
    VersionedBoundary value;
    value.canonicalId = canonical;
    value.physicalId = physical;
    value.splitAfterOp = split;
    value.totalOps = 100;
    value.upstreamNeighborhoodHash = upstream;
    value.downstreamNeighborhoodHash = downstream;
    value.supportedCodecs = {CodecKind::RAW32, CodecKind::FP16, CodecKind::BLOCK_INT8};
    VersionedTensorContract tensor;
    tensor.stableId = canonical + ".feature";
    tensor.tensorName = physical + ".output";
    tensor.shape = shape;
    tensor.dtype = "float32";
    tensor.layout = "NCHW";
    value.cutSet.push_back(tensor);
    return value;
}

ModelVersionManifestV4 manifest(const std::string& version, const std::string& structure,
                                const std::string& content, const std::string& id) {
    ModelVersionManifestV4 value;
    value.modelName = "MobileNetV2";
    value.modelVersion = version;
    value.modelPath = "/data/local/tmp/" + version + ".mnn";
    value.structureHash = structure;
    value.contentHash = content;
    value.manifestId = id;
    value.tensorPacketProtocolVersion = 3;
    value.artifactHashes["weights:block0"] = content;
    value.boundaries.push_back(boundary("stage2.output", "op27", 27, {1, 32, 56, 56}));
    value.boundaries.push_back(boundary("stage3.output", "op51", 51, {1, 64, 28, 28}));
    return value;
}

}  // namespace

int main() {
    // Published SHA-256 test vector protects the model identity implementation.
    if (VersionHasher::sha256("abc") !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 1;

    auto oldModel = manifest("v1", "structure-a", "content-a", "manifest-a");
    auto weightOnly = manifest("v2", "structure-a", "content-b", "manifest-b");
    ModelChangeSet weights = ModelDiffEngine::compare(oldModel, weightOnly);
    if (!weights.contentChanged || weights.structureChanged ||
        weights.preliminaryLevel != ImpactLevel::I1_PARAMETERS) return 2;

    DecisionDependencyGraph dependencies;
    dependencies.add({"network-profile", DecisionArtifactKind::RESIDUAL, "", "",
                      {"network:*"}});
    dependencies.add({"stage2-prefix", DecisionArtifactKind::PREFIX_PROFILE, "split-stage2",
                      "stage2.output", {"boundary:stage2.output:neighborhood", "model:structure"}});
    dependencies.add({"stage2-codec", DecisionArtifactKind::CODEC_PROFILE, "split-stage2",
                      "stage2.output", {"model:content", "boundary:stage2.output:contract"}});
    dependencies.add({"stage2-accuracy", DecisionArtifactKind::ACCURACY_PROFILE, "split-stage2",
                      "stage2.output", {"artifact:stage2-codec"}});
    ImpactPlan weightImpact = dependencies.analyze(weights, true);
    if (weightImpact.invalidatedArtifacts.size() != 2 ||
        !weightImpact.invalidatedArtifacts.count("stage2-codec") ||
        !weightImpact.invalidatedArtifacts.count("stage2-accuracy") ||
        weightImpact.closureRatio != 0.5) return 3;
    ImpactPlan full = dependencies.analyze(weights, false);
    if (full.invalidatedArtifacts.size() != dependencies.size()) return 4;

    // Physical layer/tensor names move, while the canonical contract is stable.
    auto renamed = weightOnly;
    renamed.modelVersion = "v3";
    renamed.structureHash = "structure-c";
    renamed.contentHash = "content-c";
    renamed.manifestId = "manifest-c";
    renamed.boundaries[0].physicalId = "op31";
    renamed.boundaries[0].splitAfterOp = 31;
    renamed.boundaries[0].cutSet[0].tensorName = "renamed_tensor";
    renamed.boundaries[0].upstreamNeighborhoodHash = "up-modified";
    ModelChangeSet localChange = ModelDiffEngine::compare(weightOnly, renamed);
    if (localChange.preliminaryLevel != ImpactLevel::I2_LOCAL_COMPUTE) return 5;
    UpdateConfig config;
    config.compatibleMappingThreshold = 0.80;
    CompatibilityMap mappings = BoundaryCompatibilityMapper::build(weightOnly, renamed, config);
    if (mappings.boundaries.size() != 2 ||
        mappings.boundaries[0].kind != MappingKind::CONTRACT_COMPATIBLE ||
        mappings.boundaries[0].newPhysicalId != "op31" ||
        mappings.boundaries[1].kind != MappingKind::EXACT) return 6;

    StateMigrationPlan migration = ControllerStateMigrator::plan(mappings, localChange, true);
    if (migration.reusedCount < 3 || migration.resetCount == 0 || migration.hysteresisMigrated) return 7;

    IncrementalQualificationManager qualification(config);
    qualification.seed("split-stage2-new", MappingKind::CONTRACT_COMPATIBLE);
    for (std::size_t i = 0; i < config.qualificationMinimumSamples; ++i) {
        qualification.observe("split-stage2-new", true, 2.0, 0.005);
    }
    if (qualification.get("split-stage2-new").state != QualificationState::SHADOW_READY ||
        !qualification.activate("split-stage2-new") ||
        qualification.get("split-stage2-new").state != QualificationState::ACTIVE) return 8;
    qualification.seed("unsafe", MappingKind::CALIBRATION_REQUIRED);
    if (qualification.observe("unsafe", true, 1.0, 0.02) != QualificationState::QUARANTINED) return 9;

    VersionedUpdateCoordinator coordinator(config);
    std::string error;
    if (!coordinator.bootstrap(oldModel, &error)) return 10;
    if (!coordinator.prepare("tx-1", renamed, localChange, weightImpact, mappings, migration, &error)) return 11;
    if (!coordinator.markReady("tx-1", true, false, &error) || coordinator.commit("tx-1", &error)) return 12;
    if (!coordinator.markReady("tx-1", false, true, &error) || !coordinator.commit("tx-1", &error) ||
        !coordinator.commit("tx-1", &error)) return 13;  // COMMIT is idempotent.
    std::string reason;
    if (!coordinator.splitCompatible("content-c", "content-c", "manifest-c", 3, &reason)) return 14;
    if (coordinator.splitCompatible("content-c", "content-b", "manifest-c", 3, &reason) ||
        reason != "content_hash_mismatch") return 15;
    if (!coordinator.rollback("injected_failure", &error) ||
        coordinator.active().contentHash != "content-a") return 16;

    UpdateConfig safe = config;
    safe.mode = UpdateMode::SAFE_ONLY;
    coordinator.setConfig(safe);
    if (coordinator.splitAllowed() || coordinator.splitCompatible(
        "content-a", "content-a", "manifest-a", 3, &reason)) return 17;

    std::cout << "VERSIONED_UPDATE_TEST=PASS impact_ratio=" << weightImpact.closureRatio
              << " mapping=" << mappingKindName(mappings.boundaries[0].kind)
              << " migration_reused=" << migration.reusedCount
              << " transaction=" << updateTransactionStateName(coordinator.transaction().state)
              << std::endl;
    return 0;
}
