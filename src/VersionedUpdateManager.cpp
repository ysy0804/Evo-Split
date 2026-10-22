#include "VersionedUpdateManager.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

uint64_t unixMs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string escapeJson(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                  << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
                else out << static_cast<char>(c);
        }
    }
    return out.str();
}

template <typename Values>
std::string stringArrayJson(const Values& values) {
    std::ostringstream out;
    out << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) out << ',';
        first = false;
        out << '"' << escapeJson(value) << '"';
    }
    out << ']';
    return out.str();
}

const char* codecNameV4(CodecKind codec) {
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

bool dependencyMatches(const std::string& dependency, const std::string& seed) {
    if (dependency == seed || dependency == "model:any") return true;
    if (!dependency.empty() && dependency.back() == '*') {
        const std::string prefix = dependency.substr(0, dependency.size() - 1);
        return seed.compare(0, prefix.size(), prefix) == 0;
    }
    return false;
}

// Small self-contained SHA-256 implementation.  It avoids an OpenSSL runtime
// dependency on Android while giving model identities cryptographic strength.
class Sha256 {
public:
    Sha256() { reset(); }
    void update(const uint8_t* data, std::size_t length) {
        for (std::size_t i = 0; i < length; ++i) {
            block_[blockLength_++] = data[i];
            if (blockLength_ == 64) {
                transform();
                bitLength_ += 512;
                blockLength_ = 0;
            }
        }
    }
    std::string finish() {
        std::size_t i = blockLength_;
        block_[i++] = 0x80;
        if (i > 56) {
            while (i < 64) block_[i++] = 0;
            transform();
            i = 0;
        }
        while (i < 56) block_[i++] = 0;
        bitLength_ += static_cast<uint64_t>(blockLength_) * 8;
        for (int shift = 56; shift >= 0; shift -= 8) block_[i++] = (bitLength_ >> shift) & 0xff;
        transform();
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (uint32_t word : state_) out << std::setw(8) << word;
        return out.str();
    }
private:
    static uint32_t rotate(uint32_t value, uint32_t bits) {
        return (value >> bits) | (value << (32 - bits));
    }
    void reset() {
        state_ = {{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
        blockLength_ = 0; bitLength_ = 0;
    }
    void transform() {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block_[i * 4]) << 24) |
                   (static_cast<uint32_t>(block_[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block_[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(block_[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotate(e,6)^rotate(e,11)^rotate(e,25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + k[i] + w[i];
            const uint32_t s0 = rotate(a,2)^rotate(a,13)^rotate(a,22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }
    std::array<uint32_t,8> state_;
    std::array<uint8_t,64> block_{};
    std::size_t blockLength_{0};
    uint64_t bitLength_{0};
};

}  // namespace

const char* updateModeName(UpdateMode v) {
    switch(v){case UpdateMode::OFF:return "off";case UpdateMode::AUDIT:return "audit";
    case UpdateMode::SHADOW:return "shadow";case UpdateMode::ENFORCE:return "enforce";
    case UpdateMode::SAFE_ONLY:return "safe_only";} return "unknown";
}
const char* impactLevelName(ImpactLevel v) {
    switch(v){case ImpactLevel::I0_NONE:return "I0_NONE";case ImpactLevel::I1_PARAMETERS:return "I1_PARAMETERS";
    case ImpactLevel::I2_LOCAL_COMPUTE:return "I2_LOCAL_COMPUTE";case ImpactLevel::I3_INTERFACE:return "I3_INTERFACE";
    case ImpactLevel::I4_GLOBAL:return "I4_GLOBAL";} return "UNKNOWN";
}
const char* mappingKindName(MappingKind v) {
    switch(v){case MappingKind::EXACT:return "exact";case MappingKind::CONTRACT_COMPATIBLE:return "contract_compatible";
    case MappingKind::CALIBRATION_REQUIRED:return "calibration_required";case MappingKind::INCOMPATIBLE:return "incompatible";}
    return "unknown";
}
const char* qualificationStateName(QualificationState v) {
    switch(v){case QualificationState::UNAVAILABLE:return "unavailable";case QualificationState::PROFILING:return "profiling";
    case QualificationState::SHADOW_READY:return "shadow_ready";case QualificationState::ACTIVE:return "active";
    case QualificationState::QUARANTINED:return "quarantined";} return "unknown";
}
const char* updateTransactionStateName(UpdateTransactionState v) {
    switch(v){case UpdateTransactionState::IDLE:return "idle";case UpdateTransactionState::PREPARING:return "preparing";
    case UpdateTransactionState::PREPARED:return "prepared";case UpdateTransactionState::COMMITTED:return "committed";
    case UpdateTransactionState::ABORTED:return "aborted";case UpdateTransactionState::ROLLED_BACK:return "rolled_back";}
    return "unknown";
}

bool parseUpdateMode(const std::string& text, UpdateMode& value) {
    if(text=="off")value=UpdateMode::OFF; else if(text=="audit")value=UpdateMode::AUDIT;
    else if(text=="shadow")value=UpdateMode::SHADOW; else if(text=="enforce")value=UpdateMode::ENFORCE;
    else if(text=="safe_only")value=UpdateMode::SAFE_ONLY; else return false; return true;
}

std::string VersionHasher::sha256(const void* data, std::size_t size) {
    Sha256 hash; hash.update(static_cast<const uint8_t*>(data), size); return hash.finish();
}
std::string VersionHasher::sha256(const std::string& text) { return sha256(text.data(), text.size()); }

bool VersionHasher::hashFileMerkle(const std::string& path, FileMerkleIdentity& result,
                                   std::string* error, std::size_t chunkBytes) {
    result = FileMerkleIdentity();
    if (chunkBytes == 0) { if(error)*error="chunk size must be positive"; return false; }
    result.chunkBytes = chunkBytes;
    std::ifstream input(path, std::ios::binary);
    if (!input) { if(error)*error="cannot open model file"; return false; }
    std::vector<char> buffer(chunkBytes);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count <= 0) break;
        result.fileBytes += static_cast<uint64_t>(count);
        result.chunkHashes.push_back(sha256(buffer.data(), static_cast<std::size_t>(count)));
    }
    if (input.bad() || result.chunkHashes.empty()) { if(error)*error="cannot read model file"; return false; }
    std::ostringstream root;
    root << result.algorithm << ':' << result.fileBytes << ':' << result.chunkBytes << ':';
    for (const auto& hash : result.chunkHashes) root << hash << ';';
    result.rootHash = sha256(root.str());
    return true;
}

std::string VersionedTensorContract::signature() const {
    std::ostringstream out; out << dtype << ':' << layout << ':' << shape.size() << ':';
    for (auto dimension : shape) out << dimension << ',';
    return VersionHasher::sha256(out.str());
}

std::string VersionedBoundary::contractSignature() const {
    std::ostringstream out; out << cutSet.size() << ':';
    for (const auto& tensor : cutSet) out << tensor.signature() << ';';
    return VersionHasher::sha256(out.str());
}

std::string VersionedBoundary::semanticFingerprint() const {
    std::ostringstream out; out << contractSignature() << ':' << upstreamNeighborhoodHash << ':'
        << downstreamNeighborhoodHash << ':';
    for (CodecKind codec : supportedCodecs) out << codecNameV4(codec) << ',';
    return VersionHasher::sha256(out.str());
}

bool ModelVersionManifestV4::valid(std::string* error) const {
    if (modelVersion.empty() || structureHash.empty() || contentHash.empty() || manifestId.empty()) {
        if(error)*error="model version, hashes and manifest id are required"; return false;
    }
    if (tensorPacketProtocolVersion == 0) { if(error)*error="packet protocol version must be positive"; return false; }
    std::set<std::string> canonical, physical;
    for (const auto& boundary : boundaries) {
        if (boundary.canonicalId.empty() || boundary.physicalId.empty() || boundary.cutSet.empty() ||
            !canonical.insert(boundary.canonicalId).second || !physical.insert(boundary.physicalId).second) {
            if(error)*error="invalid or duplicate boundary"; return false;
        }
    }
    return true;
}

const VersionedBoundary* ModelVersionManifestV4::findBoundary(const std::string& id) const {
    for (const auto& boundary : boundaries) if (boundary.canonicalId == id) return &boundary;
    return nullptr;
}

std::set<std::string> ModelChangeSet::dependencySeeds() const {
    std::set<std::string> seeds;
    if (contentChanged) seeds.insert("model:content");
    if (structureChanged) seeds.insert("model:structure");
    if (protocolChanged) seeds.insert("protocol");
    if (configChanged) seeds.insert("config");
    for (const auto& value : changedArtifacts) seeds.insert("artifact:" + value);
    for (const auto& value : addedBoundaries) seeds.insert("boundary:" + value + ":added");
    for (const auto& value : removedBoundaries) seeds.insert("boundary:" + value + ":removed");
    for (const auto& value : changedBoundaryContracts) seeds.insert("boundary:" + value + ":contract");
    for (const auto& value : changedBoundaryNeighborhoods) seeds.insert("boundary:" + value + ":neighborhood");
    return seeds;
}

std::string ModelChangeSet::toJson() const {
    std::ostringstream out; out << "{\"schemaVersion\":" << schemaVersion
        << ",\"oldContentHash\":\"" << escapeJson(oldContentHash) << "\",\"newContentHash\":\""
        << escapeJson(newContentHash) << "\",\"contentChanged\":" << (contentChanged?"true":"false")
        << ",\"structureChanged\":" << (structureChanged?"true":"false")
        << ",\"protocolChanged\":" << (protocolChanged?"true":"false")
        << ",\"configChanged\":" << (configChanged?"true":"false")
        << ",\"impactLevel\":\"" << impactLevelName(preliminaryLevel) << "\",\"diffMs\":" << diffMs
        << ",\"changedArtifacts\":" << stringArrayJson(changedArtifacts)
        << ",\"addedBoundaries\":" << stringArrayJson(addedBoundaries)
        << ",\"removedBoundaries\":" << stringArrayJson(removedBoundaries)
        << ",\"changedBoundaryContracts\":" << stringArrayJson(changedBoundaryContracts)
        << ",\"changedBoundaryNeighborhoods\":" << stringArrayJson(changedBoundaryNeighborhoods) << '}';
    return out.str();
}

ModelChangeSet ModelDiffEngine::compare(const ModelVersionManifestV4& oldModel,
                                        const ModelVersionManifestV4& newModel) {
    const auto start=std::chrono::steady_clock::now(); ModelChangeSet result;
    result.oldContentHash=oldModel.contentHash; result.newContentHash=newModel.contentHash;
    result.contentChanged=oldModel.contentHash!=newModel.contentHash;
    result.structureChanged=oldModel.structureHash!=newModel.structureHash;
    result.protocolChanged=oldModel.tensorPacketProtocolVersion!=newModel.tensorPacketProtocolVersion;
    std::set<std::string> keys;
    for(const auto& e:oldModel.artifactHashes)keys.insert(e.first);
    for(const auto& e:newModel.artifactHashes)keys.insert(e.first);
    for(const auto& key:keys){
        const auto a=oldModel.artifactHashes.find(key), b=newModel.artifactHashes.find(key);
        if(a==oldModel.artifactHashes.end()||b==newModel.artifactHashes.end()||a->second!=b->second){
            result.changedArtifacts.insert(key);
            if(key.find("config")!=std::string::npos||key.find("preprocess")!=std::string::npos) result.configChanged=true;
        }
    }
    std::map<std::string,const VersionedBoundary*> oldById,newById;
    for(const auto& b:oldModel.boundaries)oldById[b.canonicalId]=&b;
    for(const auto& b:newModel.boundaries)newById[b.canonicalId]=&b;
    for(const auto& e:oldById)if(!newById.count(e.first))result.removedBoundaries.insert(e.first);
    for(const auto& e:newById)if(!oldById.count(e.first))result.addedBoundaries.insert(e.first);
    for(const auto& e:oldById){auto n=newById.find(e.first);if(n==newById.end())continue;
        if(e.second->contractSignature()!=n->second->contractSignature())result.changedBoundaryContracts.insert(e.first);
        else if(e.second->semanticFingerprint()!=n->second->semanticFingerprint())result.changedBoundaryNeighborhoods.insert(e.first);
    }
    if(result.protocolChanged)result.preliminaryLevel=ImpactLevel::I4_GLOBAL;
    else if(!result.addedBoundaries.empty()||!result.removedBoundaries.empty()||!result.changedBoundaryContracts.empty())
        result.preliminaryLevel=ImpactLevel::I3_INTERFACE;
    else if(result.structureChanged||!result.changedBoundaryNeighborhoods.empty())result.preliminaryLevel=ImpactLevel::I2_LOCAL_COMPUTE;
    else if(result.contentChanged||!result.changedArtifacts.empty()||result.configChanged)result.preliminaryLevel=ImpactLevel::I1_PARAMETERS;
    result.diffMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); return result;
}

bool DecisionDependencyGraph::add(const DecisionArtifactDependency& artifact) {
    if(artifact.artifactId.empty()||artifacts_.count(artifact.artifactId))return false;
    artifacts_[artifact.artifactId]=artifact; return true;
}
void DecisionDependencyGraph::clear(){artifacts_.clear();}
std::size_t DecisionDependencyGraph::size()const{return artifacts_.size();}

ImpactPlan DecisionDependencyGraph::analyze(const ModelChangeSet& changes, bool minimal) const {
    const auto start=std::chrono::steady_clock::now(); ImpactPlan plan; plan.level=changes.preliminaryLevel;
    const auto seeds=changes.dependencySeeds(); std::set<std::string> invalid;
    if(!minimal){for(const auto& e:artifacts_)invalid.insert(e.first);plan.reason="full_invalidation_disabled";}
    else if(changes.protocolChanged){
        // A protocol mismatch is global: no old split artifact is safe to reuse.
        for(const auto& e:artifacts_)invalid.insert(e.first);
        plan.reason="global_protocol_incompatibility";
    }
    else {
        bool progress=true;
        while(progress){progress=false;for(const auto& e:artifacts_){if(invalid.count(e.first))continue;
            bool affected=false;for(const auto& dep:e.second.dependencies){
                for(const auto& seed:seeds)if(dependencyMatches(dep,seed)){affected=true;break;}
                if(affected)break;
                if(dep.compare(0,9,"artifact:")==0&&invalid.count(dep.substr(9))){affected=true;break;}
            }
            if(affected){invalid.insert(e.first);progress=true;}
        }}
        plan.reason=invalid.empty()?"no_decision_dependency_reached":"minimal_dependency_closure";
    }
    for(const auto& e:artifacts_){
        if(invalid.count(e.first)){plan.invalidatedArtifacts.insert(e.first);if(!e.second.actionKey.empty())plan.invalidatedActions.insert(e.second.actionKey);
            if(e.second.kind==DecisionArtifactKind::ACCURACY_PROFILE)plan.requiredProbes.insert("accuracy:"+e.first);
            else if(e.second.kind==DecisionArtifactKind::CODEC_PROFILE)plan.requiredProbes.insert("codec:"+e.first);
            else plan.requiredProbes.insert("latency:"+e.first);
        } else {plan.reusableArtifacts.insert(e.first);if(!e.second.actionKey.empty())plan.reusableActions.insert(e.second.actionKey);}
    }
    for (const auto& action : plan.invalidatedActions) plan.reusableActions.erase(action);
    plan.closureRatio=artifacts_.empty()?0.0:static_cast<double>(invalid.size())/artifacts_.size();
    plan.analysisMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(); return plan;
}

std::string ImpactPlan::toJson() const {
    std::ostringstream out;out<<"{\"schemaVersion\":"<<schemaVersion<<",\"level\":\""<<impactLevelName(level)
      <<"\",\"closureRatio\":"<<closureRatio<<",\"analysisMs\":"<<analysisMs<<",\"reason\":\""<<escapeJson(reason)
      <<"\",\"reusableArtifacts\":"<<stringArrayJson(reusableArtifacts)<<",\"invalidatedArtifacts\":"<<stringArrayJson(invalidatedArtifacts)
      <<",\"requiredProbes\":"<<stringArrayJson(requiredProbes)<<",\"reusableActions\":"<<stringArrayJson(reusableActions)
      <<",\"invalidatedActions\":"<<stringArrayJson(invalidatedActions)<<'}';return out.str();
}

CompatibilityMap BoundaryCompatibilityMapper::build(const ModelVersionManifestV4& oldModel,
                                                     const ModelVersionManifestV4& newModel,
                                                     const UpdateConfig& config) {
    const auto start=std::chrono::steady_clock::now();CompatibilityMap result;
    result.oldModelHash=oldModel.contentHash;result.newModelHash=newModel.contentHash;
    std::set<std::string> used;
    for(const auto& oldB:oldModel.boundaries){BoundaryMapping mapping;mapping.oldCanonicalId=oldB.canonicalId;mapping.oldPhysicalId=oldB.physicalId;
        const VersionedBoundary* best=nullptr;double bestScore=-1.0;
        // Preserve semantic identity first. Equal tensor shapes alone cannot
        // establish identity when CNN blocks repeat.
        const VersionedBoundary* sameId=newModel.findBoundary(oldB.canonicalId);
        if(sameId&&!used.count(sameId->canonicalId)) best=sameId;
        const auto scoreCandidate=[&](const VersionedBoundary& newB){
            const bool contract=oldB.contractSignature()==newB.contractSignature();
            double score=contract?0.55:0.0;
            if(oldB.upstreamNeighborhoodHash==newB.upstreamNeighborhoodHash)score+=0.15;
            if(oldB.downstreamNeighborhoodHash==newB.downstreamNeighborhoodHash)score+=0.15;
            if(oldB.totalOps>0&&newB.totalOps>0){const double po=static_cast<double>(oldB.splitAfterOp)/oldB.totalOps;
                const double pn=static_cast<double>(newB.splitAfterOp)/newB.totalOps;score+=0.15*std::max(0.0,1.0-std::fabs(po-pn));}
            if(newB.canonicalId==oldB.canonicalId)score+=1e-6;
            return score;
        };
        if(best){
            const bool semanticExact=
                oldB.contractSignature()==best->contractSignature()&&
                oldB.upstreamNeighborhoodHash==best->upstreamNeighborhoodHash&&
                oldB.downstreamNeighborhoodHash==best->downstreamNeighborhoodHash;
            bestScore=semanticExact?1.0:scoreCandidate(*best);
        }
        mapping.score=std::max(0.0,std::min(1.0,bestScore));
        if(!config.compatibilityMappingEnabled||best==nullptr){mapping.kind=MappingKind::INCOMPATIBLE;mapping.reason="mapping_disabled_or_missing";}
        else {mapping.newCanonicalId=best->canonicalId;mapping.newPhysicalId=best->physicalId;
            const bool contract=oldB.contractSignature()==best->contractSignature();
            if(contract&&mapping.score>=config.exactMappingThreshold){mapping.kind=MappingKind::EXACT;mapping.reason="contract_and_neighborhood_exact";}
            else if(contract&&mapping.score>=config.compatibleMappingThreshold){mapping.kind=MappingKind::CONTRACT_COMPATIBLE;mapping.reason="contract_stable_local_graph_changed";}
            else if(contract){mapping.kind=MappingKind::CALIBRATION_REQUIRED;mapping.reason="contract_only_match";}
            else {mapping.kind=MappingKind::INCOMPATIBLE;mapping.reason="cutset_contract_changed";}
            mapping.adapterRequired=mapping.kind==MappingKind::CONTRACT_COMPATIBLE&&oldB.physicalId!=best->physicalId;
            if(mapping.kind!=MappingKind::INCOMPATIBLE)used.insert(best->canonicalId);
        }
        result.boundaries.push_back(mapping);
    }
    result.mappingMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return result;
}

std::string CompatibilityMap::toJson() const {
    std::ostringstream out;out<<"{\"schemaVersion\":"<<schemaVersion<<",\"oldModelHash\":\""<<escapeJson(oldModelHash)
      <<"\",\"newModelHash\":\""<<escapeJson(newModelHash)<<"\",\"mappingMs\":"<<mappingMs<<",\"boundaries\":[";
    for(size_t i=0;i<boundaries.size();++i){if(i)out<<',';const auto& b=boundaries[i];out<<"{\"oldCanonicalId\":\""<<escapeJson(b.oldCanonicalId)
      <<"\",\"oldPhysicalId\":\""<<escapeJson(b.oldPhysicalId)<<"\",\"newCanonicalId\":\""<<escapeJson(b.newCanonicalId)
      <<"\",\"newPhysicalId\":\""<<escapeJson(b.newPhysicalId)<<"\",\"kind\":\""<<mappingKindName(b.kind)
      <<"\",\"score\":"<<b.score<<",\"adapterRequired\":"<<(b.adapterRequired?"true":"false")<<",\"reason\":\""<<escapeJson(b.reason)<<"\"}";}
    out<<"]}";return out.str();
}

StateMigrationPlan ControllerStateMigrator::plan(const CompatibilityMap& mappings,
                                                  const ModelChangeSet& changes, bool enabled) {
    const auto start=std::chrono::steady_clock::now();StateMigrationPlan result;
    auto add=[&](std::string key,std::string oldKey,std::string newKey,double rho,bool reset,std::string reason){
        result.entries.push_back({std::move(key),std::move(oldKey),std::move(newKey),rho,reset,std::move(reason)});
        if(reset)++result.resetCount;else ++result.reusedCount;};
    add("network.residual","global","global",enabled?1.0:0.0,!enabled,enabled?"model_independent":"migration_disabled");
    add("cloud.queue.residual","global","global",enabled?1.0:0.0,!enabled,enabled?"model_independent":"migration_disabled");
    for(const auto& mapping:mappings.boundaries){double compute=0.0;bool reset=true;
        if(enabled&&mapping.kind==MappingKind::EXACT){compute=changes.structureChanged?0.8:1.0;reset=false;}
        else if(enabled&&mapping.kind==MappingKind::CONTRACT_COMPATIBLE){compute=0.5;reset=false;}
        add("compute.residual",mapping.oldCanonicalId,mapping.newCanonicalId,compute,reset,mapping.reason);
        const bool codecReusable=enabled&&mapping.kind==MappingKind::EXACT&&!changes.contentChanged;
        add("codec.residual",mapping.oldCanonicalId,mapping.newCanonicalId,codecReusable?1.0:0.0,!codecReusable,
            codecReusable?"activation_distribution_unchanged":"weights_or_contract_require_recheck");
    }
    result.hysteresisMigrated=enabled&&!mappings.boundaries.empty()&&std::all_of(mappings.boundaries.begin(),mappings.boundaries.end(),
        [](const BoundaryMapping& m){return m.kind==MappingKind::EXACT;});
    double sum=0;for(const auto& e:result.entries)sum+=e.reuseFactor;result.averageReuseFactor=result.entries.empty()?0:sum/result.entries.size();
    result.migrationMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return result;
}

std::string StateMigrationPlan::toJson() const {
    std::ostringstream out;out<<"{\"schemaVersion\":"<<schemaVersion<<",\"reusedCount\":"<<reusedCount<<",\"resetCount\":"<<resetCount
      <<",\"averageReuseFactor\":"<<averageReuseFactor<<",\"hysteresisMigrated\":"<<(hysteresisMigrated?"true":"false")
      <<",\"migrationMs\":"<<migrationMs<<",\"entries\":[";
    for(size_t i=0;i<entries.size();++i){if(i)out<<',';const auto&e=entries[i];out<<"{\"stateKey\":\""<<escapeJson(e.stateKey)
      <<"\",\"oldActionKey\":\""<<escapeJson(e.oldActionKey)<<"\",\"newActionKey\":\""<<escapeJson(e.newActionKey)
      <<"\",\"reuseFactor\":"<<e.reuseFactor<<",\"reset\":"<<(e.reset?"true":"false")<<",\"reason\":\""<<escapeJson(e.reason)<<"\"}";}
    out<<"]}";return out.str();
}

void IncrementalQualificationManager::setConfig(const UpdateConfig& config){std::lock_guard<std::mutex> lock(mutex_);config_=config;}
void IncrementalQualificationManager::seed(const std::string& key, MappingKind mapping){std::lock_guard<std::mutex> lock(mutex_);auto&r=records_[key];
    r=QualificationRecord();if(mapping==MappingKind::EXACT)r.state=QualificationState::SHADOW_READY;
    else if(mapping==MappingKind::INCOMPATIBLE)r.state=QualificationState::UNAVAILABLE;else r.state=QualificationState::PROFILING;}
QualificationState IncrementalQualificationManager::observe(const std::string& key,bool success,double error,double accuracy){std::lock_guard<std::mutex> lock(mutex_);auto&r=records_[key];
    if(r.state==QualificationState::QUARANTINED)return r.state;++r.samples;if(!success)++r.failures;
    if(std::isfinite(error))r.absoluteErrorSumMs+=std::fabs(error);if(std::isfinite(accuracy))r.maximumAccuracyDrop=std::max(r.maximumAccuracyDrop,accuracy);
    const double mae=r.samples?r.absoluteErrorSumMs/r.samples:std::numeric_limits<double>::infinity();
    if(!success||r.maximumAccuracyDrop>config_.qualificationMaxAccuracyDrop)r.state=QualificationState::QUARANTINED;
    else if(r.samples>=config_.qualificationMinimumSamples)r.state=mae<=config_.qualificationMaxMaeMs?QualificationState::SHADOW_READY:QualificationState::QUARANTINED;
    else r.state=QualificationState::PROFILING;return r.state;}
QualificationRecord IncrementalQualificationManager::get(const std::string& key)const{std::lock_guard<std::mutex> lock(mutex_);auto i=records_.find(key);return i==records_.end()?QualificationRecord():i->second;}
bool IncrementalQualificationManager::activate(const std::string& key){std::lock_guard<std::mutex> lock(mutex_);auto i=records_.find(key);if(i==records_.end()||i->second.state!=QualificationState::SHADOW_READY)return false;i->second.state=QualificationState::ACTIVE;return true;}
void IncrementalQualificationManager::quarantine(const std::string& key){std::lock_guard<std::mutex> lock(mutex_);records_[key].state=QualificationState::QUARANTINED;}
void IncrementalQualificationManager::clearModel(const std::string& fragment){std::lock_guard<std::mutex> lock(mutex_);for(auto i=records_.begin();i!=records_.end();)if(i->first.find(fragment)!=std::string::npos)i=records_.erase(i);else ++i;}

void VersionedUpdateCoordinator::setConfig(const UpdateConfig& c){std::lock_guard<std::mutex>lock(mutex_);config_=c;}
UpdateConfig VersionedUpdateCoordinator::config()const{std::lock_guard<std::mutex>lock(mutex_);return config_;}
bool VersionedUpdateCoordinator::bootstrap(const ModelVersionManifestV4& active,std::string* error){std::string e;if(!active.valid(&e)){if(error)*error=e;return false;}std::lock_guard<std::mutex>lock(mutex_);active_=active;hasActive_=true;return true;}
bool VersionedUpdateCoordinator::prepare(const std::string& id,const ModelVersionManifestV4& staged,const ModelChangeSet& changes,const ImpactPlan& impact,const CompatibilityMap& compatibility,const StateMigrationPlan& migration,std::string* error){
    std::string e;if(id.empty()||!staged.valid(&e)){if(error)*error=id.empty()?"transaction id required":e;return false;}std::lock_guard<std::mutex>lock(mutex_);
    if(transaction_.state==UpdateTransactionState::PREPARING||transaction_.state==UpdateTransactionState::PREPARED){if(transaction_.transactionId==id)return true;if(error)*error="another update is in progress";return false;}
    staged_=staged;hasStaged_=true;transaction_=UpdateTransaction();transaction_.transactionId=id;transaction_.state=UpdateTransactionState::PREPARING;
    transaction_.stagedManifest=staged;transaction_.changes=changes;transaction_.impact=impact;transaction_.compatibility=compatibility;transaction_.migration=migration;
    transaction_.startedUnixMs=transaction_.updatedUnixMs=unixMs();return true;}
bool VersionedUpdateCoordinator::markReady(const std::string&id,bool edge,bool cloud,std::string*error){std::lock_guard<std::mutex>lock(mutex_);
    if(transaction_.transactionId!=id||!hasStaged_){if(error)*error="unknown transaction";return false;}if(edge)transaction_.edgeReady=true;if(cloud)transaction_.cloudReady=true;
    if(transaction_.edgeReady&&transaction_.cloudReady)transaction_.state=UpdateTransactionState::PREPARED;transaction_.updatedUnixMs=unixMs();return true;}
bool VersionedUpdateCoordinator::commit(const std::string&id,std::string*error){std::lock_guard<std::mutex>lock(mutex_);
    if(transaction_.transactionId==id&&transaction_.state==UpdateTransactionState::COMMITTED)return true;
    if(transaction_.transactionId!=id||transaction_.state!=UpdateTransactionState::PREPARED||!hasStaged_){if(error)*error="transaction is not prepared on both sides";return false;}
    if(hasActive_){rollback_=active_;hasRollback_=true;}active_=staged_;hasActive_=true;hasStaged_=false;staged_=ModelVersionManifestV4();
    transaction_.state=UpdateTransactionState::COMMITTED;transaction_.updatedUnixMs=unixMs();return true;}
bool VersionedUpdateCoordinator::abort(const std::string&id,const std::string&reason,std::string*error){std::lock_guard<std::mutex>lock(mutex_);
    if(transaction_.transactionId!=id){if(error)*error="unknown transaction";return false;}hasStaged_=false;staged_=ModelVersionManifestV4();transaction_.state=UpdateTransactionState::ABORTED;transaction_.failureReason=reason;transaction_.updatedUnixMs=unixMs();return true;}
bool VersionedUpdateCoordinator::rollback(const std::string&reason,std::string*error){std::lock_guard<std::mutex>lock(mutex_);if(!hasRollback_){if(error)*error="rollback slot empty";return false;}
    ModelVersionManifestV4 failed=active_;active_=rollback_;rollback_=failed;hasActive_=true;hasRollback_=true;transaction_.state=UpdateTransactionState::ROLLED_BACK;transaction_.failureReason=reason;transaction_.updatedUnixMs=unixMs();return true;}
bool VersionedUpdateCoordinator::splitCompatible(const std::string&edge,const std::string&cloud,const std::string&manifest,uint32_t protocol,std::string*reason)const{std::lock_guard<std::mutex>lock(mutex_);
    auto reject=[&](const std::string&r){if(reason)*reason=r;return false;};if(!hasActive_)return reject("no_active_model");if(config_.mode==UpdateMode::SAFE_ONLY)return reject("safe_only_mode");
    if(edge!=active_.contentHash||cloud!=active_.contentHash)return reject("content_hash_mismatch");if(manifest!=active_.manifestId)return reject("manifest_mismatch");if(protocol!=active_.tensorPacketProtocolVersion)return reject("packet_protocol_mismatch");return true;}
bool VersionedUpdateCoordinator::splitAllowed()const{std::lock_guard<std::mutex>lock(mutex_);return hasActive_&&config_.mode!=UpdateMode::SAFE_ONLY;}
bool VersionedUpdateCoordinator::hasActive()const{std::lock_guard<std::mutex>lock(mutex_);return hasActive_;}
ModelVersionManifestV4 VersionedUpdateCoordinator::active()const{std::lock_guard<std::mutex>lock(mutex_);return active_;}
ModelVersionManifestV4 VersionedUpdateCoordinator::staged()const{std::lock_guard<std::mutex>lock(mutex_);return staged_;}
ModelVersionManifestV4 VersionedUpdateCoordinator::rollbackSlot()const{std::lock_guard<std::mutex>lock(mutex_);return rollback_;}
UpdateTransaction VersionedUpdateCoordinator::transaction()const{std::lock_guard<std::mutex>lock(mutex_);return transaction_;}
std::string VersionedUpdateCoordinator::statusJson()const{std::lock_guard<std::mutex>lock(mutex_);std::ostringstream out;out<<"{\"schemaVersion\":1,\"mode\":\""<<updateModeName(config_.mode)
    <<"\",\"transactionState\":\""<<updateTransactionStateName(transaction_.state)<<"\",\"transactionId\":\""<<escapeJson(transaction_.transactionId)
    <<"\",\"hasActive\":"<<(hasActive_?"true":"false")<<",\"hasStaged\":"<<(hasStaged_?"true":"false")<<",\"hasRollback\":"<<(hasRollback_?"true":"false")
    <<",\"activeContentHash\":\""<<escapeJson(active_.contentHash)<<"\",\"activeManifestId\":\""<<escapeJson(active_.manifestId)
    <<"\",\"stagedContentHash\":\""<<escapeJson(staged_.contentHash)<<"\",\"edgeReady\":"<<(transaction_.edgeReady?"true":"false")
    <<",\"cloudReady\":"<<(transaction_.cloudReady?"true":"false")<<",\"splitAllowed\":"<<((hasActive_&&config_.mode!=UpdateMode::SAFE_ONLY)?"true":"false")<<'}';return out.str();}

std::string UpdateAuditRecord::toJson()const{std::ostringstream out;out<<"{\"schemaVersion\":"<<schemaVersion<<",\"timestampUnixMs\":"<<timestampUnixMs
    <<",\"transactionId\":\""<<escapeJson(transactionId)<<"\",\"event\":\""<<escapeJson(event)<<"\",\"oldModelHash\":\""<<escapeJson(oldModelHash)
    <<"\",\"newModelHash\":\""<<escapeJson(newModelHash)<<"\",\"durationMs\":"<<durationMs<<",\"detail\":"<<(detailJson.empty()?"{}":detailJson)<<'}';return out.str();}
bool JsonlUpdateAuditSink::append(const UpdateAuditRecord&record){std::lock_guard<std::mutex>lock(mutex_);std::ofstream out(path_,std::ios::app);if(!out)return false;out<<record.toJson()<<'\n';return static_cast<bool>(out);}
