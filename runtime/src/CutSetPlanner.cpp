#include "CutSetPlanner.h"
#include "VersionedUpdateManager.h"

#include <cstdio>
#include <map>
#include <sstream>
#include <vector>

#ifdef HAS_MNN
#include <MNN_generated.h>
#include <flatbuffers/flatbuffers.h>
#endif

std::vector<std::string> CutSetPlan::tensorNames() const {
    std::vector<std::string> names;
    names.reserve(tensors.size());
    for (const auto& tensor : tensors) names.push_back(tensor.tensorName);
    return names;
}

std::string CutSetPlan::cacheKey() const {
    std::ostringstream oss;
    oss << modelFingerprint << ':' << splitAfterOp << ':';
    for (const auto& tensor : tensors) {
        oss << tensor.tensorIndex << ':' << tensor.tensorName.size() << ':' << tensor.tensorName << ';';
    }
    return oss.str();
}

#ifdef HAS_MNN
namespace {

bool loadFile(const std::string& path, std::vector<unsigned char>& data) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) return false;
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(file);
        return false;
    }
    data.resize(static_cast<size_t>(size));
    bool ok = std::fread(data.data(), 1, data.size(), file) == data.size();
    std::fclose(file);
    return ok;
}

uint64_t fnv1a64(const std::vector<unsigned char>& data) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned char value : data) {
        hash ^= static_cast<uint64_t>(value);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool isCloudLocalConstant(const MNN::Op* op) {
    return op != nullptr && (op->type() == MNN::OpType_Const || op->type() == MNN::OpType_TrainableParam);
}

std::string getOpName(const MNN::Op* op, int index) {
    if (op != nullptr && op->name() != nullptr) return op->name()->str();
    return std::string("op_") + std::to_string(index);
}

}  // namespace
#endif

bool CutSetPlanner::fingerprintModel(const std::string& modelPath,
                                     uint64_t& fingerprint,
                                     std::string& error) {
    fingerprint = 0;
    error.clear();
#ifdef HAS_MNN
    std::vector<unsigned char> model;
    if (!loadFile(modelPath, model)) {
        error = "cannot read model";
        return false;
    }
    fingerprint = fnv1a64(model);
    if (fingerprint == 0) {
        error = "invalid zero model fingerprint";
        return false;
    }
    return true;
#else
    (void)modelPath;
    error = "MNN support is disabled";
    return false;
#endif
}

bool CutSetPlanner::fingerprintStructure(const std::string& modelPath,
                                         std::string& fingerprint,
                                         std::string& error) {
    fingerprint.clear();
    error.clear();
#ifdef HAS_MNN
    std::vector<unsigned char> model;
    if (!loadFile(modelPath, model)) { error = "cannot read model"; return false; }
    flatbuffers::Verifier verifier(model.data(), model.size());
    if (!MNN::VerifyNetBuffer(verifier)) { error = "invalid MNN flatbuffer"; return false; }
    const auto* net = MNN::GetNet(model.data());
    if (net == nullptr || net->oplists() == nullptr || net->tensorName() == nullptr) {
        error = "model has no op/tensor table";
        return false;
    }
    std::ostringstream canonical;
    canonical << "mnn-structure-v1|ops=" << net->oplists()->size()
              << "|tensors=" << net->tensorName()->size() << '|';
    for (int i = 0; i < static_cast<int>(net->tensorName()->size()); ++i) {
        const auto* name = net->tensorName()->Get(i);
        canonical << "t" << i << ':' << (name ? name->str() : std::string()) << ';';
    }
    for (int i = 0; i < static_cast<int>(net->oplists()->size()); ++i) {
        const auto* op = net->oplists()->Get(i);
        if (!op) { canonical << "null;"; continue; }
        canonical << "o" << i << ':' << static_cast<int>(op->type())
                  << ':' << static_cast<int>(op->main_type()) << ":i";
        if (op->inputIndexes()) for (int j = 0; j < static_cast<int>(op->inputIndexes()->size()); ++j) {
            canonical << op->inputIndexes()->Get(j) << ',';
        }
        canonical << ":o";
        if (op->outputIndexes()) for (int j = 0; j < static_cast<int>(op->outputIndexes()->size()); ++j) {
            canonical << op->outputIndexes()->Get(j) << ',';
        }
        canonical << ';';
    }
    fingerprint = VersionHasher::sha256(canonical.str());
    return true;
#else
    (void)modelPath;
    error = "MNN support is disabled";
    return false;
#endif
}

bool CutSetPlanner::buildFromModel(const std::string& modelPath,
                                   int splitAfterOp,
                                   CutSetPlan& plan,
                                   std::string& error) {
    plan = CutSetPlan();
    error.clear();
#ifdef HAS_MNN
    std::vector<unsigned char> model;
    if (!loadFile(modelPath, model)) {
        error = "cannot read model";
        return false;
    }
    plan.modelFingerprint = fnv1a64(model);
    flatbuffers::Verifier verifier(model.data(), model.size());
    if (!MNN::VerifyNetBuffer(verifier)) {
        error = "invalid MNN flatbuffer";
        return false;
    }
    const auto* net = MNN::GetNet(model.data());
    if (net == nullptr || net->oplists() == nullptr || net->tensorName() == nullptr) {
        error = "model has no op/tensor table";
        return false;
    }
    const int opCount = static_cast<int>(net->oplists()->size());
    const int tensorCount = static_cast<int>(net->tensorName()->size());
    if (splitAfterOp < 0 || splitAfterOp >= opCount - 1) {
        error = "splitAfterOp must leave at least one cloud op";
        return false;
    }

    std::vector<int> producer(tensorCount, -1);
    std::vector<int> producerSlot(tensorCount, -1);
    for (int opIndex = 0; opIndex < opCount; ++opIndex) {
        const auto* op = net->oplists()->Get(opIndex);
        if (op == nullptr || op->outputIndexes() == nullptr) continue;
        for (int slot = 0; slot < static_cast<int>(op->outputIndexes()->size()); ++slot) {
            int tensorIndex = op->outputIndexes()->Get(slot);
            if (tensorIndex >= 0 && tensorIndex < tensorCount) {
                producer[tensorIndex] = opIndex;
                producerSlot[tensorIndex] = slot;
            }
        }
    }

    std::map<int, CutTensorDescriptor> cutByTensor;
    for (int consumerIndex = splitAfterOp + 1; consumerIndex < opCount; ++consumerIndex) {
        const auto* consumer = net->oplists()->Get(consumerIndex);
        if (consumer == nullptr || consumer->inputIndexes() == nullptr) continue;
        for (int slot = 0; slot < static_cast<int>(consumer->inputIndexes()->size()); ++slot) {
            int tensorIndex = consumer->inputIndexes()->Get(slot);
            if (tensorIndex < 0 || tensorIndex >= tensorCount) continue;
            int producerIndex = producer[tensorIndex];
            if (producerIndex > splitAfterOp) continue;
            const auto* producerOp = producerIndex >= 0 ? net->oplists()->Get(producerIndex) : nullptr;
            if (isCloudLocalConstant(producerOp)) continue;

            auto it = cutByTensor.find(tensorIndex);
            if (it == cutByTensor.end()) {
                CutTensorDescriptor descriptor;
                descriptor.tensorIndex = tensorIndex;
                descriptor.tensorName = net->tensorName()->Get(tensorIndex)->str();
                descriptor.producerOpIndex = producerIndex;
                descriptor.producerOpName = producerIndex >= 0 ? getOpName(producerOp, producerIndex) : "<model-input>";
                descriptor.producerOutputSlot = producerSlot[tensorIndex];
                descriptor.modelInput = producerIndex < 0 ||
                                        (producerOp != nullptr && producerOp->type() == MNN::OpType_Input);
                it = cutByTensor.emplace(tensorIndex, std::move(descriptor)).first;
            }
            CutConsumerBinding binding;
            binding.opIndex = consumerIndex;
            binding.opName = getOpName(consumer, consumerIndex);
            binding.inputSlot = slot;
            it->second.consumers.push_back(std::move(binding));
        }
    }

    if (cutByTensor.empty()) {
        error = "empty cut-set: disconnected split or non-topological op order";
        return false;
    }
    plan.splitAfterOp = splitAfterOp;
    plan.totalOps = opCount;
    for (auto& item : cutByTensor) plan.tensors.push_back(std::move(item.second));
    return true;
#else
    (void)modelPath;
    (void)splitAfterOp;
    error = "MNN support is disabled";
    return false;
#endif
}
