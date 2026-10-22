#include "CloudInjectionEngine.h"
#include "ActivationCodec.h"
#include "TensorSerializer.h"

#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <map>

#ifdef HAS_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <flatbuffers/flatbuffers.h>
#include <MNN_generated.h>
#endif

CloudInjectionEngine::CloudInjectionEngine() = default;
CloudInjectionEngine::~CloudInjectionEngine() { unload(); }

// ============================================================
//  loadModel — load a full-model session shared by all split boundaries
// ============================================================

bool CloudInjectionEngine::loadModel(const std::string& modelPath,
                                     const CutSetPlan& cutSetPlan,
                                     const std::string& outputName,
                                     int backendType,
                                     int numThread) {
#ifdef HAS_MNN
    unload();
    if (cutSetPlan.tensors.empty() || cutSetPlan.modelFingerprint == 0 || outputName.empty()) {
        std::cerr << "[CloudInject] empty cut-set/output" << std::endl;
        return false;
    }
    modelPath_ = modelPath;
    splitInputs_ = cutSetPlan.tensorNames();
    outputName_ = outputName;
    splitAfterOp_ = cutSetPlan.splitAfterOp;
    modelFingerprint_ = cutSetPlan.modelFingerprint;

    auto* net = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (net == nullptr) {
        std::cerr << "[CloudInject] createFromFile failed: " << modelPath << std::endl;
        return false;
    }
    interpreter_.reset(net);

    // Resolve exact graph consumers for every cut tensor. Injection is keyed
    // by (consumer op name, input slot), never by shape.
    FILE* modelFile = fopen(modelPath.c_str(), "rb");
    if (!modelFile) return false;
    fseek(modelFile, 0, SEEK_END);
    const long modelSize = ftell(modelFile);
    fseek(modelFile, 0, SEEK_SET);
    if (modelSize <= 0) {
        fclose(modelFile);
        return false;
    }
    std::vector<uint8_t> modelBytes(static_cast<size_t>(modelSize));
    const bool modelRead = fread(modelBytes.data(), 1, modelBytes.size(), modelFile) == modelBytes.size();
    fclose(modelFile);
    if (!modelRead) return false;
    const auto* model = MNN::GetNet(modelBytes.data());
    if (!model || !model->tensorName() || !model->oplists()) return false;
    opNameToIndex_.clear();
    for (int i = 0; i < static_cast<int>(model->oplists()->size()); ++i) {
        const auto* op = model->oplists()->Get(i);
        if (op && op->name()) opNameToIndex_[op->name()->str()] = i;
    }
    injectionBindings_.clear();
    cutTensorIndexes_.clear();
    for (const auto& cut : cutSetPlan.tensors) {
        cutTensorIndexes_[cut.tensorName] = cut.tensorIndex;
        for (const auto& consumer : cut.consumers) {
            injectionBindings_.push_back({cut.tensorName, consumer.opName, consumer.inputSlot});
        }
    }
    if (opNameToIndex_.empty() || injectionBindings_.empty()) {
        std::cerr << "[CloudInject] cannot resolve exact consumer bindings" << std::endl;
        unload();
        return false;
    }

    MNN::ScheduleConfig config;
    config.type = static_cast<MNNForwardType>(backendType);
    config.numThread = std::max(1, numThread);
    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) {
        std::cerr << "[CloudInject] createSession failed" << std::endl;
        interpreter_.reset();
        return false;
    }
    loaded_ = true;
    std::cout << "[CloudInject] selective suffix session loaded, splitAfter=" << splitAfterOp_
              << " cutTensors=" << splitInputs_.size()
              << " bindings=" << injectionBindings_.size() << " out=" << outputName_
              << " backend=" << backendType << " threads=" << config.numThread << std::endl;
    return true;
#else
    (void)modelPath; (void)cutSetPlan; (void)outputName;
    return false;
#endif
}

// ============================================================
//  runBackHalf — full graph with before-callback injection of T_k into the suffix
// ============================================================

bool CloudInjectionEngine::runBackHalf(const std::vector<uint8_t>& tLBytes,
                                       std::vector<float>& outputData,
                                       std::vector<int>& outputShape,
                                       double& cloudTimeMs) {
    CloudFrameworkProfile profile;
    const bool ok = runBackHalfProfile(tLBytes, outputData, outputShape, profile);
    cloudTimeMs = profile.sessionMs;
    return ok;
}

bool CloudInjectionEngine::runBackHalfProfile(const std::vector<uint8_t>& tLBytes,
                                              std::vector<float>& outputData,
                                              std::vector<int>& outputShape,
                                              CloudFrameworkProfile& profile) {
#ifdef HAS_MNN
    profile = CloudFrameworkProfile();
    using ProfileClock = std::chrono::steady_clock;
    const auto totalStart = ProfileClock::now();
    if (!loaded_ || interpreter_ == nullptr || session_ == nullptr) {
        std::cerr << "[CloudInject] not loaded" << std::endl;
        return false;
    }

    std::vector<TensorPacket> packets;
    const auto deserializeStart = ProfileClock::now();
    CutBundleMetadata metadata;
    std::string bundleError;
    uint32_t bundleVersion = 0;
    if (!TensorSerializer::inspectBundleVersion(tLBytes, bundleVersion)) {
        std::cerr << "[CloudInject] cut-set bundle header invalid" << std::endl;
        return false;
    }
    if (bundleVersion == TensorSerializer::CODEC_BUNDLE_VERSION) {
        std::vector<EncodedTensorPacket> encoded;
        if (!TensorSerializer::deserializeEncodedBundle(tLBytes, encoded, &bundleError, &metadata) ||
            !ActivationCodec::decodeBundle(encoded, packets, profile.codec, &bundleError)) {
            std::cerr << "[CloudInject] codec bundle decode failed: " << bundleError << std::endl;
            return false;
        }
        profile.codec.bundleBytes = tLBytes.size();
    } else {
        if (!TensorSerializer::deserializeBundle(tLBytes, packets, &bundleError, &metadata)) {
            std::cerr << "[CloudInject] cut-set bundle parse failed: " << bundleError << std::endl;
            return false;
        }
        profile.codec.enabled = false;
        profile.codec.requested = CodecKind::RAW32;
        profile.codec.tensorCount = static_cast<uint32_t>(packets.size());
        for (const auto& packet : packets) profile.codec.rawTensorBytes += packet.data.size();
        profile.codec.encodedTensorBytes = profile.codec.rawTensorBytes;
        profile.codec.bundleBytes = tLBytes.size();
    }
    if (metadata.valid() && (metadata.splitAfterOp != splitAfterOp_ ||
                             metadata.modelFingerprint != modelFingerprint_)) {
        std::cerr << "[CloudInject] stale or mismatched cut bundle" << std::endl;
        return false;
    }
    if (packets.size() != splitInputs_.size()) {
        std::cerr << "[CloudInject] cut-set cardinality mismatch" << std::endl;
        return false;
    }

    std::map<std::string, std::unique_ptr<MNN::Tensor>> hosts;
    for (const auto& packet : packets) {
        if (std::find(splitInputs_.begin(), splitInputs_.end(), packet.name) == splitInputs_.end()) {
            std::cerr << "[CloudInject] undeclared cut tensor " << packet.name << std::endl;
            return false;
        }
        if (packet.layout != 0) {
            std::cerr << "[CloudInject] consumer injection requires canonical NCHW" << std::endl;
            return false;
        }
        std::unique_ptr<MNN::Tensor> host(
            MNN::Tensor::create<float>(packet.shape, nullptr, MNN::Tensor::CAFFE));
        if (host == nullptr || host->host<uint8_t>() == nullptr ||
            static_cast<size_t>(host->size()) != packet.data.size()) {
            std::cerr << "[CloudInject] physical tensor size mismatch for " << packet.name << std::endl;
            return false;
        }
        std::memcpy(host->host<uint8_t>(), packet.data.data(), packet.data.size());
        hosts[packet.name] = std::move(host);
    }
    profile.deserializeGuardMs = std::chrono::duration<double, std::milli>(
        ProfileClock::now() - deserializeStart).count();

    bool injectionFailed = false;
    auto before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        if (!info) return false;
        auto op = opNameToIndex_.find(info->originalName());
        const bool run = op != opNameToIndex_.end() && op->second > splitAfterOp_;
        if (!run) return false;
        for (const auto& cut : cutTensorIndexes_) {
            auto host = hosts.find(cut.first);
            auto* target = interpreter_->getSessionTensor(session_, cut.second);
            if (host == hosts.end() || target == nullptr) {
                injectionFailed = true;
                continue;
            }
            for (auto* tensor : tensors) {
                if (tensor == target) {
                    const auto copyStart = ProfileClock::now();
                    const bool copied = tensor->copyFromHostTensor(host->second.get());
                    profile.injectionCallbackMs += std::chrono::duration<double, std::milli>(
                        ProfileClock::now() - copyStart).count();
                    ++profile.injectionCopyCount;
                    if (!copied) injectionFailed = true;
                }
            }
        }
        return true;
    };
    auto after = [](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; };
    auto t0 = ProfileClock::now();
    auto ret = interpreter_->runSessionWithCallBackInfoSelective(session_, before, after, false, true);
    auto t1 = ProfileClock::now();
    profile.sessionMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (profile.injectionCopyCount < cutTensorIndexes_.size()) injectionFailed = true;
    if (ret != MNN::NO_ERROR || injectionFailed) {
        std::cerr << "[CloudInject] run error " << ret
                  << " injectionCopies=" << profile.injectionCopyCount
                  << " expectedAtLeast=" << cutTensorIndexes_.size() << std::endl;
        return false;
    }

    auto* out = interpreter_->getSessionOutput(session_, outputName_.c_str());
    if (out == nullptr) out = interpreter_->getSessionOutput(session_, nullptr);
    if (out == nullptr) {
        std::cerr << "[CloudInject] no output " << outputName_ << std::endl;
        return false;
    }
    const auto outputCopyStart = ProfileClock::now();
    std::unique_ptr<MNN::Tensor> hostOutput(
        MNN::Tensor::create(out->shape(), out->getType(), nullptr, MNN::Tensor::CAFFE));
    if (!hostOutput || !out->copyToHostTensor(hostOutput.get())) {
        std::cerr << "[CloudInject] output host unavailable" << std::endl;
        return false;
    }
    outputShape = out->shape();
    int outElements = 1;
    for (int d : outputShape) outElements *= d;
    if (outElements <= 0) outElements = 1;
    outputData.assign(hostOutput->host<float>(), hostOutput->host<float>() + outElements);
    profile.outputCopyMs = std::chrono::duration<double, std::milli>(
        ProfileClock::now() - outputCopyStart).count();
    profile.totalMs = std::chrono::duration<double, std::milli>(
        ProfileClock::now() - totalStart).count();
    return true;
#else
    (void)tLBytes; (void)outputData; (void)outputShape; (void)profile;
    return false;
#endif
}

// ============================================================
//  unload
// ============================================================

void CloudInjectionEngine::unload() {
#ifdef HAS_MNN
    if (interpreter_ != nullptr && session_ != nullptr) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    interpreter_.reset();
#endif
    loaded_ = false;
    splitInputs_.clear();
    injectionBindings_.clear();
    cutTensorIndexes_.clear();
    opNameToIndex_.clear();
    splitAfterOp_ = -1;
    modelFingerprint_ = 0;
}

bool CloudInjectionEngine::loadModelLegacy(const std::string& modelPath,
                                           const std::vector<std::string>& splitInputs,
                                           const std::string& outputName,
                                           int backendType,
                                           int numThread) {
#ifdef HAS_MNN
    FILE* file = fopen(modelPath.c_str(), "rb");
    if (!file) return false;
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size <= 0) {
        fclose(file);
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    const bool read = fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    fclose(file);
    if (!read) return false;
    const auto* model = MNN::GetNet(bytes.data());
    if (!model || !model->tensorName() || !model->oplists()) return false;

    CutSetPlan legacy;
    legacy.splitAfterOp = -1;
    legacy.totalOps = static_cast<int>(model->oplists()->size());
    std::string fingerprintError;
    if (!CutSetPlanner::fingerprintModel(modelPath, legacy.modelFingerprint, fingerprintError)) return false;
    for (const auto& requestedName : splitInputs) {
        int tensorIndex = -1;
        for (int i = 0; i < static_cast<int>(model->tensorName()->size()); ++i) {
            const auto* name = model->tensorName()->Get(i);
            if (name && name->str() == requestedName) {
                tensorIndex = i;
                break;
            }
        }
        if (tensorIndex < 0) return false;
        CutTensorDescriptor descriptor;
        descriptor.tensorIndex = tensorIndex;
        descriptor.tensorName = requestedName;
        for (int opIndex = 0; opIndex < legacy.totalOps; ++opIndex) {
            const auto* op = model->oplists()->Get(opIndex);
            if (!op || !op->name() || !op->inputIndexes()) continue;
            for (int slot = 0; slot < static_cast<int>(op->inputIndexes()->size()); ++slot) {
                if (op->inputIndexes()->Get(slot) == tensorIndex) {
                    CutConsumerBinding binding;
                    binding.opIndex = opIndex;
                    binding.opName = op->name()->str();
                    binding.inputSlot = slot;
                    descriptor.consumers.push_back(std::move(binding));
                }
            }
        }
        if (descriptor.consumers.empty()) return false;
        legacy.tensors.push_back(std::move(descriptor));
    }
    return loadModel(modelPath, legacy, outputName, backendType, numThread);
#else
    (void)modelPath; (void)splitInputs; (void)outputName; (void)backendType; (void)numThread;
    return false;
#endif
}
