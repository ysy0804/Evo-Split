#include "SplitInferenceEngine.h"
#include "ActivationCodec.h"
#include "TensorSerializer.h"

#include <iostream>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <set>

#ifdef HAS_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <flatbuffers/flatbuffers.h>
#include <MNN_generated.h>
#endif

#ifdef HAS_MNN
// Parse the FlatBuffer op list and map layer IDs to op names for capture.
static std::map<int, std::string> buildLayerId2Name(const std::string& modelPath) {
    std::map<int, std::string> m;
    FILE* f = fopen(modelPath.c_str(), "rb");
    if (!f) return m;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(size);
    if (fread(buf.data(), 1, size, f) != (size_t)size) { fclose(f); return m; }
    fclose(f);
    auto* net = MNN::GetNet(buf.data());
    if (!net || !net->oplists()) return m;
    auto* ops = net->oplists();
    for (int i = 0; i < (int)ops->size(); ++i) {
        auto* op = ops->Get(i);
        if (op && op->name()) m[i] = op->name()->str();
    }
    return m;
}
#endif

SplitInferenceEngine::SplitInferenceEngine() = default;
SplitInferenceEngine::~SplitInferenceEngine() { unload(); }

// ============================================================
//  isValidSplitPoint -- validate a boundary against the MNN runtime graph
//  Create a lightweight interpreter and session, then query the output.
//  saveTensors asks MNN to retain the selected tensor.
//  A null result means that the tensor is unavailable.
// ============================================================

bool SplitInferenceEngine::isValidSplitPoint(
    const std::string& modelPath,
    const std::string& inputName,
    const std::string& splitOutput) {
#ifdef HAS_MNN
    auto* net = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (net == nullptr) return false;

    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    config.numThread = 1;
    config.saveTensors = {splitOutput};

    auto* session = net->createSession(config);
    if (session == nullptr) { delete net; return false; }

    auto* tensor = net->getSessionOutput(session, splitOutput.c_str());
    bool valid = (tensor != nullptr);

    if (!valid) {
        std::cerr << "[SplitInference] MNN output map missing: " << splitOutput << std::endl;
    }

    net->releaseSession(session);
    delete net;
    return valid;
#else
    (void)modelPath; (void)inputName; (void)splitOutput;
    return false;
#endif
}
// ============================================================
//  loadModel -- prepare prefix execution with the Interpreter API
// ============================================================

bool SplitInferenceEngine::loadModel(const std::string& modelPath,
                                     const std::string& inputName,
                                     const std::string& splitOutput,
                                     int splitLayer,
                                     int numThread) {
#ifdef HAS_MNN
    unload();

    modelPath_ = modelPath;
    inputName_ = inputName;
    splitOutput_ = splitOutput;
    splitLayer_ = splitLayer;

    fullMode_ = (splitLayer < 0);
    cutSetPlan_ = CutSetPlan();
    opNameToIndex_.clear();
    if (!fullMode_) {
        std::string cutError;
        if (!CutSetPlanner::buildFromModel(modelPath, splitLayer, cutSetPlan_, cutError)) {
            std::cerr << "[SplitInference] cut-set planning failed: " << cutError << std::endl;
            return false;
        }
        auto id2name = buildLayerId2Name(modelPath);
        for (const auto& item : id2name) opNameToIndex_[item.second] = item.first;
    }

    auto* net = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (net == nullptr) {
        std::cerr << "[SplitInference] loadModel: createFromFile failed" << std::endl;
        return false;
    }
    interpreter_.reset(net);

    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    config.numThread = numThread;
    // Preserve/materialize every graph-cut tensor. This scheduler-level
    // bypass Copy constraint keeps fused producers intact while making their
    // virtual boundary outputs safe to capture at the cloud-side consumer.
    if (!fullMode_) config.saveTensors = cutSetPlan_.tensorNames();
    // Do not rely on saveTensors/getSessionOutput because fused outputs may fail.
    // Use bypass capture callbacks to materialize fused and unfused boundary tensors.

    session_ = interpreter_->createSession(config);
    if (session_ == nullptr) {
        std::cerr << "[SplitInference] loadModel: createSession failed" << std::endl;
        interpreter_.reset();
        return false;
    }

    loaded_ = true;
    std::cout << "[SplitInference] OK split=" << splitLayer_
              << " output=" << splitOutput
              << " cutTensors=" << cutSetPlan_.tensors.size()
              << (fullMode_ ? " [FULL]" : "") << std::endl;
    return true;
#else
    (void)modelPath; (void)inputName; (void)splitOutput; (void)splitLayer;
    return false;
#endif
}

// ============================================================
//  runForward -- execute prefix inference with the Interpreter API
// ============================================================

bool SplitInferenceEngine::runForward(const std::vector<float>& inputData,
                                      const std::vector<int>& inputShape,
                                      std::vector<uint8_t>& outputBytes,
                                      double& edgeTimeMs) {
    EdgeFrameworkProfile profile;
    const bool ok = runForwardProfile(inputData, inputShape, outputBytes, profile, true);
    edgeTimeMs = profile.sessionMs;
    return ok;
}

bool SplitInferenceEngine::runForwardProfile(const std::vector<float>& inputData,
                                             const std::vector<int>& inputShape,
                                             std::vector<uint8_t>& outputBytes,
                                             EdgeFrameworkProfile& profile,
                                             bool capturePayload) {
#ifdef HAS_MNN
    profile = EdgeFrameworkProfile();
    const auto totalStart = std::chrono::high_resolution_clock::now();
    if (!loaded_ || interpreter_ == nullptr || session_ == nullptr) {
        std::cerr << "[SplitInference] not loaded" << std::endl;
        return false;
    }

    // Input tensor
    auto* inTensor = interpreter_->getSessionInput(session_, inputName_.c_str());
    if (inTensor == nullptr) {
        inTensor = interpreter_->getSessionInput(session_, nullptr);
    }
    if (inTensor == nullptr) {
        std::cerr << "[SplitInference] no input tensor" << std::endl;
        return false;
    }

    std::vector<int> dims = {inputShape[0], inputShape[1], inputShape[2], inputShape[3]};
    interpreter_->resizeTensor(inTensor, dims);
    interpreter_->resizeSession(session_);

    // External inputs have canonical NCHW semantics. Let MNN convert to the
    // backend's private layout (often NC4HW4); getDimensionType() does not
    // reliably expose that private storage format.
    std::unique_ptr<MNN::Tensor> hostInput(
        MNN::Tensor::create<float>(dims, const_cast<float*>(inputData.data()), MNN::Tensor::CAFFE));
    if (!hostInput || !inTensor->copyFromHostTensor(hostInput.get())) {
        std::cerr << "[SplitInference] input layout conversion failed" << std::endl;
        return false;
    }
    inputElements_ = (int)inputData.size();

    // Local-only mode executes the full graph without capture.
    if (fullMode_) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ret = interpreter_->runSession(session_);
        auto t1 = std::chrono::high_resolution_clock::now();
        profile.sessionMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ret != MNN::NO_ERROR) {
            std::cerr << "[SplitInference] runSession error " << ret << std::endl;
            return false;
        }
        outputBytes.clear();
        profile.totalMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - totalStart).count();
        return true;
    }

    // Capture each graph-cut tensor at the exact cloud-side consumer input.
    // This observes the post-fusion/post-Raster semantics actually consumed by
    // the suffix; a fused producer callback output may not equal a graph tensor.
    capturedCutTensors_.clear();
    std::set<int> observedCutTensors;
    auto capture = [&](const CutTensorDescriptor& descriptor, MNN::Tensor* source) {
        if (source == nullptr || observedCutTensors.count(descriptor.tensorIndex) != 0) return;
        observedCutTensors.insert(descriptor.tensorIndex);
        if (!capturePayload) return;
        // The wire representation is canonical NCHW. MNN's public
        // getDimensionType() reports CAFFE even for internal NC4HW4 tensors,
        // so preserving the device layout would silently mislabel bytes.
        std::shared_ptr<MNN::Tensor> host(
            MNN::Tensor::create(source->shape(), source->getType(), nullptr, MNN::Tensor::CAFFE));
        if (host != nullptr && source->copyToHostTensor(host.get())) {
            capturedCutTensors_[descriptor.tensorIndex] = std::move(host);
        }
    };

    auto before = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        if (info == nullptr) return true;
        auto it = opNameToIndex_.find(info->originalName());
        const bool isCloudConsumer = it != opNameToIndex_.end() && it->second > splitLayer_;
        if (isCloudConsumer) {
            const auto callbackStart = std::chrono::high_resolution_clock::now();
            for (const auto& descriptor : cutSetPlan_.tensors) {
                auto* target = interpreter_->getSessionTensor(session_, descriptor.tensorIndex);
                for (auto* tensor : tensors) {
                    if (tensor == target) capture(descriptor, tensor);
                }
            }
            profile.captureCallbackMs += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - callbackStart).count();
        }
        return it == opNameToIndex_.end() || it->second <= splitLayer_;
    };
    auto after = [&](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
        (void)tensors;
        (void)info;
        return observedCutTensors.size() < cutSetPlan_.tensors.size();
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ret = interpreter_->runSessionWithCallBackInfo(session_, before, after);
    auto t1 = std::chrono::high_resolution_clock::now();
    profile.sessionMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (ret != MNN::NO_ERROR && ret != MNN::CALL_BACK_STOP) {
        std::cerr << "[SplitInference] runSessionWithCallBackInfo error " << ret << std::endl;
        return false;
    }

    profile.cutTensors = static_cast<int>(observedCutTensors.size());
    if (!capturePayload) {
        outputBytes.clear();
        profile.totalMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - totalStart).count();
        if (observedCutTensors.size() != cutSetPlan_.tensors.size()) {
            std::cerr << "[SplitInference] P0 did not observe complete cut-set" << std::endl;
            return false;
        }
        return true;
    }

    std::vector<TensorPacket> packets;
    packets.reserve(cutSetPlan_.tensors.size());
    for (const auto& descriptor : cutSetPlan_.tensors) {
        auto it = capturedCutTensors_.find(descriptor.tensorIndex);
        if (it == capturedCutTensors_.end()) {
            std::cerr << "[SplitInference] incomplete cut-set, missing tensor "
                      << descriptor.tensorIndex << " " << descriptor.tensorName << std::endl;
            return false;
        }
        TensorPacket packet;
        if (!TensorSerializer::tensorToPacket(it->second.get(), descriptor.tensorIndex,
                                              descriptor.tensorName, packet)) {
            std::cerr << "[SplitInference] cannot serialize cut tensor " << descriptor.tensorName << std::endl;
            return false;
        }
        packets.push_back(std::move(packet));
    }
    CutBundleMetadata metadata;
    metadata.splitAfterOp = cutSetPlan_.splitAfterOp;
    metadata.modelFingerprint = cutSetPlan_.modelFingerprint;

    if (!codecSpec_.enabled) {
        const auto serializeStart = std::chrono::high_resolution_clock::now();
        outputBytes = TensorSerializer::serializeBundle(packets, &metadata);
        profile.serializeMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - serializeStart).count();
        profile.codec.enabled = false;
        profile.codec.requested = CodecKind::RAW32;
        profile.codec.tensorCount = static_cast<uint32_t>(packets.size());
        for (const auto& packet : packets) profile.codec.rawTensorBytes += packet.data.size();
        profile.codec.encodedTensorBytes = profile.codec.rawTensorBytes;
    } else {
        std::vector<EncodedTensorPacket> encoded;
        std::string codecError;
        if (!ActivationCodec::encodeBundle(packets, codecSpec_, encoded, profile.codec, &codecError)) {
            std::cerr << "[SplitInference] codec encode failed: " << codecError << std::endl;
            return false;
        }
        const auto serializeStart = std::chrono::high_resolution_clock::now();
        outputBytes = TensorSerializer::serializeEncodedBundle(encoded, metadata);
        profile.serializeMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - serializeStart).count();
    }
    if (outputBytes.empty()) {
        std::cerr << "[SplitInference] bundle serialization failed" << std::endl;
        return false;
    }
    profile.bundleBytes = outputBytes.size();
    profile.codec.bundleBytes = outputBytes.size();
    profile.totalMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - totalStart).count();
    return true;
#else
    (void)inputData; (void)inputShape; (void)outputBytes; (void)profile; (void)capturePayload;
    return false;
#endif
}

// ============================================================
//  unload -- release the model
// ============================================================

void SplitInferenceEngine::unload() {
#ifdef HAS_MNN
    if (interpreter_ != nullptr && session_ != nullptr) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    interpreter_.reset();
    capturedCutTensors_.clear();
#endif
    loaded_ = false;
    inputElements_ = 0;
    cutSetPlan_ = CutSetPlan();
    opNameToIndex_.clear();
}
