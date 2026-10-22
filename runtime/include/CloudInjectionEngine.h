#ifndef EDGE_AI_CLOUD_INJECTION_ENGINE_H
#define EDGE_AI_CLOUD_INJECTION_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <map>

#include "CutSetPlanner.h"
#include "CodecTypes.h"

struct CloudFrameworkProfile {
    double deserializeGuardMs{0.0};
    double sessionMs{0.0};
    double injectionCallbackMs{0.0};
    double outputCopyMs{0.0};
    double totalMs{0.0};
    uint32_t injectionCopyCount{0};
    CodecProfile codec;
};

#ifdef HAS_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#endif

/**
 * Cloud injection engine.
 *
 * Unlike CloudInferenceEngine (which loads a Module subgraph), Module::load
 * only accepts runtime-visible, unfused tensors as inputs. This engine loads
 * the complete model and injects edge tensors at suffix consumer inputs via
 * the before callback of runSessionWithCallBackInfo. This supports semantic
 * split boundaries even when their producer operators are fused.
 *
 * Data path: edge capture -> serialization -> deserialization and input-side
 * injection -> cloud suffix -> logits.
 */
class CloudInjectionEngine {
public:
    CloudInjectionEngine();
    ~CloudInjectionEngine();

    /**
     * Load a full-model session.
     * @param modelPath  model path, matching the edge model
     * @param splitInput T_k tensor name (split output and injection target)
     * @param outputName final output tensor name, for example logits
     */
    bool loadModel(const std::string& modelPath,
                   const CutSetPlan& cutSetPlan,
                   const std::string& outputName,
                   int backendType = 0,
                   int numThread = 4);
    bool loadModelLegacy(const std::string& modelPath,
                         const std::vector<std::string>& splitInputs,
                         const std::string& outputName,
                         int backendType = 0,
                         int numThread = 4);

    /**
     * Run suffix inference from T_L using a full graph and input-side injection.
     * @param tLBytes serialized T_L bytes received from the edge
     * @param outputData output FP32 logits
     * @param outputShape output tensor shape
     * @param cloudTimeMs output cloud inference latency in milliseconds
     */
    bool runBackHalf(const std::vector<uint8_t>& tLBytes,
                     std::vector<float>& outputData,
                     std::vector<int>& outputShape,
                     double& cloudTimeMs);

    bool runBackHalfProfile(const std::vector<uint8_t>& tLBytes,
                            std::vector<float>& outputData,
                            std::vector<int>& outputShape,
                            CloudFrameworkProfile& profile);

    void unload();
    bool isLoaded() const { return loaded_; }

private:
    struct InjectionBinding {
        std::string tensorName;
        std::string consumerOpName;
        int inputSlot{-1};
    };
#ifdef HAS_MNN
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
#endif
    std::string modelPath_;
    std::vector<std::string> splitInputs_;
    std::map<std::string, int> cutTensorIndexes_;
    std::vector<InjectionBinding> injectionBindings_;
    std::map<std::string, int> opNameToIndex_;
    std::string outputName_;
    int splitAfterOp_{-1};
    uint64_t modelFingerprint_{0};
    bool loaded_{false};
};

#endif // EDGE_AI_CLOUD_INJECTION_ENGINE_H
