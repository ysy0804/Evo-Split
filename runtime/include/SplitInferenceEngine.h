#ifndef EDGE_AI_SPLIT_INFERENCE_ENGINE_H
#define EDGE_AI_SPLIT_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <map>

#include "CutSetPlanner.h"
#include "CodecTypes.h"

struct EdgeFrameworkProfile {
    double sessionMs{0.0};
    double captureCallbackMs{0.0};
    double serializeMs{0.0};
    double totalMs{0.0};
    std::size_t bundleBytes{0};
    int cutTensors{0};
    CodecProfile codec;
};

#ifdef HAS_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#endif

/**
 * Edge split inference engine based on the Interpreter API.
 *
 * The Interpreter API provides the following runtime controls:
 * 1. ScheduleConfig.saveTensors can retain selected intermediate tensors
 * 2. getSessionOutput safely reports unavailable boundary tensors
 * 3. invalid candidates fail gracefully instead of being dereferenced
 */
class SplitInferenceEngine {
public:
    SplitInferenceEngine();
    ~SplitInferenceEngine();

    /**
     * Load a model and prepare prefix inference.
     * @param modelPath  model path
     * @param inputName  model input tensor name
     * @param splitOutput boundary output tensor name
     * @param splitLayer  physical boundary index used for telemetry
     * @return true on success
     */
    bool loadModel(const std::string& modelPath,
                   const std::string& inputName,
                   const std::string& splitOutput,
                   int splitLayer,
                   int numThread = 4);

    /**
     * Run prefix inference.
     * @param inputData   FP32 input data
     * @param inputShape  input tensor shape
     * @param outputBytes output serialized boundary bundle
     * @param edgeTimeMs  output edge latency in milliseconds
     * @return true on success
     */
    bool runForward(const std::vector<float>& inputData,
                    const std::vector<int>& inputShape,
                    std::vector<uint8_t>& outputBytes,
                    double& edgeTimeMs);

    /**
     * B1 microbenchmark entry. When capturePayload is false the same selective
     * schedule observes the complete cut-set but performs no host copy or
     * serialization (paired P0 baseline).
     */
    bool runForwardProfile(const std::vector<float>& inputData,
                           const std::vector<int>& inputShape,
                           std::vector<uint8_t>& outputBytes,
                           EdgeFrameworkProfile& profile,
                           bool capturePayload = true);

    /**
     * Return the model input element count.
     */
    int getInputElements() const { return inputElements_; }

    /**
     * Preflight a split candidate against the optimized MNN graph.
     * Some early MobileNetV2 operators may be fused and unavailable as session outputs.
     * Historically visible examples include Clip, Add, reduce-conv, and later convolution outputs.
     */
    static bool isValidSplitPoint(const std::string& modelPath,
                                  const std::string& inputName,
                                  const std::string& splitOutput);

    /**
     * Return the current physical boundary.
     */
    int getSplitLayer() const { return splitLayer_; }

    const CutSetPlan& getCutSetPlan() const { return cutSetPlan_; }

    void setCodecSpec(const CodecSpec& spec) { codecSpec_ = spec; }
    const CodecSpec& getCodecSpec() const { return codecSpec_; }

    /**
     * Unload the model.
     */
    void unload();

    bool isLoaded() const { return loaded_; }

private:
#ifdef HAS_MNN
    std::shared_ptr<MNN::Interpreter> interpreter_;
    MNN::Session* session_ = nullptr;
    std::map<int, std::shared_ptr<MNN::Tensor>> capturedCutTensors_;
#endif
    bool loaded_{false};
    bool fullMode_{false};      // splitLayer<0: full local inference without capture or skipping
    std::string modelPath_;
    std::string inputName_;
    std::string splitOutput_;
    int splitLayer_{0};
    int inputElements_{0};
    CutSetPlan cutSetPlan_;
    std::map<std::string, int> opNameToIndex_;
    CodecSpec codecSpec_;
};

#endif // EDGE_AI_SPLIT_INFERENCE_ENGINE_H
