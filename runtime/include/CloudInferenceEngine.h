#ifndef EDGE_AI_CLOUD_INFERENCE_ENGINE_H
#define EDGE_AI_CLOUD_INFERENCE_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#ifdef HAS_MNN
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#endif

/**
 * Cloud suffix inference engine.
 *
 * Receives the boundary tensor T_L and continues suffix execution
 * to produce the final logits.
 *
 * Pairs with SplitInferenceEngine to form a complete split pipeline:
 *   edge: input -> ... -> T_L (SplitInferenceEngine)
 *   transport: serialized T_L bytes
 *   cloud: T_L -> ... -> logits (CloudInferenceEngine)
 */
class CloudInferenceEngine {
public:
    CloudInferenceEngine();
    ~CloudInferenceEngine();

    /**
     * Load the model suffix module.
     * @param modelPath  model path, matching the edge model
     * @param splitInput T_L tensor name used as the suffix input
     * @param outputName final output tensor name, for example logits
     * @return true on success
     */
    bool loadModel(const std::string& modelPath,
                   const std::string& splitInput,
                   const std::string& outputName);

    /**
     * Run suffix inference from T_L.
     * @param tLBytes   serialized T_L bytes from the edge
     * @param outputData output FP32 logits
     * @param outputShape output tensor shape
     * @param cloudTimeMs output cloud inference latency in milliseconds
     * @return true on success
     */
    bool runBackHalf(const std::vector<uint8_t>& tLBytes,
                     std::vector<float>& outputData,
                     std::vector<int>& outputShape,
                     double& cloudTimeMs);

    /**
     * Unload the model.
     */
    void unload();

    bool isLoaded() const { return loaded_; }

private:
#ifdef HAS_MNN
    std::unique_ptr<MNN::Express::Module> module_;
#endif
    bool loaded_{false};
    std::string modelPath_;
    std::string splitInput_;
    std::string outputName_;
};

#endif // EDGE_AI_CLOUD_INFERENCE_ENGINE_H
