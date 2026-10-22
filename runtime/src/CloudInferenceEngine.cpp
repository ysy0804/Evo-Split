#include "CloudInferenceEngine.h"
#include "TensorSerializer.h"

#include <iostream>
#include <chrono>
#include <cstring>

#ifdef HAS_MNN
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#endif

CloudInferenceEngine::CloudInferenceEngine() = default;
CloudInferenceEngine::~CloudInferenceEngine() { unload(); }

// ============================================================
//  loadModel — load the suffix inference module
// ============================================================

bool CloudInferenceEngine::loadModel(const std::string& modelPath,
                                     const std::string& splitInput,
                                     const std::string& outputName) {
#ifdef HAS_MNN
    unload();

    modelPath_ = modelPath;
    splitInput_ = splitInput;
    outputName_ = outputName;

    // Use the CPU backend with T_L as input and the final tensor as output.
    MNN::Express::Module::Config config;
    config.shapeMutable = true;
    config.backend = nullptr;

    auto* rawModule = MNN::Express::Module::load(
        {splitInput}, {outputName}, modelPath.c_str(), &config);

    if (rawModule == nullptr) {
        std::cerr << "[CloudInference] ❌ failed to load the suffix module: "
                  << modelPath << " (input=" << splitInput
                  << " output=" << outputName << ")" << std::endl;
        return false;
    }

    module_.reset(rawModule);
    loaded_ = true;
    std::cout << "[CloudInference] ✅ cloud module loaded: input=" << splitInput_
              << " output=" << outputName_ << std::endl;
    return true;
#else
    std::cerr << "[CloudInference] ⚠️ MNN support is not enabled" << std::endl;
    return false;
#endif
}

// ============================================================
//  runBackHalf -- execute the suffix from T_L
// ============================================================

bool CloudInferenceEngine::runBackHalf(const std::vector<uint8_t>& tLBytes,
                                       std::vector<float>& outputData,
                                       std::vector<int>& outputShape,
                                       double& cloudTimeMs) {
#ifdef HAS_MNN
    if (!loaded_ || module_ == nullptr) {
        std::cerr << "[CloudInference] ❌ module is not loaded" << std::endl;
        return false;
    }

    // Deserialize T_L.
    std::vector<int> tLShape;
    size_t tLElements = 0;
    if (!TensorSerializer::parseHeader(tLBytes, tLShape, tLElements)) {
        std::cerr << "[CloudInference] ❌ failed to parse the T_L header" << std::endl;
        return false;
    }

    if (tLShape.empty()) {
        std::cerr << "[CloudInference] ❌ T_L shape is empty" << std::endl;
        return false;
    }

    // Create the T_L input VARP using the serialized NC4HW4 layout.
    auto input = MNN::Express::_Input(tLShape, MNN::Express::NC4HW4);
    auto* inputPtr = input->writeMap<float>();
    if (inputPtr == nullptr) {
        std::cerr << "[CloudInference] ❌ cannot access the T_L buffer" << std::endl;
        return false;
    }

    // Copy payload data after the fixed header and shape metadata.
    size_t headerBytes = 4 + 4 + 4 * tLShape.size() + 4 + 4;
    if (headerBytes + tLElements * sizeof(float) > tLBytes.size()) {
        std::cerr << "[CloudInference] ❌ T_L payload is too short" << std::endl;
        return false;
    }
    std::memcpy(inputPtr, tLBytes.data() + headerBytes,
                tLElements * sizeof(float));

    // Measure execution time.
    auto start = std::chrono::high_resolution_clock::now();

    // Run suffix inference.
    auto outputs = module_->onForward({input});
    if (outputs.empty() || outputs[0] == nullptr) {
        std::cerr << "[CloudInference] ❌ suffix inference failed" << std::endl;
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    cloudTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Read the final output.
    auto outputVar = outputs[0];
    auto* outputPtr = outputVar->readMap<float>();
    if (outputPtr == nullptr) {
        std::cerr << "[CloudInference] ❌ cannot read the final output" << std::endl;
        return false;
    }

    auto info = outputVar->getInfo();
    outputShape = info ? info->dim : std::vector<int>{1};
    int outElements = 1;
    for (int d : outputShape) outElements *= d;
    if (outElements <= 0) outElements = 1;

    outputData.resize(outElements);
    std::memcpy(outputData.data(), outputPtr, outElements * sizeof(float));

    return true;
#else
    return false;
#endif
}

// ============================================================
//  unload
// ============================================================

void CloudInferenceEngine::unload() {
#ifdef HAS_MNN
    if (module_) {
        module_.reset();
    }
#endif
    loaded_ = false;
}
