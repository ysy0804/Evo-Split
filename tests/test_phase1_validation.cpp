#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CloudInjectionEngine.h"
#include "CutSetPlanner.h"
#include "SplitInferenceEngine.h"
#include "TensorSerializer.h"

#ifdef HAS_MNN
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN_generated.h>
#include <flatbuffers/flatbuffers.h>
#endif

#ifdef HAS_MNN
namespace {

int printCutCatalog(const std::string& modelPath) {
    std::ifstream input(modelPath, std::ios::binary);
    if (!input) return 1;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (bytes.empty() || !MNN::VerifyNetBuffer(verifier)) return 2;
    const auto* net = MNN::GetNet(bytes.data());
    if (!net || !net->oplists()) return 3;
    std::vector<std::string> outputs;
    if (net->outputName()) {
        for (int i = 0; i < static_cast<int>(net->outputName()->size()); ++i) {
            std::string output = net->outputName()->Get(i)->str();
            std::replace(output.begin(), output.end(), '\t', ' ');
            outputs.push_back(output);
        }
    }
    int earliestOutputProducer = static_cast<int>(net->oplists()->size());
    if (net->tensorName()) {
        for (const auto& output : outputs) {
            int tensorIndex = -1;
            for (int t = 0; t < static_cast<int>(net->tensorName()->size()); ++t) {
                if (net->tensorName()->Get(t)->str() == output) tensorIndex = t;
            }
            for (int opIndex = 0; tensorIndex >= 0 && opIndex < static_cast<int>(net->oplists()->size()); ++opIndex) {
                const auto* op = net->oplists()->Get(opIndex);
                if (!op || !op->outputIndexes()) continue;
                for (int slot = 0; slot < static_cast<int>(op->outputIndexes()->size()); ++slot) {
                    if (op->outputIndexes()->Get(slot) == tensorIndex) {
                        earliestOutputProducer = std::min(earliestOutputProducer, opIndex);
                    }
                }
            }
        }
    }
    std::cout << "MODEL_INFO\t" << net->oplists()->size() << '\t';
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << outputs[i];
    }
    std::cout << '\t' << earliestOutputProducer << '\n';
    std::cout << "CATALOG_HEADER\tsplit_index\top_name\top_type\tvalid\tcut_tensors\tbindings\terror\n";
    const int semanticLimit = std::min(
        static_cast<int>(net->oplists()->size()) - 1, earliestOutputProducer);
    for (int i = 0; i < semanticLimit; ++i) {
        const auto* op = net->oplists()->Get(i);
        std::string name = op && op->name() ? op->name()->str() : "op_" + std::to_string(i);
        std::replace(name.begin(), name.end(), '\t', ' ');
        std::string type = op ? MNN::EnumNameOpType(op->type()) : "Unknown";
        CutSetPlan plan;
        std::string error;
        const bool valid = CutSetPlanner::buildFromModel(modelPath, i, plan, error);
        size_t bindings = 0;
        for (const auto& tensor : plan.tensors) bindings += tensor.consumers.size();
        std::replace(error.begin(), error.end(), '\t', ' ');
        std::cout << "CATALOG\t" << i << '\t' << name << '\t' << type << '\t'
                  << (valid ? 1 : 0) << '\t' << plan.tensors.size() << '\t'
                  << bindings << '\t' << error << '\n';
    }
    return 0;
}

bool runFullReference(const std::string& modelPath, const std::string& inputName,
                      const std::string& outputName, const std::vector<float>& inputData,
                      const std::vector<int>& inputShape, std::vector<float>& outputData) {
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!net) return false;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    auto* session = net->createSession(config);
    if (!session) return false;
    auto* input = net->getSessionInput(session, inputName.c_str());
    if (!input) input = net->getSessionInput(session, nullptr);
    if (!input) return false;
    net->resizeTensor(input, inputShape);
    net->resizeSession(session);
    std::unique_ptr<MNN::Tensor> hostInput(
        MNN::Tensor::create<float>(inputShape, const_cast<float*>(inputData.data()), MNN::Tensor::CAFFE));
    if (!hostInput || !input->copyFromHostTensor(hostInput.get()) ||
        net->runSession(session) != MNN::NO_ERROR) return false;
    auto* output = net->getSessionOutput(session, outputName.c_str());
    if (!output) output = net->getSessionOutput(session, nullptr);
    if (!output) return false;
    std::unique_ptr<MNN::Tensor> hostOutput(MNN::Tensor::createHostTensorFromDevice(output, true));
    if (!hostOutput || !hostOutput->host<float>()) return false;
    outputData.assign(hostOutput->host<float>(), hostOutput->host<float>() + hostOutput->elementSize());
    net->releaseSession(session);
    return true;
}

bool captureReferenceCut(const std::string& modelPath, const std::string& inputName,
                         const std::vector<float>& inputData, const std::vector<int>& inputShape,
                         const CutSetPlan& cutPlan, std::vector<TensorPacket>& packets) {
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(modelPath.c_str()));
    if (!net) return false;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    config.saveTensors = cutPlan.tensorNames();
    auto* session = net->createSession(config);
    if (!session) return false;
    auto* input = net->getSessionInput(session, inputName.c_str());
    if (!input) input = net->getSessionInput(session, nullptr);
    if (!input) return false;
    net->resizeTensor(input, inputShape);
    net->resizeSession(session);
    std::unique_ptr<MNN::Tensor> hostInput(
        MNN::Tensor::create<float>(inputShape, const_cast<float*>(inputData.data()), MNN::Tensor::CAFFE));
    if (!hostInput || !input->copyFromHostTensor(hostInput.get()) ||
        net->runSession(session) != MNN::NO_ERROR) return false;
    packets.clear();
    for (const auto& descriptor : cutPlan.tensors) {
        auto* tensor = net->getSessionOutput(session, descriptor.tensorName.c_str());
        if (!tensor) return false;
        std::unique_ptr<MNN::Tensor> host(
            MNN::Tensor::create(tensor->shape(), tensor->getType(), nullptr, MNN::Tensor::CAFFE));
        TensorPacket packet;
        if (!host || !tensor->copyToHostTensor(host.get()) ||
            !TensorSerializer::tensorToPacket(host.get(), descriptor.tensorIndex,
                                              descriptor.tensorName, packet)) return false;
        packets.push_back(std::move(packet));
    }
    net->releaseSession(session);
    return true;
}

bool runModuleTail(const std::string& modelPath, const std::vector<TensorPacket>& packets,
                   const std::string& outputName, std::vector<float>& outputData) {
    std::vector<std::string> names;
    for (const auto& packet : packets) names.push_back(packet.name);
    MNN::Express::Module::Config config;
    config.shapeMutable = true;
    std::unique_ptr<MNN::Express::Module> module(
        MNN::Express::Module::load(names, {outputName}, modelPath.c_str(), &config));
    if (!module) return false;
    std::vector<MNN::Express::VARP> inputs;
    for (const auto& packet : packets) {
        auto input = MNN::Express::_Input(packet.shape, MNN::Express::NCHW);
        auto* ptr = input->writeMap<float>();
        if (!ptr) return false;
        std::memcpy(ptr, packet.data.data(), packet.data.size());
        inputs.push_back(input);
    }
    auto outputs = module->onForward(inputs);
    if (outputs.empty() || outputs[0] == nullptr) return false;
    auto* ptr = outputs[0]->readMap<float>();
    auto* info = outputs[0]->getInfo();
    if (!ptr || !info) return false;
    size_t count = 1;
    for (int dim : info->dim) count *= dim;
    outputData.assign(ptr, ptr + count);
    return true;
}

void printNumbers(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

}  // namespace
#endif

int main(int argc, char* argv[]) {
#ifdef HAS_MNN
    if (argc == 3 && std::string(argv[1]) == "--catalog") return printCutCatalog(argv[2]);
    if (argc >= 5 && std::string(argv[1]) == "--consume") {
        const std::string modelPath = argv[2];
        const int splitIndex = std::stoi(argv[3]);
        const std::string bundlePath = argv[4];
        const std::string outputName = argc > 5 ? argv[5] : "logits";
        CutSetPlan plan;
        std::string error;
        if (!CutSetPlanner::buildFromModel(modelPath, splitIndex, plan, error)) return 30;
        CloudInjectionEngine cloud;
        if (!cloud.loadModel(modelPath, plan, outputName)) return 31;
        std::ifstream input(bundlePath, std::ios::binary);
        std::vector<uint8_t> bundle((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
        std::vector<float> output;
        std::vector<int> shape;
        double elapsed = 0.0;
        const bool accepted = cloud.runBackHalf(bundle, output, shape, elapsed);
        std::cout << "CONSUME_JSON {\"accepted\":" << (accepted ? "true" : "false")
                  << ",\"elapsedMs\":" << elapsed << "}" << std::endl;
        return accepted ? 0 : 32;
    }
#endif
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mnn> <input_name> <split_output_name> <split_index> [shape...]"
                  << " [--output name] [--warmup N] [--reps N] [--seed N]"
                  << " [--skip-module] [--save path]" << std::endl;
        return 1;
    }
#ifndef HAS_MNN
    return 1;
#else
    const std::string modelPath = argv[1];
    const std::string inputName = argv[2];
    const std::string splitOutput = argv[3];
    const int splitIndex = std::stoi(argv[4]);
    std::vector<int> inputShape;
    std::string outputName = "logits";
    std::string savePath;
    int warmup = 2;
    int reps = 3;
    int seed = 0;
    bool skipModule = false;
    for (int i = 5; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) outputName = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::max(0, std::stoi(argv[++i]));
        else if (arg == "--reps" && i + 1 < argc) reps = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoi(argv[++i]);
        else if (arg == "--save" && i + 1 < argc) savePath = argv[++i];
        else if (arg == "--skip-module") skipModule = true;
        else if (arg.rfind("--", 0) == 0) return 2;
        else inputShape.push_back(std::stoi(arg));
    }
    if (inputShape.empty()) inputShape = {1, 3, 224, 224};
    if (inputShape.size() != 4) return 3;
    int inputElements = 1;
    for (int d : inputShape) inputElements *= d;
    std::vector<float> inputData(inputElements);
    for (int i = 0; i < inputElements; ++i) inputData[i] = 0.01f * ((i + seed * 7) % 101) - 0.5f;

    SplitInferenceEngine engine;
    if (!engine.loadModel(modelPath, inputName, splitOutput, splitIndex)) return 10;
    std::vector<uint8_t> lastOutputBytes;
    std::vector<TensorPacket> lastCutPackets;
    std::vector<double> edgeSamples;
    std::vector<double> edgePipelineSamples;
    for (int run = -warmup; run < reps; ++run) {
        std::vector<uint8_t> bytes;
        double edgeMs = 0.0;
        const auto begin = std::chrono::high_resolution_clock::now();
        if (!engine.runForward(inputData, inputShape, bytes, edgeMs)) return 11;
        const auto end = std::chrono::high_resolution_clock::now();
        std::vector<TensorPacket> packets;
        std::string error;
        if (!TensorSerializer::deserializeBundle(bytes, packets, &error)) return 12;
        if (run >= 0) {
            edgeSamples.push_back(edgeMs);
            edgePipelineSamples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        }
        lastOutputBytes = std::move(bytes);
        lastCutPackets = std::move(packets);
    }

    CloudInjectionEngine cloudEngine;
    if (!cloudEngine.loadModel(modelPath, engine.getCutSetPlan(), outputName)) return 13;
    std::vector<float> lastCloudOutput;
    std::vector<double> cloudSamples;
    for (int run = -warmup; run < reps; ++run) {
        std::vector<float> output;
        std::vector<int> shape;
        double cloudMs = 0.0;
        if (!cloudEngine.runBackHalf(lastOutputBytes, output, shape, cloudMs)) return 14;
        if (run >= 0) cloudSamples.push_back(cloudMs);
        lastCloudOutput = std::move(output);
    }

    bool fingerprintGuard = false;
    if (lastOutputBytes.size() > 16) {
        auto stale = lastOutputBytes;
        stale[16] ^= 0x01;
        std::vector<float> output;
        std::vector<int> shape;
        double elapsed = 0.0;
        fingerprintGuard = !cloudEngine.runBackHalf(stale, output, shape, elapsed);
    }
    if (!fingerprintGuard) return 15;

    std::vector<TensorPacket> referenceCutPackets;
    if (!captureReferenceCut(modelPath, inputName, inputData, inputShape,
                             engine.getCutSetPlan(), referenceCutPackets) ||
        referenceCutPackets.size() != lastCutPackets.size()) return 16;
    double cutMaxAbsError = 0.0;
    for (size_t p = 0; p < lastCutPackets.size(); ++p) {
        if (lastCutPackets[p].data.size() != referenceCutPackets[p].data.size()) return 17;
        const float* actual = reinterpret_cast<const float*>(lastCutPackets[p].data.data());
        const float* reference = reinterpret_cast<const float*>(referenceCutPackets[p].data.data());
        for (size_t i = 0; i < lastCutPackets[p].data.size() / sizeof(float); ++i) {
            const double error = std::fabs(static_cast<double>(actual[i]) - reference[i]);
            if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) || !std::isfinite(error)) {
                cutMaxAbsError = 1e300;
            } else {
                cutMaxAbsError = std::max(cutMaxAbsError, error);
            }
        }
    }

    std::vector<float> referenceOutput;
    if (!runFullReference(modelPath, inputName, outputName, inputData, inputShape, referenceOutput) ||
        referenceOutput.size() != lastCloudOutput.size()) return 18;
    double maxAbsError = 0.0;
    double meanAbsError = 0.0;
    for (size_t i = 0; i < referenceOutput.size(); ++i) {
        const double error = std::fabs(static_cast<double>(referenceOutput[i]) - lastCloudOutput[i]);
        if (!std::isfinite(referenceOutput[i]) || !std::isfinite(lastCloudOutput[i]) || !std::isfinite(error)) {
            maxAbsError = 1e300;
            meanAbsError = 1e300;
        } else if (meanAbsError < 1e299) {
            maxAbsError = std::max(maxAbsError, error);
            meanAbsError += error;
        }
    }
    if (!referenceOutput.empty() && meanAbsError < 1e299) meanAbsError /= referenceOutput.size();

    bool moduleAvailable = false;
    double moduleMaxError = -1.0;
    if (!skipModule) {
        std::vector<float> moduleOutput;
        moduleAvailable = runModuleTail(modelPath, referenceCutPackets, outputName, moduleOutput) &&
                          moduleOutput.size() == referenceOutput.size();
        if (moduleAvailable) {
            moduleMaxError = 0.0;
            for (size_t i = 0; i < moduleOutput.size(); ++i) {
                moduleMaxError = std::max(moduleMaxError,
                    std::fabs(static_cast<double>(moduleOutput[i]) - referenceOutput[i]));
            }
        }
    }
    size_t bindings = 0;
    for (const auto& cut : engine.getCutSetPlan().tensors) bindings += cut.consumers.size();
    if (!savePath.empty()) {
        FILE* file = fopen(savePath.c_str(), "wb");
        if (file) {
            fwrite(lastOutputBytes.data(), 1, lastOutputBytes.size(), file);
            fclose(file);
        }
    }
    std::ostringstream json;
    json << "{\"splitIndex\":" << splitIndex
         << ",\"cutTensors\":" << lastCutPackets.size()
         << ",\"bindings\":" << bindings
         << ",\"bundleBytes\":" << lastOutputBytes.size()
         << ",\"edgeSamplesMs\":";
    printNumbers(json, edgeSamples);
    json << ",\"edgePipelineSamplesMs\":";
    printNumbers(json, edgePipelineSamples);
    json << ",\"cloudSamplesMs\":";
    printNumbers(json, cloudSamples);
    json << ",\"cutMaxAbsError\":" << cutMaxAbsError
         << ",\"maxAbsError\":" << maxAbsError
         << ",\"meanAbsError\":" << meanAbsError
         << ",\"moduleAvailable\":" << (moduleAvailable ? "true" : "false")
         << ",\"moduleMaxAbsError\":" << moduleMaxError
         << ",\"fingerprintGuard\":" << (fingerprintGuard ? "true" : "false") << '}';
    std::cout << "PHASE1_JSON " << json.str() << std::endl;
    return maxAbsError <= 1e-4 ? 0 : 20;
#endif
}
