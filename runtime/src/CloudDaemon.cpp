#include "CloudDaemon.h"
#include "Win32Socket.h"
#include "CloudInjectionEngine.h"
#include "TensorSerializer.h"
#include "CutSetPlanner.h"

#include <iostream>
#include <cstring>
#include <chrono>

#ifdef HAS_MNN
#include <MNN/expr/Module.hpp>
#endif

CloudDaemon::CloudDaemon(int port)
    : port_(port), serverSocket_(-1) {}

CloudDaemon::~CloudDaemon() {
    stop();
    if (serverSocket_ >= 0) close(serverSocket_);
}

bool CloudDaemon::initServer() {
    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ < 0) {
        std::cerr << "[CloudDaemon] Failed to create socket" << std::endl;
        return false;
    }
    int opt = 1;
    setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);

    if (bind(serverSocket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[CloudDaemon] Bind failed port " << port_ << std::endl;
        return false;
    }
    if (listen(serverSocket_, 5) < 0) {
        std::cerr << "[CloudDaemon] Listen failed" << std::endl;
        return false;
    }
    return true;
}

void CloudDaemon::start() {
    std::cout << "[CloudDaemon] Cloud service started port=" << port_ << std::endl;
    std::cout << "[CloudDaemon] model=" << modelPath_ << std::endl;

    if (!initServer()) return;

    // Preload the injection engine (full-model + before-callback inject)
    cloudEngine_ = std::make_unique<CloudInjectionEngine>();

    running_ = true;

    while (running_) {
        struct sockaddr_in clientAddr{};
        socklen_t addrLen = sizeof(clientAddr);
        int client = accept(serverSocket_, (struct sockaddr*)&clientAddr, &addrLen);
        if (client < 0) {
            if (running_) std::cerr << "[CloudDaemon] Accept failed" << std::endl;
            continue;
        }
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, ip, INET_ADDRSTRLEN);
        std::cout << "[CloudDaemon] Request from " << ip << std::endl;

        handleConnection(client);
        close(client);
    }
    std::cout << "[CloudDaemon] Shutdown" << std::endl;
}

void CloudDaemon::stop() {
    running_ = false;
    if (serverSocket_ >= 0) {
        shutdown(serverSocket_, SHUT_RDWR);
    }
}

static bool readAll(int fd, void* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t r = netRead(fd, (uint8_t*)buf + off, n - off);
        if (r <= 0) return false;
        off += r;
    }
    return true;
}

static bool writeAll(int fd, const void* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = netWrite(fd, (const uint8_t*)buf + off, n - off);
        if (w <= 0) return false;
        off += w;
    }
    return true;
}

void CloudDaemon::handleConnection(int clientSock) {
#ifdef HAS_MNN
    // 1) Read request header: [4B magic][4B totalLen][4B inputNameLen]
    uint32_t header[3];
    if (!readAll(clientSock, header, 12)) {
        std::cerr << "[CloudDaemon] header read fail" << std::endl; return;
    }

    const bool codecProfiled = header[0] == CloudProtocol::MAGIC_REQUEST_CODEC_PROFILE;
    const bool profiled = header[0] == CloudProtocol::MAGIC_REQUEST_PROFILE || codecProfiled;
    if (header[0] != CloudProtocol::MAGIC_REQUEST && !profiled) {
        std::cerr << "[CloudDaemon] bad magic" << std::endl;
        uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
        writeAll(clientSock, resp, 12);
        return;
    }
    uint32_t inputNameLen = header[2];
    uint32_t remaining = header[1] - 4;

    // 2) Read body: inputName + outputNameLen + outputName + T_L
    std::vector<uint8_t> rest(remaining);
    if (!readAll(clientSock, rest.data(), remaining)) {
        std::cerr << "[CloudDaemon] body read fail" << std::endl; return;
    }
    // Start server accounting only after the complete request body has arrived.
    // The client-side residual then contains connect + upload + response download.
    const auto serverStart = std::chrono::steady_clock::now();

    size_t off = 0;
    auto readU32 = [&]() -> uint32_t {
        uint32_t v; memcpy(&v, rest.data() + off, 4); off += 4; return v;
    };

    std::string splitInput((const char*)rest.data() + off, inputNameLen);
    off += inputNameLen;

    uint32_t outputNameLen = readU32();
    std::string outputName((const char*)rest.data() + off, outputNameLen);
    off += outputNameLen;

    std::vector<uint8_t> tLData(rest.begin() + off, rest.end());

    // MCUT bundles are self-describing. Keep accepting the legacy one-tensor
    // MNTL payload so cloud-only and old clients remain wire-compatible.
    std::vector<TensorPacket> cutPackets;
    CutBundleMetadata bundleMetadata;
    std::string bundleError;
    bool parsedCutBundle = false;
    uint32_t bundleVersion = 0;
    if (TensorSerializer::inspectBundleVersion(tLData, bundleVersion) &&
        bundleVersion == TensorSerializer::CODEC_BUNDLE_VERSION) {
        // Parse only metadata here. Actual decoding belongs to
        // CloudInjectionEngine and is timed exactly once as cloud codec tax.
        std::vector<EncodedTensorPacket> encodedPackets;
        parsedCutBundle = TensorSerializer::deserializeEncodedBundle(
            tLData, encodedPackets, &bundleError, &bundleMetadata);
        if (parsedCutBundle) {
            cutPackets.reserve(encodedPackets.size());
            for (const auto& encoded : encodedPackets) {
                TensorPacket packet;
                packet.tensorIndex = encoded.tensorIndex;
                packet.name = encoded.name;
                packet.shape = encoded.shape;
                packet.dataType = encoded.dataType;
                packet.layout = encoded.layout;
                cutPackets.push_back(std::move(packet));
            }
        }
    } else {
        parsedCutBundle = TensorSerializer::deserializeBundle(
            tLData, cutPackets, &bundleError, &bundleMetadata);
    }
    if (!parsedCutBundle) {
        std::vector<int> shape;
        size_t elements = 0;
        if (!TensorSerializer::parseHeader(tLData, shape, elements)) {
            std::cerr << "[CloudDaemon] invalid tensor payload: " << bundleError << std::endl;
            uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
            writeAll(clientSock, resp, 12);
            return;
        }
        const size_t headerBytes = 4 + 4 + 4 * shape.size() + 4 + 4;
        if (headerBytes > tLData.size()) {
            uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
            writeAll(clientSock, resp, 12);
            return;
        }
        TensorPacket packet;
        packet.name = splitInput;
        packet.shape = shape;
        packet.layout = 1;
        packet.data.assign(tLData.begin() + headerBytes, tLData.end());
        cutPackets.push_back(std::move(packet));
        tLData = TensorSerializer::serializeBundle(cutPackets);
    }
    std::vector<std::string> cutNames;
    std::string cutCacheKey;
    for (const auto& packet : cutPackets) {
        cutNames.push_back(packet.name);
        cutCacheKey += std::to_string(packet.tensorIndex) + ":" + packet.name + ";";
    }

    CutSetPlan verifiedPlan;
    if (bundleMetadata.valid()) {
        const std::string planKey = std::to_string(bundleMetadata.modelFingerprint) + ":" +
                                    std::to_string(bundleMetadata.splitAfterOp);
        auto cached = verifiedPlanCache_.find(planKey);
        if (cached != verifiedPlanCache_.end()) {
            verifiedPlan = cached->second;
        } else {
            std::string planError;
            if (!CutSetPlanner::buildFromModel(modelPath_, bundleMetadata.splitAfterOp,
                                               verifiedPlan, planError) ||
                verifiedPlan.modelFingerprint != bundleMetadata.modelFingerprint ||
                verifiedPlan.tensorNames() != cutNames) {
                std::cerr << "[CloudDaemon] stale/model-mismatched MCUT rejected: "
                          << planError << std::endl;
                cloudEngine_->unload();
                splitInputName_.clear();
                uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
                writeAll(clientSock, resp, 12);
                return;
            }
            verifiedPlanCache_[planKey] = verifiedPlan;
        }
        if (verifiedPlan.tensorNames() != cutNames) {
            uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
            writeAll(clientSock, resp, 12);
            return;
        }
        cutCacheKey = verifiedPlan.cacheKey() + ":" + outputName;
    } else {
        if (cachedModelFingerprint_ == 0) {
            std::string fingerprintError;
            if (!CutSetPlanner::fingerprintModel(modelPath_, cachedModelFingerprint_, fingerprintError)) {
                uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
                writeAll(clientSock, resp, 12);
                return;
            }
        }
        cutCacheKey = std::string("legacy:") + std::to_string(cachedModelFingerprint_) + ":" +
                      cutCacheKey + ":" + outputName;
    }

    // 3) Load a selective suffix session, cached by model version + complete cut-set.
    if (!cloudEngine_->isLoaded() || splitInputName_ != cutCacheKey) {
        cloudEngine_->unload();
        const bool loaded = bundleMetadata.valid()
            ? cloudEngine_->loadModel(modelPath_, verifiedPlan, outputName, backendType_, numThread_)
            : cloudEngine_->loadModelLegacy(modelPath_, cutNames, outputName, backendType_, numThread_);
        if (!loaded) {
            uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
            writeAll(clientSock, resp, 12);
            return;
        }
        splitInputName_ = cutCacheKey;
    }

    std::vector<float> outData;
    std::vector<int> outShape;
    double cloudTimeMs = 0;
    CloudFrameworkProfile cloudProfile;
    const bool inferenceOk = profiled
        ? cloudEngine_->runBackHalfProfile(tLData, outData, outShape, cloudProfile)
        : cloudEngine_->runBackHalf(tLData, outData, outShape, cloudTimeMs);
    if (!inferenceOk) {
        uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
        writeAll(clientSock, resp, 12);
        return;
    }

    // 4) Response. Legacy clients keep the original 12-byte header. B2 clients
    // receive a 28-byte profile header so network time can be separated from
    // server parsing/guard, injection, suffix compute, and output copying.
    uint32_t elements = (uint32_t)outData.size();
    if (profiled) {
        const double serverTotalMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - serverStart).count();
        auto bits = [](float value) {
            uint32_t result = 0;
            std::memcpy(&result, &value, sizeof(result));
            return result;
        };
        if (codecProfiled) {
            const uint64_t decodeStart = cloudProfile.codec.startSteadyNs;
            const uint64_t decodeEnd = cloudProfile.codec.endSteadyNs;
            uint32_t response[17] = {
                CloudProtocol::STATUS_OK, elements,
                bits(static_cast<float>(cloudProfile.sessionMs)),
                bits(static_cast<float>(cloudProfile.deserializeGuardMs)),
                bits(static_cast<float>(cloudProfile.injectionCallbackMs)),
                bits(static_cast<float>(cloudProfile.outputCopyMs)),
                bits(static_cast<float>(serverTotalMs)),
                bits(static_cast<float>(cloudProfile.codec.elapsedMs)),
                static_cast<uint32_t>(cloudProfile.codec.rawTensorBytes),
                static_cast<uint32_t>(cloudProfile.codec.encodedTensorBytes),
                static_cast<uint32_t>(cloudProfile.codec.bundleBytes),
                cloudProfile.codec.fallbackCount,
                static_cast<uint32_t>(cloudProfile.codec.requested) |
                    (cloudProfile.codec.enabled ? 0x80000000u : 0u),
                static_cast<uint32_t>(decodeStart), static_cast<uint32_t>(decodeStart >> 32),
                static_cast<uint32_t>(decodeEnd), static_cast<uint32_t>(decodeEnd >> 32),
            };
            writeAll(clientSock, response, sizeof(response));
        } else {
            uint32_t response[7] = {
                CloudProtocol::STATUS_OK, elements,
                bits(static_cast<float>(cloudProfile.sessionMs)),
                bits(static_cast<float>(cloudProfile.deserializeGuardMs)),
                bits(static_cast<float>(cloudProfile.injectionCallbackMs)),
                bits(static_cast<float>(cloudProfile.outputCopyMs)),
                bits(static_cast<float>(serverTotalMs)),
            };
            writeAll(clientSock, response, sizeof(response));
        }
        writeAll(clientSock, outData.data(), elements * sizeof(float));
        std::cout << "B2_CLOUD_PROFILE server=" << serverTotalMs
                  << " deserialize=" << cloudProfile.deserializeGuardMs
                  << " inject=" << cloudProfile.injectionCallbackMs
                  << " injectionCopies=" << cloudProfile.injectionCopyCount
                  << " suffix=" << (cloudProfile.sessionMs - cloudProfile.injectionCallbackMs)
                  << std::endl;
        return;
    }
    float timeF = (float)cloudTimeMs;
    uint32_t respHeader[3] = {CloudProtocol::STATUS_OK, *(uint32_t*)&timeF, elements};
    writeAll(clientSock, respHeader, 12);
    writeAll(clientSock, outData.data(), elements * sizeof(float));

    std::cout << "[CloudDaemon] OK cloudTime=" << cloudTimeMs << "ms elements=" << elements << std::endl;
#else
    uint32_t resp[3] = {CloudProtocol::STATUS_ERROR, 0, 0};
    writeAll(clientSock, resp, 12);
#endif
}
