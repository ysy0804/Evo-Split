#ifndef EDGE_AI_CLOUD_DAEMON_H
#define EDGE_AI_CLOUD_DAEMON_H

#include <string>
#include <cstdint>
#include <atomic>
#include <memory>
#include <map>

#include "CutSetPlanner.h"

class CloudInjectionEngine;

class CloudDaemon {
public:
    CloudDaemon(int port = 24877);
    ~CloudDaemon();

    void setModelPath(const std::string& path) { modelPath_ = path; }
    void setBackend(int backendType, int numThread) {
        backendType_ = backendType;
        numThread_ = numThread > 0 ? numThread : 1;
    }
    void start();
    void stop();

private:
    int port_;
    int serverSocket_;
    std::string modelPath_;
    std::atomic<bool> running_{false};

    // Preloaded injection engine (full-model + before-callback inject, supports fused layers)
    std::unique_ptr<CloudInjectionEngine> cloudEngine_;
    std::string splitInputName_;
    std::map<std::string, CutSetPlan> verifiedPlanCache_;
    uint64_t cachedModelFingerprint_{0};
    int backendType_{0};
    int numThread_{4};

    bool initServer();
    void handleConnection(int clientSock);
};

namespace CloudProtocol {
    constexpr uint32_t MAGIC_REQUEST = 0x53544E4D;
    constexpr uint32_t MAGIC_REQUEST_PROFILE = 0x50544E4D;
    constexpr uint32_t MAGIC_REQUEST_CODEC_PROFILE = 0x43544E4D;
    constexpr uint32_t STATUS_OK     = 0;
    constexpr uint32_t STATUS_ERROR  = 1;
}

#endif
