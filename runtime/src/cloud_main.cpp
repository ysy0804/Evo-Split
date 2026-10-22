#include <iostream>
#include <csignal>
#include <memory>
#include <string>
#include <algorithm>
#include <MNN/MNNForwardType.h>
#include "CloudDaemon.h"
#include "Win32Socket.h"

std::unique_ptr<CloudDaemon> g_cloudDaemon;

void sigHandler(int sig) {
    std::cerr << "[CloudDaemon] signal " << sig << ", shutting down..." << std::endl;
    if (g_cloudDaemon) g_cloudDaemon->stop();
    exit(0);
}

int main(int argc, char* argv[]) {
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);
    net_init();  // WSAStartup on Windows

    int port = 24877;
    std::string modelPath = "/mnt/c/Users/ysy/Desktop/edge-ai-workbench/resource/models/model/20260801-000022/model.mnn";
    if (argc > 1) port = std::atoi(argv[1]);
    if (argc > 2) modelPath = argv[2];
    std::string backend = argc > 3 ? argv[3] : "cpu";
    int numThread = argc > 4 ? std::max(1, std::atoi(argv[4])) : 4;
    int backendType = MNN_FORWARD_CPU;
    if (backend == "cuda") {
        backendType = MNN_FORWARD_CUDA;
        numThread = 1;
    } else if (backend != "cpu") {
        std::cerr << "Usage: cloud_daemon [port] [model.mnn] [cpu|cuda] [cpu_threads]" << std::endl;
        return 2;
    }

    std::cout << "[CloudDaemon] Starting on port " << port << std::endl;
    std::cout << "[CloudDaemon] Model: " << modelPath << std::endl;
    std::cout << "[CloudDaemon] Backend: " << backend << ", threads: " << numThread << std::endl;

    g_cloudDaemon = std::make_unique<CloudDaemon>(port);
    g_cloudDaemon->setModelPath(modelPath);
    g_cloudDaemon->setBackend(backendType, numThread);
    g_cloudDaemon->start();
    return 0;
}
