#include "CloudClient.h"
#include "CloudDaemon.h"
#include "Win32Socket.h"

#include <iostream>
#include <cstring>
#include <chrono>
#include <cerrno>

// Set nonblocking mode on supported platforms.
static void setNonBlocking(int fd, bool on) {
#ifdef _WIN32
    u_long mode = on ? 1 : 0;
    ioctlsocket(fd, FIONBIO, &mode);
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (on) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    else    fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
#endif
}

CloudClient::CloudClient() : sock_(-1) {}

CloudClient::~CloudClient() { disconnect(); }

bool CloudClient::connect(const std::string& host, int port) {
    sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_ < 0) {
        std::cerr << "[CloudClient] socket failed" << std::endl;
        return false;
    }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "[CloudClient] inet_pton failed for " << host << std::endl;
        close(sock_); sock_ = -1;
        return false;
    }

    // ═══════════════════════════════════════════════════
    //  Use nonblocking connect with a ten-second select timeout.
    //  Avoid long TCP connect stalls on the daemon thread.
    // ═══════════════════════════════════════════════════
    static const int CONNECT_TIMEOUT_SEC = 10;
    setNonBlocking(sock_, true);

    int ret = ::connect(sock_, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0) {
#ifdef _WIN32
        int err = WSAGetLastError();
        bool inProgress = (err == WSAEWOULDBLOCK || err == WSAEINPROGRESS);
#else
        bool inProgress = (errno == EINPROGRESS);
#endif
        if (!inProgress) {
            std::cerr << "[CloudClient] connect immediate fail to " << host << ":" << port
                      << " errno=" << errno << std::endl;
            close(sock_); sock_ = -1;
            return false;
        }

        // Wait for connection completion with a timeout.
        fd_set wset;
        FD_ZERO(&wset);
        FD_SET(sock_, &wset);
        struct timeval tv;
        tv.tv_sec = CONNECT_TIMEOUT_SEC;
        tv.tv_usec = 0;

        ret = select(sock_ + 1, nullptr, &wset, nullptr, &tv);
        if (ret <= 0) {
            std::cerr << "[CloudClient] connect timeout (" << CONNECT_TIMEOUT_SEC
                      << "s) to " << host << ":" << port << std::endl;
            close(sock_); sock_ = -1;
            return false;
        }

        // Check whether the connection succeeded.
        int soErr = 0;
        socklen_t soErrLen = sizeof(soErr);
        getsockopt(sock_, SOL_SOCKET, SO_ERROR,
#ifdef _WIN32
                   (char*)&soErr,
#else
                   &soErr,
#endif
                   &soErrLen);
        if (soErr != 0) {
            std::cerr << "[CloudClient] connect failed after select to " << host << ":" << port
                      << " err=" << soErr << std::endl;
            close(sock_); sock_ = -1;
            return false;
        }
    }

    setNonBlocking(sock_, false);  // Restore blocking mode.
    std::cout << "[CloudClient] ✅ connected to cloud " << host << ":" << port << std::endl;
    return true;
}

void CloudClient::disconnect() {
    if (sock_ >= 0) { close(sock_); sock_ = -1; }
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

static bool readAll(int fd, void* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t r = netRead(fd, (uint8_t*)buf + off, n - off);
        if (r <= 0) return false;
        off += r;
    }
    return true;
}

bool CloudClient::sendTLSync(const std::string& splitInput,
                             const std::string& outputName,
                             const std::vector<uint8_t>& tLBytes,
                             std::vector<float>& outData,
                             double& cloudTimeMs) {
    if (sock_ < 0) return false;

    // Build request: [4B magic][4B totalLen][4B inputNameLen][inputName][4B outputNameLen][outputName][T_L]
    uint32_t inputNameLen = (uint32_t)splitInput.size();
    uint32_t outputNameLen = (uint32_t)outputName.size();
    uint32_t bodyLen = 4 + inputNameLen + 4 + outputNameLen + (uint32_t)tLBytes.size();
    uint32_t header[3] = {CloudProtocol::MAGIC_REQUEST, bodyLen, inputNameLen};

    if (!writeAll(sock_, header, 12)) { std::cerr << "[CloudClient] header write fail" << std::endl; return false; }
    if (!writeAll(sock_, splitInput.data(), inputNameLen)) { std::cerr << "[CloudClient] name write fail" << std::endl; return false; }
    if (!writeAll(sock_, &outputNameLen, 4)) { return false; }
    if (!writeAll(sock_, outputName.data(), outputNameLen)) { return false; }
    if (!writeAll(sock_, tLBytes.data(), tLBytes.size())) { std::cerr << "[CloudClient] T_L write fail" << std::endl; return false; }

    // Read response: [4B status][4B cloudTimeMs(float)][4B elements][data...]
    uint32_t respHeader[3];
    if (!readAll(sock_, respHeader, 12)) { std::cerr << "[CloudClient] response read fail" << std::endl; return false; }

    if (respHeader[0] != CloudProtocol::STATUS_OK) {
        std::cerr << "[CloudClient] cloud returned error status=" << respHeader[0] << std::endl;
        return false;
    }

    cloudTimeMs = *(float*)&respHeader[1];
    uint32_t elements = respHeader[2];

    if (elements > 1000000) { return false; } // sanity
    outData.resize(elements);
    if (!readAll(sock_, outData.data(), elements * sizeof(float))) { return false; }

    std::cout << "[CloudClient] ✅ cloud result: cloudTime=" << cloudTimeMs << "ms elements=" << elements << std::endl;
    return true;
}

bool CloudClient::sendTLProfileSync(const std::string& splitInput,
                                    const std::string& outputName,
                                    const std::vector<uint8_t>& tLBytes,
                                    std::vector<float>& outData,
                                    RemoteCloudProfile& profile) {
    profile = RemoteCloudProfile();
    if (sock_ < 0) return false;
    const uint32_t inputNameLen = static_cast<uint32_t>(splitInput.size());
    const uint32_t outputNameLen = static_cast<uint32_t>(outputName.size());
    const uint32_t bodyLen = 4 + inputNameLen + 4 + outputNameLen + static_cast<uint32_t>(tLBytes.size());
    const uint32_t header[3] = {CloudProtocol::MAGIC_REQUEST_PROFILE, bodyLen, inputNameLen};
    if (!writeAll(sock_, header, sizeof(header)) ||
        !writeAll(sock_, splitInput.data(), inputNameLen) ||
        !writeAll(sock_, &outputNameLen, sizeof(outputNameLen)) ||
        !writeAll(sock_, outputName.data(), outputNameLen) ||
        !writeAll(sock_, tLBytes.data(), tLBytes.size())) return false;

    // [status u32][elements u32][session/deser/inject/outputCopy/serverTotal f32]
    uint32_t response[7];
    if (!readAll(sock_, response, sizeof(response)) || response[0] != CloudProtocol::STATUS_OK) return false;
    const uint32_t elements = response[1];
    if (elements > 10000000) return false;
    auto asFloat = [](uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return static_cast<double>(value);
    };
    profile.sessionMs = asFloat(response[2]);
    profile.deserializeGuardMs = asFloat(response[3]);
    profile.injectionMs = asFloat(response[4]);
    profile.outputCopyMs = asFloat(response[5]);
    profile.serverTotalMs = asFloat(response[6]);
    outData.resize(elements);
    return readAll(sock_, outData.data(), elements * sizeof(float));
}

bool CloudClient::sendTLCodecProfileSync(const std::string& splitInput,
                                         const std::string& outputName,
                                         const std::vector<uint8_t>& tLBytes,
                                         std::vector<float>& outData,
                                         RemoteCloudProfile& profile) {
    profile = RemoteCloudProfile();
    if (sock_ < 0) return false;
    if (splitInput.size() > UINT32_MAX || outputName.size() > UINT32_MAX ||
        tLBytes.size() > UINT32_MAX - 8u - splitInput.size() - outputName.size()) return false;
    const uint32_t inputNameLen = static_cast<uint32_t>(splitInput.size());
    const uint32_t outputNameLen = static_cast<uint32_t>(outputName.size());
    const uint32_t bodyLen = 4 + inputNameLen + 4 + outputNameLen + static_cast<uint32_t>(tLBytes.size());
    const uint32_t header[3] = {CloudProtocol::MAGIC_REQUEST_CODEC_PROFILE, bodyLen, inputNameLen};
    if (!writeAll(sock_, header, sizeof(header)) ||
        !writeAll(sock_, splitInput.data(), inputNameLen) ||
        !writeAll(sock_, &outputNameLen, sizeof(outputNameLen)) ||
        !writeAll(sock_, outputName.data(), outputNameLen) ||
        !writeAll(sock_, tLBytes.data(), tLBytes.size())) return false;

    uint32_t response[17];
    if (!readAll(sock_, response, sizeof(response)) || response[0] != CloudProtocol::STATUS_OK) return false;
    const uint32_t elements = response[1];
    if (elements > 10000000) return false;
    auto asFloat = [](uint32_t bits) {
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return static_cast<double>(value);
    };
    auto asU64 = [](uint32_t low, uint32_t high) {
        return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
    };
    profile.sessionMs = asFloat(response[2]);
    profile.deserializeGuardMs = asFloat(response[3]);
    profile.injectionMs = asFloat(response[4]);
    profile.outputCopyMs = asFloat(response[5]);
    profile.serverTotalMs = asFloat(response[6]);
    profile.codec.enabled = (response[12] & 0x80000000u) != 0;
    const uint32_t codecValue = response[12] & 0x7fffffffu;
    if (codecValue > static_cast<uint32_t>(CodecKind::LZ4)) return false;
    profile.codec.requested = static_cast<CodecKind>(codecValue);
    profile.codec.elapsedMs = asFloat(response[7]);
    profile.codec.rawTensorBytes = response[8];
    profile.codec.encodedTensorBytes = response[9];
    profile.codec.bundleBytes = response[10];
    profile.codec.fallbackCount = response[11];
    profile.codec.startSteadyNs = asU64(response[13], response[14]);
    profile.codec.endSteadyNs = asU64(response[15], response[16]);
    profile.codec.tensorBytesSaved = profile.codec.rawTensorBytes > profile.codec.encodedTensorBytes
        ? profile.codec.rawTensorBytes - profile.codec.encodedTensorBytes : 0;
    outData.resize(elements);
    return readAll(sock_, outData.data(), elements * sizeof(float));
}
