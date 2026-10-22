#ifndef EDGE_AI_CLOUD_CLIENT_H
#define EDGE_AI_CLOUD_CLIENT_H

#include <string>
#include <vector>
#include <cstdint>

#include "CodecTypes.h"

struct RemoteCloudProfile {
    double sessionMs{0.0};
    double deserializeGuardMs{0.0};
    double injectionMs{0.0};
    double outputCopyMs{0.0};
    double serverTotalMs{0.0};
    CodecProfile codec;
};

/**
 * Edge TCP client that sends T_L to CloudDaemon over a real network.
 *
 * Usage:
 *   CloudClient client;
 *   if (client.connect("127.0.0.1", 24877)) {
 *       double cloudTimeMs = 0;
 *       client.sendTLSync(splitInputName, outputName, tLBytes, outData, cloudTimeMs);
 *   }
 */
class CloudClient {
public:
    CloudClient();
    ~CloudClient();

    /**
     * Connect to the cloud daemon.
     * @param host IP address
     * @param port port number
     * @return true on success
     */
    bool connect(const std::string& host, int port);

    /**
     * Disconnect.
     */
    void disconnect();

    /**
     * Send T_L synchronously and wait for returned logits.
     * @param splitInput boundary output tensor used as the cloud input
     * @param outputName final output tensor name, for example logits
     * @param tLBytes    T_L payload in TensorSerializer format
     * @param outData    returned FP32 output
     * @param cloudTimeMs returned cloud latency in milliseconds
     * @return true on success
     */
    bool sendTLSync(const std::string& splitInput,
                    const std::string& outputName,
                    const std::vector<uint8_t>& tLBytes,
                    std::vector<float>& outData,
                    double& cloudTimeMs);

    // B2 profile protocol. The payload is wire-compatible with sendTLSync;
    // only the request magic and response header are extended.
    bool sendTLProfileSync(const std::string& splitInput,
                           const std::string& outputName,
                           const std::vector<uint8_t>& tLBytes,
                           std::vector<float>& outData,
                           RemoteCloudProfile& profile);

    bool sendTLCodecProfileSync(const std::string& splitInput,
                                const std::string& outputName,
                                const std::vector<uint8_t>& tLBytes,
                                std::vector<float>& outData,
                                RemoteCloudProfile& profile);

private:
    int sock_;
};

#endif
