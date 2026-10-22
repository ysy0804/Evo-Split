#ifndef EDGE_AI_ACTIVATION_CODEC_H
#define EDGE_AI_ACTIVATION_CODEC_H

#include <string>
#include <vector>

#include "CodecTypes.h"
#include "TensorSerializer.h"

class ActivationCodec {
public:
    static bool encodeBundle(const std::vector<TensorPacket>& input,
                             const CodecSpec& spec,
                             std::vector<EncodedTensorPacket>& output,
                             CodecProfile& profile,
                             std::string* error = nullptr);

    static bool decodeBundle(const std::vector<EncodedTensorPacket>& input,
                             std::vector<TensorPacket>& output,
                             CodecProfile& profile,
                             std::string* error = nullptr);

    static AccuracyMetrics compare(const std::vector<float>& reference,
                                   const std::vector<float>& actual);

private:
    static bool encodeTensor(const TensorPacket& input,
                             const CodecSpec& spec,
                             EncodedTensorPacket& output,
                             bool& nonFinite,
                             std::string& error);
    static bool decodeTensor(const EncodedTensorPacket& input,
                             TensorPacket& output,
                             std::string& error);
};

#endif
