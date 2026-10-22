#include "ActivationCodec.h"
#include "TensorSerializer.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

namespace {

TensorPacket makePacket(int index, const std::string& name, int count, float phase = 0.0f) {
    TensorPacket packet;
    packet.tensorIndex = index;
    packet.name = name;
    packet.shape = {1, count};
    packet.dataType = 0;
    packet.layout = 0;
    packet.data.resize(static_cast<size_t>(count) * sizeof(float));
    float* values = reinterpret_cast<float*>(packet.data.data());
    for (int i = 0; i < count; ++i) {
        values[i] = std::sin(i * 0.071f + phase) * (1.0f + (i % 17) * 0.1f);
    }
    return packet;
}

bool sameBytes(const TensorPacket& a, const TensorPacket& b) {
    return a.tensorIndex == b.tensorIndex && a.name == b.name && a.shape == b.shape &&
           a.dataType == b.dataType && a.layout == b.layout && a.data == b.data;
}

bool runCodec(CodecKind kind, uint32_t blockSize, double maxError,
              bool requireSmaller, bool requireExact = false) {
    std::vector<TensorPacket> input = {makePacket(7, "cut/a", 257), makePacket(9, "cut/b", 33, 0.3f)};
    CodecSpec spec;
    spec.enabled = true;
    spec.kind = kind;
    spec.blockSize = blockSize;
    spec.measureReconstruction = true;
    std::vector<EncodedTensorPacket> encoded;
    CodecProfile encodeProfile;
    std::string error;
    if (!ActivationCodec::encodeBundle(input, spec, encoded, encodeProfile, &error)) {
        std::cerr << "encode failed " << codecKindName(kind) << ": " << error << std::endl;
        return false;
    }
    CutBundleMetadata metadata;
    metadata.splitAfterOp = 12;
    metadata.modelFingerprint = 0x123456789abcdef0ULL;
    auto bundle = TensorSerializer::serializeEncodedBundle(encoded, metadata);
    if (bundle.empty()) return false;
    uint32_t version = 0;
    if (!TensorSerializer::inspectBundleVersion(bundle, version) ||
        version != TensorSerializer::CODEC_BUNDLE_VERSION) return false;
    std::vector<EncodedTensorPacket> parsed;
    CutBundleMetadata parsedMetadata;
    if (!TensorSerializer::deserializeEncodedBundle(bundle, parsed, &error, &parsedMetadata) ||
        parsedMetadata.splitAfterOp != metadata.splitAfterOp ||
        parsedMetadata.modelFingerprint != metadata.modelFingerprint) return false;
    std::vector<TensorPacket> decoded;
    CodecProfile decodeProfile;
    if (!ActivationCodec::decodeBundle(parsed, decoded, decodeProfile, &error) ||
        decoded.size() != input.size()) return false;
    for (size_t p = 0; p < input.size(); ++p) {
        const float* reference = reinterpret_cast<const float*>(input[p].data.data());
        const float* actual = reinterpret_cast<const float*>(decoded[p].data.data());
        for (size_t i = 0; i < input[p].data.size() / sizeof(float); ++i) {
            if (std::fabs(reference[i] - actual[i]) > maxError) return false;
        }
        if (requireExact && !sameBytes(input[p], decoded[p])) return false;
    }
    if (!encodeProfile.reconstruction.available ||
        encodeProfile.reconstruction.maxAbsoluteError > maxError) return false;
    if (requireSmaller && encodeProfile.encodedTensorBytes >= encodeProfile.rawTensorBytes) return false;
    if (encodeProfile.startSteadyNs == 0 || encodeProfile.endSteadyNs < encodeProfile.startSteadyNs ||
        decodeProfile.startSteadyNs == 0 || decodeProfile.endSteadyNs < decodeProfile.startSteadyNs) return false;
    std::vector<TensorPacket> compatibilityDecoded;
    if (!TensorSerializer::deserializeBundle(bundle, compatibilityDecoded, &error) ||
        compatibilityDecoded.size() != input.size()) return false;
    auto corrupted = bundle;
    corrupted.back() ^= 0x40;
    if (TensorSerializer::deserializeEncodedBundle(corrupted, parsed, &error, nullptr)) return false;
    std::cout << codecKindName(kind) << " raw=" << encodeProfile.rawTensorBytes
              << " encoded=" << encodeProfile.encodedTensorBytes
              << " maxError=" << encodeProfile.reconstruction.maxAbsoluteError << std::endl;
    return true;
}

bool fallbackTest() {
    TensorPacket packet = makePacket(3, "cut/nonfinite", 8);
    reinterpret_cast<float*>(packet.data.data())[2] = std::numeric_limits<float>::quiet_NaN();
    CodecSpec spec;
    spec.enabled = true;
    spec.kind = CodecKind::BLOCK_INT8;
    spec.blockSize = 4;
    spec.allowRawFallback = true;
    std::vector<EncodedTensorPacket> encoded;
    CodecProfile profile;
    std::string error;
    if (!ActivationCodec::encodeBundle({packet}, spec, encoded, profile, &error)) return false;
    if (profile.fallbackCount != 1 || profile.nonFiniteFallbackCount != 1 ||
        encoded[0].codec != CodecKind::RAW32 ||
        (encoded[0].flags & CODEC_FLAG_NON_FINITE_FALLBACK) == 0) return false;
    spec.allowRawFallback = false;
    return !ActivationCodec::encodeBundle({packet}, spec, encoded, profile, &error);
}

bool noGainFallbackTest() {
    TensorPacket packet = makePacket(4, "cut/tiny", 1);
    CodecSpec spec;
    spec.enabled = true;
    spec.kind = CodecKind::BLOCK_INT8;
    spec.blockSize = 64;
    spec.allowRawFallback = true;
    std::vector<EncodedTensorPacket> encoded;
    CodecProfile profile;
    std::string error;
    return ActivationCodec::encodeBundle({packet}, spec, encoded, profile, &error) &&
           encoded.size() == 1 && encoded[0].codec == CodecKind::RAW32 &&
           profile.fallbackCount == 1 && profile.noGainFallbackCount == 1 &&
           (encoded[0].flags & CODEC_FLAG_NO_GAIN_FALLBACK) != 0;
}

bool lz4Test() {
    TensorPacket packet;
    packet.tensorIndex = 5;
    packet.name = "cut/lz4";
    packet.shape = {1, 1024};
    packet.dataType = 0;
    packet.layout = 0;
    packet.data.resize(1024 * sizeof(float), 0);
    float* values = reinterpret_cast<float*>(packet.data.data());
    for (int i = 0; i < 1024; i += 64) values[i] = 1.0f;
    CodecSpec spec;
    spec.enabled = true;
    spec.kind = CodecKind::LZ4;
    spec.measureReconstruction = true;
    std::vector<EncodedTensorPacket> encoded;
    CodecProfile encodeProfile;
    std::string error;
    if (!ActivationCodec::encodeBundle({packet}, spec, encoded, encodeProfile, &error) ||
        encoded.size() != 1 || encoded[0].codec != CodecKind::LZ4 ||
        encoded[0].data.size() >= packet.data.size() ||
        !encodeProfile.reconstruction.available ||
        encodeProfile.reconstruction.maxAbsoluteError != 0.0) return false;
    CutBundleMetadata metadata;
    metadata.splitAfterOp = 2;
    metadata.modelFingerprint = 123;
    const auto bundle = TensorSerializer::serializeEncodedBundle(encoded, metadata);
    std::vector<TensorPacket> decoded;
    if (!TensorSerializer::deserializeBundle(bundle, decoded, &error) ||
        decoded.size() != 1 || !sameBytes(packet, decoded[0])) return false;
    std::cout << "LZ4 raw=" << packet.data.size() << " encoded=" << encoded[0].data.size()
              << " hex=";
    for (uint8_t byte : encoded[0].data) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    }
    std::cout << std::dec << std::setfill(' ') << std::endl;
    auto corrupt = encoded;
    corrupt[0].data = {0x00, 0x00, 0x00};  // illegal zero-distance match
    std::vector<TensorPacket> invalid;
    CodecProfile invalidProfile;
    return !ActivationCodec::decodeBundle(corrupt, invalid, invalidProfile, &error);
}

}  // namespace

int main() {
    if (!runCodec(CodecKind::RAW32, 0, 0.0, false, true)) return 1;
    if (!runCodec(CodecKind::FP16, 0, 0.002, true)) return 2;
    if (!runCodec(CodecKind::BLOCK_INT8, 64, 0.03, true)) return 3;
    if (!runCodec(CodecKind::BLOCK_INT6, 32, 0.12, true)) return 4;
    if (!runCodec(CodecKind::BLOCK_INT4, 16, 0.55, true)) return 5;
    if (!fallbackTest()) return 6;
    if (!noGainFallbackTest()) return 7;
    if (!lz4Test()) return 8;
    std::cout << "ACTIVATION_CODEC_TEST=PASS" << std::endl;
    return 0;
}
