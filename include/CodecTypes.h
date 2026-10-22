#ifndef EDGE_AI_CODEC_TYPES_H
#define EDGE_AI_CODEC_TYPES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class CodecKind : uint32_t {
    RAW32 = 0,
    FP16 = 1,
    BLOCK_INT8 = 2,
    BLOCK_INT6 = 3,
    BLOCK_INT4 = 4,
    LZ4 = 5,
};

struct CodecSpec {
    bool enabled{false};
    CodecKind kind{CodecKind::RAW32};
    uint32_t blockSize{0};
    bool allowRawFallback{true};
    // Audit mode only: locally decode and compare activations after the timed
    // encoder section. Keep false in normal inference to avoid hidden tax.
    bool measureReconstruction{false};
};

enum CodecPacketFlags : uint32_t {
    CODEC_FLAG_NONE = 0,
    CODEC_FLAG_RAW_FALLBACK = 1u << 0,
    CODEC_FLAG_NON_FINITE_FALLBACK = 1u << 1,
    CODEC_FLAG_NO_GAIN_FALLBACK = 1u << 2,
};

struct EncodedTensorPacket {
    int32_t tensorIndex{-1};
    std::string name;
    std::vector<int> shape;
    uint32_t dataType{0};
    uint32_t layout{0};
    uint32_t rawBytes{0};
    uint32_t elementCount{0};
    CodecKind codec{CodecKind::RAW32};
    uint32_t blockSize{0};
    uint32_t flags{CODEC_FLAG_NONE};
    std::vector<float> scales;
    std::vector<uint8_t> data;
};

struct AccuracyMetrics {
    bool available{false};
    double meanSquaredError{0.0};
    double meanAbsoluteError{0.0};
    double maxAbsoluteError{0.0};
    double cosineSimilarity{1.0};
    bool top1Match{true};
    std::size_t elements{0};
};

struct CodecProfile {
    bool enabled{false};
    CodecKind requested{CodecKind::RAW32};
    uint32_t blockSize{0};
    uint64_t startSteadyNs{0};
    uint64_t endSteadyNs{0};
    double elapsedMs{0.0};
    std::size_t rawTensorBytes{0};
    std::size_t encodedTensorBytes{0};
    std::size_t bundleBytes{0};
    std::size_t tensorBytesSaved{0};
    uint32_t tensorCount{0};
    uint32_t fallbackCount{0};
    uint32_t nonFiniteFallbackCount{0};
    uint32_t noGainFallbackCount{0};
    AccuracyMetrics reconstruction;

    double compressionRatio() const {
        return encodedTensorBytes > 0
            ? static_cast<double>(rawTensorBytes) / static_cast<double>(encodedTensorBytes)
            : 0.0;
    }
};

const char* codecKindName(CodecKind kind);
bool parseCodecKind(const std::string& value, CodecKind& kind);

#endif
