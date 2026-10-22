#include "ActivationCodec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

// ============================================================
//  Return the current steady-clock timestamp in nanoseconds.
// ============================================================
uint64_t steadyNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}



// ============================================================
//  Return the default quantization block size.
// ============================================================
uint32_t defaultBlockSize(CodecKind kind) {
    switch (kind) {
        case CodecKind::BLOCK_INT8: return 256;
        case CodecKind::BLOCK_INT6: return 128;
        case CodecKind::BLOCK_INT4: return 64;
        default: return 0;
    }
}

// ============================================================
//  Convert a float to its binary representation.
// ============================================================
uint32_t floatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}


// ============================================================
//  Convert a binary representation to float.
// ============================================================
float bitsFloat(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ============================================================
//  Quantize a float to IEEE FP16.
// ============================================================
uint16_t floatToHalf(float value) {
    const uint32_t bits = floatBits(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));
    }
    int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (halfExponent <= 0) {
        if (halfExponent < -10) return static_cast<uint16_t>(sign);
        uint32_t normalized = mantissa | 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - halfExponent);
        uint32_t rounded = normalized >> shift;
        const uint32_t remainder = normalized & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t roundedMantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (roundedMantissa & 1u))) {
        ++roundedMantissa;
        if (roundedMantissa == 0x400u) {
            roundedMantissa = 0;
            ++halfExponent;
            if (halfExponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExponent) << 10) |
                                 roundedMantissa);
}


// ============================================================
//  Convert IEEE FP16 to float.
// ============================================================
float halfToFloat(uint16_t half) {
    const uint32_t sign = (static_cast<uint32_t>(half & 0x8000u)) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t e = -14;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return bitsFloat(bits);
}

// ============================================================
//  Pack signed integers at the requested bit width.
// ============================================================
void packSigned(const std::vector<int32_t>& values, uint32_t bitWidth,
                std::vector<uint8_t>& output) {
    output.clear();
    output.reserve((values.size() * bitWidth + 7) / 8);
    uint64_t buffer = 0;
    uint32_t buffered = 0;
    const uint32_t mask = (1u << bitWidth) - 1u;
    for (int32_t value : values) {
        buffer |= (static_cast<uint32_t>(value) & mask) << buffered;
        buffered += bitWidth;
        while (buffered >= 8) {
            output.push_back(static_cast<uint8_t>(buffer & 0xffu));
            buffer >>= 8;
            buffered -= 8;
        }
    }
    if (buffered > 0) output.push_back(static_cast<uint8_t>(buffer & 0xffu));
}



// ============================================================
//  Unpack signed integers from a bit stream.
// ============================================================
bool unpackSigned(const std::vector<uint8_t>& input, std::size_t count,
                  uint32_t bitWidth, std::vector<int32_t>& output) {
    if (input.size() != (count * bitWidth + 7) / 8) return false;
    output.clear();
    output.reserve(count);
    uint64_t buffer = 0;
    uint32_t buffered = 0;
    std::size_t offset = 0;
    const uint32_t mask = (1u << bitWidth) - 1u;
    const uint32_t sign = 1u << (bitWidth - 1u);
    while (output.size() < count) {
        while (buffered < bitWidth && offset < input.size()) {
            buffer |= static_cast<uint64_t>(input[offset++]) << buffered;
            buffered += 8;
        }
        if (buffered < bitWidth) return false;
        uint32_t code = static_cast<uint32_t>(buffer) & mask;
        buffer >>= bitWidth;
        buffered -= bitWidth;
        output.push_back((code & sign) ? static_cast<int32_t>(code) - static_cast<int32_t>(1u << bitWidth)
                                       : static_cast<int32_t>(code));
    }
    return true;
}


// ============================================================
//  Read an unaligned 32-bit unsigned integer.
// ============================================================
uint32_t readU32Unaligned(const uint8_t* data) {
    uint32_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

// ============================================================
//  Append an LZ4 length field.
// ============================================================
void appendLz4Length(std::vector<uint8_t>& output, std::size_t length) {
    while (length >= 255) {
        output.push_back(255);
        length -= 255;
    }
    output.push_back(static_cast<uint8_t>(length));
}




// Standards-compatible raw LZ4 block. MCUT already carries the uncompressed
// size, so an additional LZ4 frame header is unnecessary.
// ============================================================
//  Compress a block using the lossless LZ4 format.
// ============================================================
bool lz4CompressBlock(const uint8_t* input, std::size_t size,
                      std::vector<uint8_t>& output) {
    // Validate input arguments.
    if (!input || size == 0) return false;
    // Initialize the hash table and output buffer.
    constexpr std::size_t HASH_SIZE = 1u << 16;
    const std::size_t invalid = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> table(HASH_SIZE, invalid);
    // Reserve output space for the worst-case expansion.
    output.clear();
    output.reserve(size + size / 255 + 16);
    // Initialize the anchor and cursor.
    std::size_t anchor = 0;
    std::size_t position = 0;
    // LZ4 block end rules: the last match must start at least 12 bytes before
    // the end and the final sequence must contain at least 5 literals.
    // Main compression loop.
       while (position + 12 <= size) {
        // Hash the current four-byte sequence and find its reference.
        const uint32_t sequence = readU32Unaligned(input + position);
        const std::size_t hash = (sequence * 2654435761u) >> 16;
        const std::size_t reference = table[hash];
        // Store the current position in the hash table.
        table[hash] = position;
        // Validate the reference position.
        if (reference == invalid || position <= reference || position - reference > 65535 ||
            reference + 4 > size || std::memcmp(input + reference, input + position, 4) != 0) {
            ++position;
            continue;
        }
        // Count the current match length.
        const std::size_t literalLength = position - anchor;
        const std::size_t tokenOffset = output.size();
        // Write the token containing literal and match lengths.
        output.push_back(0);
        output[tokenOffset] = static_cast<uint8_t>(std::min<std::size_t>(literalLength, 15) << 4);
        if (literalLength >= 15) appendLz4Length(output, literalLength - 15);
        output.insert(output.end(), input + anchor, input + position);
        // Write the match offset.
        const uint16_t offset = static_cast<uint16_t>(position - reference);
        output.push_back(static_cast<uint8_t>(offset));
        output.push_back(static_cast<uint8_t>(offset >> 8));
        const std::size_t matchStart = position;
        std::size_t matchReference = reference;
        position += 4;
        matchReference += 4;
        const std::size_t matchLimit = size - 5;
        while (position < matchLimit && input[position] == input[matchReference]) {
            ++position;
            ++matchReference;
        }
        // Count the current match length.
        const std::size_t matchCode = position - matchStart - 4;
        output[tokenOffset] |= static_cast<uint8_t>(std::min<std::size_t>(matchCode, 15));
        if (matchCode >= 15) appendLz4Length(output, matchCode - 15);
        anchor = position;
    }
    // Process trailing literals.
    const std::size_t literalLength = size - anchor;
    output.push_back(static_cast<uint8_t>(std::min<std::size_t>(literalLength, 15) << 4));
    if (literalLength >= 15) appendLz4Length(output, literalLength - 15);
    // Write trailing literals.
    output.insert(output.end(), input + anchor, input + size);
    return true;
}


// ============================================================
//  Read a length field from an LZ4 block.
// ============================================================
bool readLz4Length(const std::vector<uint8_t>& input, std::size_t& offset,
                   std::size_t initial, std::size_t& length) {
    length = initial;
    if (initial != 15) return true;
    uint8_t extension = 255;
    while (extension == 255) {
        if (offset >= input.size()) return false;
        extension = input[offset++];
        if (length > std::numeric_limits<std::size_t>::max() - extension) return false;
        length += extension;
    }
    return true;
}

// ============================================================
//  Decompress an LZ4 block.
// ============================================================
bool lz4DecompressBlock(const std::vector<uint8_t>& input, std::size_t expectedSize,
                        std::vector<uint8_t>& output) {
    if (input.empty() || expectedSize == 0) return false;
    output.clear();
    output.reserve(expectedSize);
    std::size_t offset = 0;
    while (offset < input.size()) {
        const uint8_t token = input[offset++];
        std::size_t literalLength = 0;
        if (!readLz4Length(input, offset, token >> 4, literalLength) ||
            literalLength > input.size() - offset ||
            literalLength > expectedSize - output.size()) return false;
        output.insert(output.end(), input.begin() + offset, input.begin() + offset + literalLength);
        offset += literalLength;
        if (offset == input.size()) return output.size() == expectedSize;
        if (input.size() - offset < 2) return false;
        const uint16_t distance = static_cast<uint16_t>(input[offset]) |
                                  (static_cast<uint16_t>(input[offset + 1]) << 8);
        offset += 2;
        if (distance == 0 || distance > output.size()) return false;
        std::size_t matchCode = 0;
        if (!readLz4Length(input, offset, token & 0x0f, matchCode) ||
            matchCode > std::numeric_limits<std::size_t>::max() - 4) return false;
        const std::size_t matchLength = matchCode + 4;
        if (matchLength > expectedSize - output.size()) return false;
        const std::size_t match = output.size() - distance;
        for (std::size_t i = 0; i < matchLength; ++i) output.push_back(output[match + i]);
    }
    return output.size() == expectedSize;
}


// ============================================================
//  Validate a floating-point tensor packet.
// ============================================================
bool validFloatPacket(const TensorPacket& packet, std::size_t& elements) {
    if (packet.dataType != 0 || packet.data.empty() || packet.data.size() % sizeof(float) != 0) return false;
    elements = packet.data.size() / sizeof(float);
    std::size_t shapeElements = 1;
    if (packet.shape.empty()) return false;
    for (int dim : packet.shape) {
        if (dim <= 0 || shapeElements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim)) {
            return false;
        }
        shapeElements *= static_cast<std::size_t>(dim);
    }
    return shapeElements == elements;
}


// ============================================================
//  Compare two floating-point arrays.
// ============================================================
AccuracyMetrics comparePointers(const float* reference, const float* actual, std::size_t count) {
    AccuracyMetrics metrics;
    if (!reference || !actual || count == 0) return metrics;
    long double squared = 0.0, absolute = 0.0, dot = 0.0, refNorm = 0.0, actualNorm = 0.0;
    double maxAbs = 0.0;
    std::size_t refTop = 0, actualTop = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) return metrics;
        const double diff = static_cast<double>(actual[i]) - static_cast<double>(reference[i]);
        squared += diff * diff;
        absolute += std::fabs(diff);
        maxAbs = std::max(maxAbs, std::fabs(diff));
        dot += static_cast<long double>(reference[i]) * actual[i];
        refNorm += static_cast<long double>(reference[i]) * reference[i];
        actualNorm += static_cast<long double>(actual[i]) * actual[i];
        if (reference[i] > reference[refTop]) refTop = i;
        if (actual[i] > actual[actualTop]) actualTop = i;
    }
    metrics.available = true;
    metrics.elements = count;
    metrics.meanSquaredError = static_cast<double>(squared / count);
    metrics.meanAbsoluteError = static_cast<double>(absolute / count);
    metrics.maxAbsoluteError = maxAbs;
    if (refNorm == 0.0L && actualNorm == 0.0L) metrics.cosineSimilarity = 1.0;
    else if (refNorm == 0.0L || actualNorm == 0.0L) metrics.cosineSimilarity = 0.0;
    else metrics.cosineSimilarity = static_cast<double>(dot / std::sqrt(refNorm * actualNorm));
    metrics.top1Match = refTop == actualTop;
    return metrics;
}

// ============================================================
//  Merge two reconstruction-accuracy records.
// ============================================================
void mergeAccuracy(AccuracyMetrics& aggregate, const AccuracyMetrics& next) {
    if (!next.available) return;
    if (!aggregate.available) {
        aggregate = next;
        return;
    }
    const std::size_t total = aggregate.elements + next.elements;
    aggregate.meanSquaredError =
        (aggregate.meanSquaredError * aggregate.elements + next.meanSquaredError * next.elements) / total;
    aggregate.meanAbsoluteError =
        (aggregate.meanAbsoluteError * aggregate.elements + next.meanAbsoluteError * next.elements) / total;
    aggregate.maxAbsoluteError = std::max(aggregate.maxAbsoluteError, next.maxAbsoluteError);
    aggregate.cosineSimilarity =
        (aggregate.cosineSimilarity * aggregate.elements + next.cosineSimilarity * next.elements) / total;
    aggregate.top1Match = aggregate.top1Match && next.top1Match;
    aggregate.elements = total;
}

}  // namespace


// ============================================================
//  Return the codec name used by configuration and telemetry.
// ============================================================
const char* codecKindName(CodecKind kind) {
    switch (kind) {
        case CodecKind::RAW32: return "RAW32";
        case CodecKind::FP16: return "FP16";
        case CodecKind::BLOCK_INT8: return "BLOCK_INT8";
        case CodecKind::BLOCK_INT6: return "BLOCK_INT6";
        case CodecKind::BLOCK_INT4: return "BLOCK_INT4";
        case CodecKind::LZ4: return "LZ4";
    }
    return "UNKNOWN";
}


// ============================================================
//  Parse a codec name into its enum value.
// ============================================================
bool parseCodecKind(const std::string& value, CodecKind& kind) {
    if (value == "RAW32" || value == "raw32") kind = CodecKind::RAW32;
    else if (value == "FP16" || value == "fp16") kind = CodecKind::FP16;
    else if (value == "BLOCK_INT8" || value == "INT8" || value == "int8") kind = CodecKind::BLOCK_INT8;
    else if (value == "BLOCK_INT6" || value == "INT6" || value == "int6") kind = CodecKind::BLOCK_INT6;
    else if (value == "BLOCK_INT4" || value == "INT4" || value == "int4") kind = CodecKind::BLOCK_INT4;
    else if (value == "LZ4" || value == "lz4") kind = CodecKind::LZ4;
    else return false;
    return true;
}


// ============================================================
//  Encode a tensor using the requested codec and block size.
// ============================================================
bool ActivationCodec::encodeTensor(const TensorPacket& input, const CodecSpec& requested,
                                   EncodedTensorPacket& output, bool& nonFinite,
                                   std::string& error) {
    std::size_t elements = 0;
    // Validate tensor type, shape, and element count.
    if (!validFloatPacket(input, elements) || elements > std::numeric_limits<uint32_t>::max() ||
        input.data.size() > std::numeric_limits<uint32_t>::max()) {
        error = "invalid FP32 tensor packet";
        return false;
    }
    output = EncodedTensorPacket();
    output.tensorIndex = input.tensorIndex;
    output.name = input.name;
    output.shape = input.shape;
    output.dataType = input.dataType;
    output.layout = input.layout;
    output.rawBytes = static_cast<uint32_t>(input.data.size());
    output.elementCount = static_cast<uint32_t>(elements);
    output.codec = requested.kind;
    output.blockSize = requested.blockSize ? requested.blockSize : defaultBlockSize(requested.kind);
    // View the input payload as floating-point data.
    const float* values = reinterpret_cast<const float*>(input.data.data());
    nonFinite = false;
    if (requested.kind == CodecKind::RAW32) {
        output.blockSize = 0;
        output.data = input.data;
        return true;
    }
    if (requested.kind == CodecKind::LZ4) {
        output.blockSize = 0;
        // Compress the input payload and report failures.
        if (!lz4CompressBlock(input.data.data(), input.data.size(), output.data)) {
            error = "LZ4 compression failed";
            return false;
        }
        return true;
    }
    // Reject non-finite input values such as NaN or infinity.
    for (std::size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(values[i])) {
            nonFinite = true;
            error = "non-finite activation";
            return false;
        }
    }
    if (requested.kind == CodecKind::FP16) {
        output.blockSize = 0;
        output.data.resize(elements * sizeof(uint16_t));
        for (std::size_t i = 0; i < elements; ++i) {
            const uint16_t half = floatToHalf(values[i]);
            output.data[i * 2] = static_cast<uint8_t>(half);
            output.data[i * 2 + 1] = static_cast<uint8_t>(half >> 8);
        }
        return true;
    }
    uint32_t bits = 0, qmax = 0;
    // Select the quantization width and integer range.
    if (requested.kind == CodecKind::BLOCK_INT8) { bits = 8; qmax = 127; }
    else if (requested.kind == CodecKind::BLOCK_INT6) { bits = 6; qmax = 31; }
    else if (requested.kind == CodecKind::BLOCK_INT4) { bits = 4; qmax = 7; }
    else { error = "unsupported codec"; return false; }
    // Validate the requested block size.
    if (output.blockSize == 0 || output.blockSize > (1u << 20)) {
        error = "invalid block size";
        return false;
    }
    // Compute the number of quantization blocks.
    const std::size_t blocks = (elements + output.blockSize - 1) / output.blockSize;
    output.scales.reserve(blocks);
    // Quantize floating-point values and record per-block scales.
    std::vector<int32_t> quantized(elements);
    // Compute each block scale from its maximum absolute value.
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::size_t begin = block * output.blockSize;
        const std::size_t end = std::min(elements, begin + output.blockSize);
        float maxAbs = 0.0f;
        for (std::size_t i = begin; i < end; ++i) maxAbs = std::max(maxAbs, std::fabs(values[i]));
        const float scale = maxAbs > 0.0f ? maxAbs / static_cast<float>(qmax) : 1.0f;
        output.scales.push_back(scale);
        // Quantize and clamp each value to the signed integer range.
        for (std::size_t i = begin; i < end; ++i) {
            const long rounded = std::lround(values[i] / scale);
            quantized[i] = static_cast<int32_t>(std::max(-static_cast<long>(qmax),
                                                         std::min(static_cast<long>(qmax), rounded)));
        }
    }
    // Pack quantized integers at the requested bit width.
    packSigned(quantized, bits, output.data);
    return true;
}


// ============================================================
// Decode and dequantize an activation tensor packet.
// ============================================================
bool ActivationCodec::decodeTensor(const EncodedTensorPacket& input, TensorPacket& output,
                                   std::string& error) {
    if (input.elementCount == 0 ||
        static_cast<uint64_t>(input.rawBytes) !=
            static_cast<uint64_t>(input.elementCount) * sizeof(float)) {
        error = "invalid encoded element count";
        return false;
    }
    std::size_t shapeElements = 1;
    if (input.shape.empty()) { error = "empty encoded shape"; return false; }
    // Validate the encoded tensor shape and limits.
    for (int dim : input.shape) {
        if (dim <= 0 || shapeElements > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim)) {
            error = "invalid encoded shape";
            return false;
        }
        shapeElements *= static_cast<std::size_t>(dim);
    }
    //
    if (shapeElements != input.elementCount || input.dataType != 0 || input.layout > 1) {
        error = "encoded metadata mismatch";
        return false;
    }
    output = TensorPacket();
    output.tensorIndex = input.tensorIndex;
    output.name = input.name;
    output.shape = input.shape;
    output.dataType = input.dataType;
    output.layout = input.layout;
    output.data.resize(input.rawBytes);
    float* values = reinterpret_cast<float*>(output.data.data());
    if (input.codec == CodecKind::RAW32) {
        if (input.data.size() != input.rawBytes || !input.scales.empty()) {
            error = "invalid RAW32 payload";
            return false;
        }
        output.data = input.data;
        return true;
    }
    if (input.codec == CodecKind::LZ4) {
        if (!input.scales.empty() || input.blockSize != 0 ||
            !lz4DecompressBlock(input.data, input.rawBytes, output.data)) {
            error = "invalid LZ4 payload";
            return false;
        }
        return true;
    }
    if (input.codec == CodecKind::FP16) {
        if (input.data.size() != input.elementCount * sizeof(uint16_t) || !input.scales.empty()) {
            error = "invalid FP16 payload";
            return false;
        }
        for (std::size_t i = 0; i < input.elementCount; ++i) {
            const uint16_t half = static_cast<uint16_t>(input.data[i * 2]) |
                                  (static_cast<uint16_t>(input.data[i * 2 + 1]) << 8);
            values[i] = halfToFloat(half);
        }
        return true;
    }
    uint32_t bits = 0;
    if (input.codec == CodecKind::BLOCK_INT8) bits = 8;
    else if (input.codec == CodecKind::BLOCK_INT6) bits = 6;
    else if (input.codec == CodecKind::BLOCK_INT4) bits = 4;
    else { error = "unknown encoded codec"; return false; }
    if (input.blockSize == 0 || input.blockSize > (1u << 20)) {
        error = "invalid encoded block size";
        return false;
    }
    // Verify the number of quantization scales.
    const std::size_t expectedScales = (input.elementCount + input.blockSize - 1) / input.blockSize;
    if (input.scales.size() != expectedScales) {
        error = "scale count mismatch";
        return false;
    }
    // Unpack quantized signed integers.
    std::vector<int32_t> quantized;
    if (!unpackSigned(input.data, input.elementCount, bits, quantized)) {
        error = "packed payload size mismatch";
        return false;
    }
    // Dequantize signed integers to floating-point values.
    for (std::size_t i = 0; i < input.elementCount; ++i) {
        const float scale = input.scales[i / input.blockSize];
        if (!std::isfinite(scale) || scale <= 0.0f) { error = "invalid scale"; return false; }
        values[i] = static_cast<float>(quantized[i]) * scale;
    }
    return true;
}


// ============================================================
// Encode every tensor in an activation bundle.
// ============================================================
bool ActivationCodec::encodeBundle(const std::vector<TensorPacket>& input, const CodecSpec& spec,
                                   std::vector<EncodedTensorPacket>& output, CodecProfile& profile,
                                   std::string* error) {
    profile = CodecProfile();
    profile.enabled = spec.enabled;
    profile.requested = spec.kind;
    profile.blockSize = spec.blockSize ? spec.blockSize : defaultBlockSize(spec.kind);
    profile.startSteadyNs = steadyNowNs();
    output.clear();
    output.reserve(input.size());
    for (const TensorPacket& packet : input) {
        profile.rawTensorBytes += packet.data.size();
        CodecSpec effective = spec;
        EncodedTensorPacket encoded;
        bool nonFinite = false;
        std::string tensorError;
        // Encode the tensor.
        if (!encodeTensor(packet, effective, encoded, nonFinite, tensorError)) {
            const bool failedForNonFinite = nonFinite;
            if (!spec.allowRawFallback) {
                if (error) *error = tensorError;
                return false;
            }
            effective.kind = CodecKind::RAW32;
            effective.blockSize = 0;
            // Encode the RAW32 fallback tensor.
            if (!encodeTensor(packet, effective, encoded, nonFinite, tensorError)) {
                if (error) *error = tensorError;
                return false;
            }
            encoded.flags |= CODEC_FLAG_RAW_FALLBACK;
            if (failedForNonFinite) encoded.flags |= CODEC_FLAG_NON_FINITE_FALLBACK;
            ++profile.fallbackCount;
            if (failedForNonFinite) ++profile.nonFiniteFallbackCount;
        }
        // Fall back to RAW32 when compression expands the payload and fallback is enabled.
        const size_t encodedCost = encoded.data.size() + encoded.scales.size() * sizeof(float);
        if (encoded.codec != CodecKind::RAW32 && encodedCost >= packet.data.size() &&
            spec.allowRawFallback) {
            effective.kind = CodecKind::RAW32;
            effective.blockSize = 0;
            if (!encodeTensor(packet, effective, encoded, nonFinite, tensorError)) {
                if (error) *error = tensorError;
                return false;
            }
            encoded.flags |= CODEC_FLAG_RAW_FALLBACK | CODEC_FLAG_NO_GAIN_FALLBACK;
            ++profile.fallbackCount;
            ++profile.noGainFallbackCount;
        }
        profile.encodedTensorBytes += encoded.data.size() + encoded.scales.size() * sizeof(float);
        output.push_back(std::move(encoded));
    }
    profile.tensorCount = static_cast<uint32_t>(output.size());
    profile.tensorBytesSaved = profile.rawTensorBytes > profile.encodedTensorBytes
        ? profile.rawTensorBytes - profile.encodedTensorBytes : 0;
    profile.endSteadyNs = steadyNowNs();
    profile.elapsedMs = static_cast<double>(profile.endSteadyNs - profile.startSteadyNs) / 1.0e6;
    if (spec.measureReconstruction) {
        // Measure reconstruction accuracy for the encoded bundle.
        // Accuracy audit is deliberately outside codec elapsedMs. It is not
        // part of the production encoder and must not contaminate its timing.
        for (size_t i = 0; i < input.size(); ++i) {
            TensorPacket reconstructed;
            std::string decodeError;
            // Decode the encoded tensor packet.
            if (!decodeTensor(output[i], reconstructed, decodeError)) {
                if (error) *error = "self-check decode failed: " + decodeError;
                return false;
            }
            // Compare original and decoded tensors and accumulate reconstruction metrics.
            mergeAccuracy(profile.reconstruction,
                          comparePointers(reinterpret_cast<const float*>(input[i].data.data()),
                                          reinterpret_cast<const float*>(reconstructed.data.data()),
                                          input[i].data.size() / sizeof(float)));
        }
    }
    return !output.empty();
}


// ============================================================
// Decode an activation bundle.
// ============================================================
bool ActivationCodec::decodeBundle(const std::vector<EncodedTensorPacket>& input,
                                   std::vector<TensorPacket>& output, CodecProfile& profile,
                                   std::string* error) {
    // Initialize decoder telemetry.                                
    profile = CodecProfile();
    profile.enabled = true;
    if (!input.empty()) {
        profile.requested = input.front().codec;
        profile.blockSize = input.front().blockSize;
    }
    profile.startSteadyNs = steadyNowNs();
    output.clear();
    output.reserve(input.size());
    // Decode each packet in the bundle.
    for (const EncodedTensorPacket& encoded : input) {
        TensorPacket packet;
        std::string tensorError;
        if (!decodeTensor(encoded, packet, tensorError)) {
            if (error) *error = tensorError;
            output.clear();
            return false;
        }
        profile.rawTensorBytes += encoded.rawBytes;
        profile.encodedTensorBytes += encoded.data.size() + encoded.scales.size() * sizeof(float);
        if (encoded.flags & CODEC_FLAG_RAW_FALLBACK) ++profile.fallbackCount;
        if (encoded.flags & CODEC_FLAG_NON_FINITE_FALLBACK) ++profile.nonFiniteFallbackCount;
        if (encoded.flags & CODEC_FLAG_NO_GAIN_FALLBACK) ++profile.noGainFallbackCount;
        output.push_back(std::move(packet));
    }
    profile.tensorCount = static_cast<uint32_t>(output.size());
    profile.tensorBytesSaved = profile.rawTensorBytes > profile.encodedTensorBytes
        ? profile.rawTensorBytes - profile.encodedTensorBytes : 0;
    profile.endSteadyNs = steadyNowNs();
    profile.elapsedMs = static_cast<double>(profile.endSteadyNs - profile.startSteadyNs) / 1.0e6;
    return !output.empty();
}


// ============================================================
// Compare reconstruction accuracy between two bundles.
// ============================================================
AccuracyMetrics ActivationCodec::compare(const std::vector<float>& reference,
                                         const std::vector<float>& actual) {
    if (reference.size() != actual.size()) return AccuracyMetrics();
    // Compare the floating-point payloads and return reconstruction metrics.
    return comparePointers(reference.data(), actual.data(), reference.size());
}
