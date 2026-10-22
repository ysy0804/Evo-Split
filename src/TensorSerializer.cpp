#include "TensorSerializer.h"
#include "ActivationCodec.h"

#include <cstring>
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef HAS_MNN
#include <MNN/Tensor.hpp>
#endif

namespace {

void appendU32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 24));
}

void appendU64(std::vector<uint8_t>& out, uint64_t value) {
    appendU32(out, static_cast<uint32_t>(value));
    appendU32(out, static_cast<uint32_t>(value >> 32));
}

bool readU32Checked(const std::vector<uint8_t>& data, size_t& offset, uint32_t& value) {
    if (offset > data.size() || data.size() - offset < 4) return false;
    value = static_cast<uint32_t>(data[offset]) |
            (static_cast<uint32_t>(data[offset + 1]) << 8) |
            (static_cast<uint32_t>(data[offset + 2]) << 16) |
            (static_cast<uint32_t>(data[offset + 3]) << 24);
    offset += 4;
    return true;
}

bool readU64Checked(const std::vector<uint8_t>& data, size_t& offset, uint64_t& value) {
    uint32_t low = 0, high = 0;
    if (!readU32Checked(data, offset, low) || !readU32Checked(data, offset, high)) return false;
    value = static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
    return true;
}

void setBundleError(std::string* error, const std::string& value) {
    if (error != nullptr) *error = value;
}

uint32_t crc32(const uint8_t* data, size_t size) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

void appendFloat(std::vector<uint8_t>& out, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(out, bits);
}

bool readFloatChecked(const std::vector<uint8_t>& data, size_t& offset, float& value) {
    uint32_t bits = 0;
    if (!readU32Checked(data, offset, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

}  // namespace

std::vector<uint8_t> TensorSerializer::serializeBundle(
    const std::vector<TensorPacket>& tensors,
    const CutBundleMetadata* metadata) {
    std::vector<uint8_t> result;
    appendU32(result, BUNDLE_MAGIC);
    const bool hasMetadata = metadata != nullptr && metadata->valid();
    appendU32(result, hasMetadata ? BUNDLE_VERSION : 1u);
    appendU32(result, static_cast<uint32_t>(tensors.size()));
    if (hasMetadata) {
        appendU32(result, static_cast<uint32_t>(metadata->splitAfterOp));
        appendU64(result, metadata->modelFingerprint);
    }
    for (const auto& tensor : tensors) {
        appendU32(result, static_cast<uint32_t>(tensor.tensorIndex));
        appendU32(result, static_cast<uint32_t>(tensor.name.size()));
        result.insert(result.end(), tensor.name.begin(), tensor.name.end());
        appendU32(result, static_cast<uint32_t>(tensor.shape.size()));
        for (int dimension : tensor.shape) appendU32(result, static_cast<uint32_t>(dimension));
        appendU32(result, tensor.dataType);
        appendU32(result, tensor.layout);
        appendU32(result, static_cast<uint32_t>(tensor.data.size()));
        result.insert(result.end(), tensor.data.begin(), tensor.data.end());
    }
    return result;
}

bool TensorSerializer::deserializeBundle(const std::vector<uint8_t>& data,
                                         std::vector<TensorPacket>& tensors,
                                         std::string* error,
                                         CutBundleMetadata* metadata) {
    tensors.clear();
    if (metadata != nullptr) *metadata = CutBundleMetadata();
    size_t offset = 0;
    uint32_t magic = 0, version = 0, count = 0;
    if (!readU32Checked(data, offset, magic) || magic != BUNDLE_MAGIC) {
        setBundleError(error, "bad bundle magic");
        return false;
    }
    if (!readU32Checked(data, offset, version)) {
        setBundleError(error, "unsupported bundle version");
        return false;
    }
    if (version == CODEC_BUNDLE_VERSION) {
        std::vector<EncodedTensorPacket> encoded;
        if (!deserializeEncodedBundle(data, encoded, error, metadata)) return false;
        CodecProfile ignored;
        return ActivationCodec::decodeBundle(encoded, tensors, ignored, error);
    }
    if (version != 1 && version != BUNDLE_VERSION) {
        setBundleError(error, "unsupported bundle version");
        return false;
    }
    if (!readU32Checked(data, offset, count) || count == 0 || count > 1024) {
        setBundleError(error, "invalid tensor count");
        return false;
    }
    if (version >= 2) {
        uint32_t splitAfterOp = 0;
        uint64_t fingerprint = 0;
        if (!readU32Checked(data, offset, splitAfterOp) ||
            splitAfterOp > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
            !readU64Checked(data, offset, fingerprint) || fingerprint == 0) {
            setBundleError(error, "invalid bundle metadata");
            return false;
        }
        if (metadata != nullptr) {
            metadata->splitAfterOp = static_cast<int32_t>(splitAfterOp);
            metadata->modelFingerprint = fingerprint;
        }
    }
    tensors.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        TensorPacket packet;
        uint32_t tensorIndex = 0, nameSize = 0, dimensions = 0, dataSize = 0;
        if (!readU32Checked(data, offset, tensorIndex) ||
            !readU32Checked(data, offset, nameSize) || nameSize == 0 || nameSize > 4096 ||
            offset > data.size() || data.size() - offset < nameSize) {
            setBundleError(error, "invalid tensor identity");
            return false;
        }
        packet.tensorIndex = static_cast<int32_t>(tensorIndex);
        packet.name.assign(reinterpret_cast<const char*>(data.data() + offset), nameSize);
        offset += nameSize;
        if (!readU32Checked(data, offset, dimensions) || dimensions == 0 || dimensions > 8) {
            setBundleError(error, "invalid tensor rank");
            return false;
        }
        packet.shape.reserve(dimensions);
        for (uint32_t d = 0; d < dimensions; ++d) {
            uint32_t dimension = 0;
            if (!readU32Checked(data, offset, dimension) || dimension == 0 ||
                dimension > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                setBundleError(error, "invalid tensor shape");
                return false;
            }
            packet.shape.push_back(static_cast<int>(dimension));
        }
        if (!readU32Checked(data, offset, packet.dataType) || packet.dataType != 0 ||
            !readU32Checked(data, offset, packet.layout) || packet.layout > 1 ||
            !readU32Checked(data, offset, dataSize) || dataSize == 0 ||
            offset > data.size() || data.size() - offset < dataSize) {
            setBundleError(error, "invalid tensor payload");
            return false;
        }
        packet.data.assign(data.begin() + offset, data.begin() + offset + dataSize);
        offset += dataSize;
        tensors.push_back(std::move(packet));
    }
    if (offset != data.size()) {
        setBundleError(error, "trailing bundle bytes");
        tensors.clear();
        return false;
    }
    return true;
}

bool TensorSerializer::inspectBundleVersion(const std::vector<uint8_t>& data, uint32_t& version) {
    size_t offset = 0;
    uint32_t magic = 0;
    return readU32Checked(data, offset, magic) && magic == BUNDLE_MAGIC &&
           readU32Checked(data, offset, version);
}

std::vector<uint8_t> TensorSerializer::serializeEncodedBundle(
    const std::vector<EncodedTensorPacket>& tensors,
    const CutBundleMetadata& metadata) {
    std::vector<uint8_t> result;
    if (!metadata.valid() || tensors.empty() || tensors.size() > 1024) return result;
    appendU32(result, BUNDLE_MAGIC);
    appendU32(result, CODEC_BUNDLE_VERSION);
    appendU32(result, static_cast<uint32_t>(tensors.size()));
    appendU32(result, static_cast<uint32_t>(metadata.splitAfterOp));
    appendU64(result, metadata.modelFingerprint);
    for (const auto& tensor : tensors) {
        if (tensor.tensorIndex < 0 || tensor.name.empty() || tensor.name.size() > 4096 ||
            tensor.shape.empty() || tensor.shape.size() > 8 || tensor.rawBytes == 0 ||
            tensor.elementCount == 0 || tensor.data.empty() ||
            tensor.scales.size() > tensor.elementCount ||
            tensor.data.size() > std::numeric_limits<uint32_t>::max()) return {};
        appendU32(result, static_cast<uint32_t>(tensor.tensorIndex));
        appendU32(result, static_cast<uint32_t>(tensor.name.size()));
        result.insert(result.end(), tensor.name.begin(), tensor.name.end());
        appendU32(result, static_cast<uint32_t>(tensor.shape.size()));
        for (int dimension : tensor.shape) {
            if (dimension <= 0) return {};
            appendU32(result, static_cast<uint32_t>(dimension));
        }
        appendU32(result, tensor.dataType);
        appendU32(result, tensor.layout);
        appendU32(result, tensor.rawBytes);
        appendU32(result, tensor.elementCount);
        appendU32(result, static_cast<uint32_t>(tensor.codec));
        appendU32(result, tensor.blockSize);
        appendU32(result, tensor.flags);
        appendU32(result, static_cast<uint32_t>(tensor.scales.size()));
        for (float scale : tensor.scales) appendFloat(result, scale);
        appendU32(result, static_cast<uint32_t>(tensor.data.size()));
        appendU32(result, crc32(tensor.data.data(), tensor.data.size()));
        result.insert(result.end(), tensor.data.begin(), tensor.data.end());
    }
    return result;
}

bool TensorSerializer::deserializeEncodedBundle(const std::vector<uint8_t>& data,
                                                std::vector<EncodedTensorPacket>& tensors,
                                                std::string* error,
                                                CutBundleMetadata* metadata) {
    tensors.clear();
    if (metadata) *metadata = CutBundleMetadata();
    size_t offset = 0;
    uint32_t magic = 0, version = 0, count = 0, split = 0;
    uint64_t fingerprint = 0;
    if (!readU32Checked(data, offset, magic) || magic != BUNDLE_MAGIC ||
        !readU32Checked(data, offset, version) || version != CODEC_BUNDLE_VERSION ||
        !readU32Checked(data, offset, count) || count == 0 || count > 1024 ||
        !readU32Checked(data, offset, split) ||
        split > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        !readU64Checked(data, offset, fingerprint) || fingerprint == 0) {
        setBundleError(error, "invalid MCUT v3 header");
        return false;
    }
    if (metadata) {
        metadata->splitAfterOp = static_cast<int32_t>(split);
        metadata->modelFingerprint = fingerprint;
    }
    tensors.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        EncodedTensorPacket packet;
        uint32_t tensorIndex = 0, nameSize = 0, dimensions = 0, codec = 0;
        uint32_t scaleCount = 0, payloadSize = 0, checksum = 0;
        if (!readU32Checked(data, offset, tensorIndex) ||
            tensorIndex > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
            !readU32Checked(data, offset, nameSize) || nameSize == 0 || nameSize > 4096 ||
            offset > data.size() || data.size() - offset < nameSize) {
            setBundleError(error, "invalid v3 tensor identity");
            return false;
        }
        packet.tensorIndex = static_cast<int32_t>(tensorIndex);
        packet.name.assign(reinterpret_cast<const char*>(data.data() + offset), nameSize);
        offset += nameSize;
        if (!readU32Checked(data, offset, dimensions) || dimensions == 0 || dimensions > 8) {
            setBundleError(error, "invalid v3 tensor rank");
            return false;
        }
        std::size_t shapeElements = 1;
        for (uint32_t d = 0; d < dimensions; ++d) {
            uint32_t dimension = 0;
            if (!readU32Checked(data, offset, dimension) || dimension == 0 ||
                dimension > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                shapeElements > std::numeric_limits<std::size_t>::max() / dimension) {
                setBundleError(error, "invalid v3 tensor shape");
                return false;
            }
            shapeElements *= dimension;
            packet.shape.push_back(static_cast<int>(dimension));
        }
        if (!readU32Checked(data, offset, packet.dataType) || packet.dataType != 0 ||
            !readU32Checked(data, offset, packet.layout) || packet.layout > 1 ||
            !readU32Checked(data, offset, packet.rawBytes) || packet.rawBytes == 0 ||
            !readU32Checked(data, offset, packet.elementCount) || packet.elementCount == 0 ||
            packet.elementCount != shapeElements ||
            static_cast<uint64_t>(packet.rawBytes) !=
                static_cast<uint64_t>(packet.elementCount) * sizeof(float) ||
            !readU32Checked(data, offset, codec) || codec > static_cast<uint32_t>(CodecKind::LZ4) ||
            !readU32Checked(data, offset, packet.blockSize) ||
            !readU32Checked(data, offset, packet.flags) ||
            (packet.flags & ~(CODEC_FLAG_RAW_FALLBACK | CODEC_FLAG_NON_FINITE_FALLBACK |
                              CODEC_FLAG_NO_GAIN_FALLBACK)) != 0 ||
            !readU32Checked(data, offset, scaleCount) || scaleCount > packet.elementCount ||
            data.size() - offset < static_cast<size_t>(scaleCount) * sizeof(float)) {
            setBundleError(error, "invalid v3 tensor metadata");
            return false;
        }
        packet.codec = static_cast<CodecKind>(codec);
        packet.scales.reserve(scaleCount);
        for (uint32_t scale = 0; scale < scaleCount; ++scale) {
            float value = 0.0f;
            if (!readFloatChecked(data, offset, value) || !std::isfinite(value) || value <= 0.0f) {
                setBundleError(error, "invalid v3 quantization scale");
                return false;
            }
            packet.scales.push_back(value);
        }
        if (!readU32Checked(data, offset, payloadSize) || payloadSize == 0 ||
            !readU32Checked(data, offset, checksum) || offset > data.size() ||
            data.size() - offset < payloadSize) {
            setBundleError(error, "invalid v3 payload bounds");
            return false;
        }
        if (crc32(data.data() + offset, payloadSize) != checksum) {
            setBundleError(error, "v3 payload checksum mismatch");
            return false;
        }
        packet.data.assign(data.begin() + offset, data.begin() + offset + payloadSize);
        offset += payloadSize;
        tensors.push_back(std::move(packet));
    }
    if (offset != data.size()) {
        setBundleError(error, "trailing v3 bundle bytes");
        tensors.clear();
        return false;
    }
    return true;
}

// ============================================================
//  serialize -- convert an MNN tensor to bytes
// ============================================================

#ifdef HAS_MNN
bool TensorSerializer::tensorToPacket(const MNN::Tensor* tensor,
                                      int tensorIndex,
                                      const std::string& name,
                                      TensorPacket& packet) {
    if (tensor == nullptr || tensor->host<uint8_t>() == nullptr || tensor->size() <= 0 || name.empty()) return false;
    packet = TensorPacket();
    packet.tensorIndex = tensorIndex;
    packet.name = name;
    packet.shape = tensor->shape();
    packet.dataType = 0;
    packet.layout = tensor->getDimensionType() == MNN::Tensor::CAFFE_C4 ? 1u : 0u;
    const auto* begin = tensor->host<uint8_t>();
    packet.data.assign(begin, begin + tensor->size());
    return true;
}

std::vector<uint8_t> TensorSerializer::serialize(const MNN::Tensor* tensor) {
    std::vector<uint8_t> result;

    if (tensor == nullptr) return result;

    int dims = tensor->dimensions();
    std::vector<int> shape;
    for (int d = 0; d < dims; d++) {
        shape.push_back(tensor->length(d));
    }

    int elements = tensor->elementSize();

    // Compute the total serialized size.
    size_t headerSize = 4 + 4 + 4 * dims + 4 + 4;  // magic + dims + shape + elements + type
    size_t dataSize = elements * sizeof(float);
    result.resize(headerSize + dataSize);

    size_t offset = 0;
    auto writeU32 = [&](uint32_t v) {
        std::memcpy(result.data() + offset, &v, 4);
        offset += 4;
    };

    // Header
    writeU32(MAGIC);
    writeU32(static_cast<uint32_t>(dims));
    for (int d : shape) writeU32(static_cast<uint32_t>(d));
    writeU32(static_cast<uint32_t>(elements));
    writeU32(0);  // data type: 0 = float32

    // Data (host data)
    const float* host = tensor->host<float>();
    if (host != nullptr && elements > 0) {
        std::memcpy(result.data() + offset, host, dataSize);
    }

    return result;
}

// ============================================================
//  deserialize -- convert bytes to an MNN tensor
// ============================================================

MNN::Tensor* TensorSerializer::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < 16) return nullptr;

    size_t offset = 0;
    auto readU32 = [&]() -> uint32_t {
        if (offset + 4 > data.size()) return 0;
        uint32_t v;
        std::memcpy(&v, data.data() + offset, 4);
        offset += 4;
        return v;
    };

    uint32_t magic = readU32();
    if (magic != MAGIC) {
        std::cerr << "[TensorSerializer] ❌ invalid magic: 0x"
                  << std::hex << magic << std::dec << std::endl;
        return nullptr;
    }

    uint32_t dims = readU32();
    if (dims > 8) {
        std::cerr << "[TensorSerializer] ❌ invalid dimension count: " << dims << std::endl;
        return nullptr;
    }

    std::vector<int> shape;
    for (uint32_t d = 0; d < dims; d++) {
        shape.push_back(static_cast<int>(readU32()));
    }

    uint32_t elements = readU32();
    uint32_t dataType = readU32();
    if (dataType != 1) {
        std::cerr << "[TensorSerializer] unsupported legacy data type: "
                  << dataType << std::endl;
        return nullptr;
    }

    // Validate payload length.
    size_t expectedDataSize = elements * sizeof(float);
    if (offset + expectedDataSize > data.size()) {
        std::cerr << "[TensorSerializer] ❌ payload is too short; need "
                  << expectedDataSize << " have " << (data.size() - offset)
                  << std::endl;
        return nullptr;
    }

    // Create a CPU tensor with the parsed shape and FP32 payload.
    MNN::Tensor* tensor = MNN::Tensor::create(shape, halide_type_of<float>(), NULL, MNN::Tensor::CAFFE);
    if (tensor == nullptr) return nullptr;

    float* host = tensor->host<float>();
    if (host != nullptr && elements > 0) {
        std::memcpy(host, data.data() + offset, expectedDataSize);
    }

    return tensor;
}

// ============================================================
//  parseHeader -- parse tensor metadata
// ============================================================

bool TensorSerializer::parseHeader(const std::vector<uint8_t>& data,
                                   std::vector<int>& shape,
                                   size_t& elements) {
    if (data.size() < 16) return false;

    size_t offset = 0;
    auto readU32 = [&]() -> uint32_t {
        if (offset + 4 > data.size()) return 0;
        uint32_t v;
        std::memcpy(&v, data.data() + offset, 4);
        offset += 4;
        return v;
    };

    if (readU32() != MAGIC) return false;
    uint32_t dims = readU32();
    if (dims > 8) return false;

    shape.clear();
    for (uint32_t d = 0; d < dims; d++) {
        shape.push_back(static_cast<int>(readU32()));
    }
    elements = readU32();
    return true;
}
#endif

// ============================================================
//  printFormat -- describe the binary format
// ============================================================

void TensorSerializer::printFormat() {
    std::cout << "[TensorSerializer] Binary format:" << std::endl;
    std::cout << "  [4B] magic = 0x4D4E544C" << std::endl;
    std::cout << "  [4B] dimension count N" << std::endl;
    std::cout << "  [4B*N] dimension sizes" << std::endl;
    std::cout << "  [4B] element count" << std::endl;
    std::cout << "  [4B] data type (0=float32)" << std::endl;
    std::cout << "  [4B*N] raw float32 data" << std::endl;
}
