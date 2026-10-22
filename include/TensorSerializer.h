#ifndef EDGE_AI_TENSOR_SERIALIZER_H
#define EDGE_AI_TENSOR_SERIALIZER_H

#include <vector>
#include <cstdint>
#include <string>

#include "CodecTypes.h"

struct TensorPacket {
    int32_t tensorIndex{-1};
    std::string name;
    std::vector<int> shape;
    uint32_t dataType{0};
    uint32_t layout{1};
    std::vector<uint8_t> data;
};

struct CutBundleMetadata {
    int32_t splitAfterOp{-1};
    uint64_t modelFingerprint{0};

    bool valid() const { return splitAfterOp >= 0 && modelFingerprint != 0; }
};

#ifdef HAS_MNN
#include <MNN/Tensor.hpp>
#endif

/**
 * Tensor serializer.
 *
 * Transfers intermediate tensors for edge-cloud split inference:
 *   MNN::Tensor -> binary byte stream with shape metadata
 *   binary byte stream -> MNN::Tensor
 *
 * Binary format:
 *   [4B]  magic = 0x4D4E544C ("MNTL")
 *   [4B]  dimension count N
 *   [4B*N] dimension sizes
 *   [4B]  element count
 *   [4B]  data type (0=float32, 1=float16, 2=int32, ...)
 *   [N*4B] raw data (float32)
 */
class TensorSerializer {
public:
    // Binary format magic.
    static constexpr uint32_t MAGIC = 0x4D4E544C;  // "MNTL"
    static constexpr uint32_t BUNDLE_MAGIC = 0x4D435554;  // "MCUT"
    static constexpr uint32_t BUNDLE_VERSION = 2;
    static constexpr uint32_t CODEC_BUNDLE_VERSION = 3;

    static std::vector<uint8_t> serializeBundle(
        const std::vector<TensorPacket>& tensors,
        const CutBundleMetadata* metadata = nullptr);
    static bool deserializeBundle(const std::vector<uint8_t>& data,
                                  std::vector<TensorPacket>& tensors,
                                  std::string* error = nullptr,
                                  CutBundleMetadata* metadata = nullptr);

    static std::vector<uint8_t> serializeEncodedBundle(
        const std::vector<EncodedTensorPacket>& tensors,
        const CutBundleMetadata& metadata);
    static bool deserializeEncodedBundle(const std::vector<uint8_t>& data,
                                         std::vector<EncodedTensorPacket>& tensors,
                                         std::string* error = nullptr,
                                         CutBundleMetadata* metadata = nullptr);
    static bool inspectBundleVersion(const std::vector<uint8_t>& data, uint32_t& version);

    /**
     * Serialize an MNN tensor to bytes.
     * @param tensor source tensor with host-readable data
     * @return serialized byte stream
     */
#ifdef HAS_MNN
    static std::vector<uint8_t> serialize(const MNN::Tensor* tensor);

    /**
     * Deserialize bytes into a CPU MNN tensor.
     * @param data byte stream
     * @return new tensor; the caller owns it
     */
    static MNN::Tensor* deserialize(const std::vector<uint8_t>& data);

    static bool tensorToPacket(const MNN::Tensor* tensor,
                               int tensorIndex,
                               const std::string& name,
                               TensorPacket& packet);

    /**
     * Parse a tensor descriptor without copying payload data.
     * @param data byte stream
     * @param shape output tensor shape
     * @param elements output element count
     */
    static bool parseHeader(const std::vector<uint8_t>& data,
                            std::vector<int>& shape,
                            size_t& elements);
#endif

    /**
     * Print the serialization format for debugging.
     */
    static void printFormat();
};

#endif // EDGE_AI_TENSOR_SERIALIZER_H
