#include "TensorSerializer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool samePacket(const TensorPacket& lhs, const TensorPacket& rhs) {
    return lhs.tensorIndex == rhs.tensorIndex && lhs.name == rhs.name &&
           lhs.shape == rhs.shape && lhs.dataType == rhs.dataType &&
           lhs.layout == rhs.layout && lhs.data == rhs.data;
}

TensorPacket makePacket(int index, const std::string& name,
                        const std::vector<int>& shape, int bytes) {
    TensorPacket packet;
    packet.tensorIndex = index;
    packet.name = name;
    packet.shape = shape;
    packet.layout = 1;
    packet.data.resize(bytes);
    for (int i = 0; i < bytes; ++i) packet.data[i] = static_cast<uint8_t>((index + i) & 0xff);
    return packet;
}

}  // namespace

int main() {
    std::vector<TensorPacket> input;
    input.push_back(makePacket(7, "residual", {1, 3, 2, 2}, 64));  // includes NC4HW4 padding
    input.push_back(makePacket(19, "dense_skip", {1, 8, 2, 2}, 128));

    CutBundleMetadata inputMetadata;
    inputMetadata.splitAfterOp = 42;
    inputMetadata.modelFingerprint = UINT64_C(0x123456789abcdef0);
    auto bytes = TensorSerializer::serializeBundle(input, &inputMetadata);
    std::vector<TensorPacket> output;
    CutBundleMetadata outputMetadata;
    std::string error;
    if (!TensorSerializer::deserializeBundle(bytes, output, &error, &outputMetadata)) {
        std::cerr << "round-trip parse failed: " << error << std::endl;
        return 1;
    }
    if (outputMetadata.splitAfterOp != inputMetadata.splitAfterOp ||
        outputMetadata.modelFingerprint != inputMetadata.modelFingerprint) {
        std::cerr << "bundle metadata changed" << std::endl;
        return 6;
    }
    if (output.size() != input.size()) {
        std::cerr << "tensor count changed" << std::endl;
        return 2;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        if (!samePacket(input[i], output[i])) {
            std::cerr << "packet changed at " << i << std::endl;
            return 3;
        }
    }

    auto truncated = bytes;
    truncated.pop_back();
    if (TensorSerializer::deserializeBundle(truncated, output, &error)) {
        std::cerr << "truncated bundle was accepted" << std::endl;
        return 4;
    }
    auto trailing = bytes;
    trailing.push_back(0);
    if (TensorSerializer::deserializeBundle(trailing, output, &error)) {
        std::cerr << "trailing bytes were accepted" << std::endl;
        return 5;
    }
    std::cout << "cut-set bundle protocol: PASS (" << input.size()
              << " tensors, " << bytes.size() << " bytes)" << std::endl;
    return 0;
}
