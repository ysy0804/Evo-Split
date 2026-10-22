#ifndef EDGE_AI_CUT_SET_PLANNER_H
#define EDGE_AI_CUT_SET_PLANNER_H

#include <string>
#include <vector>
#include <cstdint>

struct CutConsumerBinding {
    int opIndex{-1};
    std::string opName;
    int inputSlot{-1};
};

struct CutTensorDescriptor {
    int tensorIndex{-1};
    std::string tensorName;
    int producerOpIndex{-1};
    std::string producerOpName;
    int producerOutputSlot{-1};
    bool modelInput{false};
    std::vector<CutConsumerBinding> consumers;
};

struct CutSetPlan {
    int splitAfterOp{-1};
    int totalOps{0};
    uint64_t modelFingerprint{0};
    std::vector<CutTensorDescriptor> tensors;

    bool valid() const { return splitAfterOp >= 0 && !tensors.empty(); }
    std::vector<std::string> tensorNames() const;
    std::string cacheKey() const;
};

class CutSetPlanner {
public:
    static bool fingerprintModel(const std::string& modelPath,
                                 uint64_t& fingerprint,
                                 std::string& error);
    static bool fingerprintStructure(const std::string& modelPath,
                                     std::string& fingerprint,
                                     std::string& error);
    static bool buildFromModel(const std::string& modelPath,
                               int splitAfterOp,
                               CutSetPlan& plan,
                               std::string& error);
};

#endif
