#include <evosplit/ProfileFreeController.h>

#include <iostream>

int main() {
    StructuralActionBuilder builder;
    StructuralModelManifest manifest;
    manifest.manifestId = "consumer-smoke";
    manifest.modelHash = "consumer-smoke";
    manifest.modelVersion = "1";
    manifest.inputBytes = 224 * 224 * 3;
    manifest.totalFlops = 1.0e9;
    const auto actions = builder.build(manifest);
    std::cout << "EVOSPLIT_PACKAGE_CONSUMER=PASS actions="
              << actions.actions.size() << '\n';
    return actions.actions.size() == 2 ? 0 : 1;
}
