# Integration Guide

## CMake consumer

Configure the consumer with `EvoSplit_DIR` pointing to the installed CMake
package directory:

```cmake
find_package(EvoSplit CONFIG REQUIRED)

add_executable(my_edge_app main.cpp)
target_link_libraries(my_edge_app PRIVATE EvoSplit::core)
```

For an MNN-enabled consumer:

```cmake
target_link_libraries(my_edge_app PRIVATE EvoSplit::mnn_runtime)
```

Example configuration:

```bash
cmake -S consumer -B consumer-build \
  -DEvoSplit_DIR=/path/to/install/lib/cmake/EvoSplit
```

## Integration boundary

The host application should provide:

- model and dataset paths;
- device, memory, thermal, and network state samples;
- an MNN library compatible with the target platform;
- request lifecycle and application-specific accuracy validation;
- storage for decision and update audit records.

Evo-Split provides:

- semantic cut-set planning and tensor bundle handling;
- activation codecs and codec telemetry;
- profile-free action selection and online observations;
- version manifest comparison and minimal-impact state migration;
- optional MNN capture, injection, TCP client, and cloud daemon components.

## Version safety

The edge and cloud must agree on model content hash, semantic boundary manifest,
tensor packet protocol version, codec configuration, and consumer input slots.
Reject or fall back to a safe local action when these checks fail.

## ABI note

Evo-Split currently exposes a research-stage C++ API. Rebuild the framework and
consumer together after changing the compiler, standard library, target ABI,
MNN build, or public headers.

