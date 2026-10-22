# Build Guide

## Host-only build

Requirements:

- CMake 3.18 or later
- A C++17 compiler
- A thread library supported by CMake

```bash
cmake -S . -B build -DEVOSPLIT_WITH_MNN=OFF \
  -DEVOSPLIT_BUILD_TESTS=ON -DEVOSPLIT_BUILD_DEMO=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

To smoke-test the installed CMake package itself:

```bash
cmake --install build --prefix install
cmake -S tests/package_consumer -B consumer-build \
  -DEvoSplit_DIR=/absolute/path/to/install/lib/cmake/EvoSplit
cmake --build consumer-build --config Release
```

When `EVOSPLIT_WITH_MNN=ON`, CMake also builds `evosplit_mnn_validation`.
It accepts an MNN model, input tensor name, split tensor name, and physical
operator index; see `--help`-style usage printed when invoked without arguments.

This configuration validates the codec, optimization model, profile-free
controller, online residual logic, and version-aware recovery without a model
or device.

## MNN build

```bash
cmake -S . -B build-mnn \
  -DEVOSPLIT_WITH_MNN=ON \
  -DMNN_INCLUDE_DIR=/path/to/MNN/include \
  -DMNN_LIBRARY=/path/to/libMNN.so \
  -DMNN_SCHEMA_INCLUDE_DIR=/path/to/MNN/schema/current \
  -DFLATBUFFERS_INCLUDE_DIR=/path/to/flatbuffers/include
cmake --build build-mnn --config Release
```

MNN is not downloaded or built by this repository. The headers and library
must come from the same compatible MNN build.

## Android build

Use the NDK CMake toolchain and pass the same ABI, API level, STL, MNN headers,
and MNN library used by the host application:

```bash
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-23 \
  -DANDROID_STL=c++_shared \
  -DEVOSPLIT_WITH_MNN=ON \
  -DEVOSPLIT_BUILD_TESTS=OFF \
  -DMNN_INCLUDE_DIR=/path/to/MNN/include \
  -DMNN_LIBRARY=/path/to/android/libMNN.so
cmake --build build-android --config Release
```

Never mix Windows, Linux, and Android libraries or libraries built with
different ABIs, API levels, NDK versions, or C++ runtimes.

## Installation

```bash
cmake --install build --prefix /path/to/evosplit-install
```

The install tree contains public headers, `EvoSplit::core`, optional
`EvoSplit::mnn_runtime`, and CMake package configuration files.

