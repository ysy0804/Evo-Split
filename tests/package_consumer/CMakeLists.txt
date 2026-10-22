cmake_minimum_required(VERSION 3.18)
project(EvoSplitPackageConsumer LANGUAGES CXX)

find_package(EvoSplit CONFIG REQUIRED)

add_executable(package_consumer main.cpp)
target_compile_features(package_consumer PRIVATE cxx_std_17)
target_link_libraries(package_consumer PRIVATE EvoSplit::core)
