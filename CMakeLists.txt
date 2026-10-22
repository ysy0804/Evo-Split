cmake_minimum_required(VERSION 3.18)
project(EvoSplit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
endif()

option(EVOSPLIT_BUILD_TESTS "Build host-independent tests" ON)
option(EVOSPLIT_BUILD_DEMO "Build the minimal demonstration" ON)
option(EVOSPLIT_WITH_MNN "Build the MNN edge/cloud data plane" OFF)

find_package(Threads REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(evosplit_core STATIC
    src/TensorSerializer.cpp
    src/ActivationCodec.cpp
    src/AdaptiveController.cpp
    src/AdaptiveRuntime.cpp
    src/ProfileFreeController.cpp
    src/VersionedUpdateManager.cpp)
target_include_directories(evosplit_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(evosplit_core PUBLIC Threads::Threads)
set_target_properties(evosplit_core PROPERTIES EXPORT_NAME core)
add_library(EvoSplit::core ALIAS evosplit_core)

if(EVOSPLIT_BUILD_TESTS)
    enable_testing()
    function(evosplit_test name source)
        add_executable(${name} tests/${source})
        target_link_libraries(${name} PRIVATE evosplit_core)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    evosplit_test(test_cutset_protocol test_cutset_protocol.cpp)
    evosplit_test(test_activation_codec test_activation_codec.cpp)
    evosplit_test(test_adaptive_controller test_adaptive_controller.cpp)
    evosplit_test(test_adaptive_runtime test_adaptive_runtime.cpp)
    evosplit_test(test_profile_free_controller test_profile_free_controller.cpp)
    evosplit_test(test_versioned_update test_versioned_update.cpp)
endif()

if(EVOSPLIT_BUILD_DEMO)
    add_executable(evosplit_demo examples/minimal_demo/demo_main.cpp)
    target_link_libraries(evosplit_demo PRIVATE evosplit_core)
endif()

if(EVOSPLIT_WITH_MNN)
    if(NOT MNN_INCLUDE_DIR OR NOT MNN_LIBRARY)
        message(FATAL_ERROR "EVOSPLIT_WITH_MNN=ON requires MNN_INCLUDE_DIR and MNN_LIBRARY")
    endif()
    if(NOT EXISTS "${MNN_INCLUDE_DIR}" OR NOT EXISTS "${MNN_LIBRARY}")
        message(FATAL_ERROR "MNN_INCLUDE_DIR or MNN_LIBRARY does not exist")
    endif()

    add_library(EvoSplitMNN UNKNOWN IMPORTED GLOBAL)
    set_target_properties(EvoSplitMNN PROPERTIES
        IMPORTED_LOCATION "${MNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MNN_INCLUDE_DIR}")
    target_compile_definitions(evosplit_core PUBLIC HAS_MNN)
    target_link_libraries(evosplit_core PUBLIC EvoSplitMNN)

    add_library(evosplit_mnn_runtime STATIC
        runtime/src/CutSetPlanner.cpp
        runtime/src/SplitInferenceEngine.cpp
        runtime/src/CloudInjectionEngine.cpp
        runtime/src/CloudInferenceEngine.cpp
        runtime/src/CloudClient.cpp
        runtime/src/CloudDaemon.cpp)
    target_compile_definitions(evosplit_mnn_runtime PUBLIC HAS_MNN)
    target_include_directories(evosplit_mnn_runtime PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    if(MNN_SCHEMA_INCLUDE_DIR)
        target_include_directories(evosplit_mnn_runtime PRIVATE ${MNN_SCHEMA_INCLUDE_DIR})
    endif()
    if(FLATBUFFERS_INCLUDE_DIR)
        target_include_directories(evosplit_mnn_runtime PRIVATE ${FLATBUFFERS_INCLUDE_DIR})
    endif()
    target_link_libraries(evosplit_mnn_runtime PUBLIC evosplit_core EvoSplitMNN)
    set_target_properties(evosplit_mnn_runtime PROPERTIES EXPORT_NAME mnn_runtime)
    add_library(EvoSplit::mnn_runtime ALIAS evosplit_mnn_runtime)
    if(ANDROID)
        target_link_libraries(evosplit_mnn_runtime PUBLIC log dl)
    endif()

    add_executable(evosplit_cloud_daemon runtime/src/cloud_main.cpp)
    target_link_libraries(evosplit_cloud_daemon PRIVATE evosplit_mnn_runtime)
    if(EVOSPLIT_BUILD_TESTS)
        add_executable(evosplit_mnn_validation tests/test_phase1_validation.cpp)
        target_link_libraries(evosplit_mnn_validation PRIVATE evosplit_mnn_runtime)
    endif()
    install(TARGETS evosplit_mnn_runtime EXPORT EvoSplitTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(TARGETS evosplit_cloud_daemon RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY runtime/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/evosplit
        FILES_MATCHING PATTERN "*.h")
endif()

install(TARGETS evosplit_core EXPORT EvoSplitTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/evosplit
    FILES_MATCHING PATTERN "*.h")
configure_package_config_file(
    cmake/EvoSplitConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/EvoSplitConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EvoSplit)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/EvoSplitConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/EvoSplitConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/EvoSplitConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EvoSplit)
install(EXPORT EvoSplitTargets NAMESPACE EvoSplit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EvoSplit)

