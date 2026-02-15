# Tests for OpenGJK SIMD

# Use local GoogleTest from ext/ if available, otherwise fallback to FetchContent
set(GTEST_EXT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../ext/googletest")
if(EXISTS "${GTEST_EXT_PATH}/CMakeLists.txt")
    message(STATUS "Using local GoogleTest from ${GTEST_EXT_PATH}")
    # Force static linking to avoid dylib issues on macOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${GTEST_EXT_PATH} ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
else()
    message(STATUS "Fetching GoogleTest from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Force static linking to avoid dylib issues on macOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# Header-only test helpers
add_library(test_helpers INTERFACE)
target_include_directories(test_helpers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include # For opengjk_simd_compile_config.h
)

# S1D tests
add_executable(test_s1d unittest_s1d.cc)
target_link_libraries(test_s1d PRIVATE opengjk_simd GTest::gtest_main test_helpers)
target_include_directories(test_s1d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(test_s1d)

# S1D cross-validation tests (SIMD vs scalar)
if(TARGET opengjk_scalar_static)
    add_executable(test_s1d_crossval unittest_s1d_crossval.cc)
    target_link_libraries(test_s1d_crossval PRIVATE opengjk_simd opengjk_scalar_static GTest::gtest_main test_helpers)
    target_include_directories(test_s1d_crossval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # Scalar uses USE_32BITS, SIMD uses OPENGJK_SIMD_USE_FLOAT - need USE_32BITS for gkFloat type
    if(USE_32BITS)
        target_compile_definitions(test_s1d_crossval PRIVATE USE_32BITS)
    endif()
    gtest_discover_tests(test_s1d_crossval)
endif()

# S2D tests
add_executable(test_s2d unittest_s2d.cc)
target_link_libraries(test_s2d PRIVATE opengjk_simd GTest::gtest_main test_helpers)
target_include_directories(test_s2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(test_s2d)

# S2D cross-validation tests (SIMD vs scalar)
if(TARGET opengjk_scalar_static)
    add_executable(test_s2d_crossval unittest_s2d_crossval.cc)
    target_link_libraries(test_s2d_crossval PRIVATE opengjk_simd opengjk_scalar_static GTest::gtest_main test_helpers)
    target_include_directories(test_s2d_crossval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # Scalar uses USE_32BITS, SIMD uses OPENGJK_SIMD_USE_FLOAT - need USE_32BITS for gkFloat type
    if(USE_32BITS)
        target_compile_definitions(test_s2d_crossval PRIVATE USE_32BITS)
    endif()
    gtest_discover_tests(test_s2d_crossval)
endif()

# S3D tests
add_executable(test_s3d unittest_s3d.cc)
target_link_libraries(test_s3d PRIVATE opengjk_simd GTest::gtest_main test_helpers)
target_include_directories(test_s3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(test_s3d)

# S3D cross-validation tests (SIMD vs scalar)
if(TARGET opengjk_scalar_static)
    add_executable(test_s3d_crossval unittest_s3d_crossval.cc)
    target_link_libraries(test_s3d_crossval PRIVATE opengjk_simd opengjk_scalar_static GTest::gtest_main test_helpers)
    target_include_directories(test_s3d_crossval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # Scalar uses USE_32BITS, SIMD uses OPENGJK_SIMD_USE_FLOAT - need USE_32BITS for gkFloat type
    if(USE_32BITS)
        target_compile_definitions(test_s3d_crossval PRIVATE USE_32BITS)
    endif()
    gtest_discover_tests(test_s3d_crossval)
endif()

# Full GJK tests
add_executable(test_gjk unittest_gjk.cc)
target_link_libraries(test_gjk PRIVATE opengjk_simd GTest::gtest_main test_helpers)
target_include_directories(test_gjk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
gtest_discover_tests(test_gjk)

# Cross-validation tests (SIMD vs scalar)
if(TARGET opengjk_scalar_static)
    add_executable(test_crossval unittest_crossval.cc)
    target_link_libraries(test_crossval PRIVATE opengjk_simd opengjk_scalar_static GTest::gtest_main test_helpers)
    target_include_directories(test_crossval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # Scalar uses USE_32BITS, SIMD uses OPENGJK_SIMD_USE_FLOAT - need USE_32BITS for gkFloat type
    if(USE_32BITS)
        target_compile_definitions(test_crossval PRIVATE USE_32BITS)
    endif()
    gtest_discover_tests(test_crossval)
else()
    message(STATUS "Skipping crossval tests - scalar library not available")
endif()

