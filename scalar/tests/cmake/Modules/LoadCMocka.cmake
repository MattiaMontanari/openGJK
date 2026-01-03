
find_path(CMOCKA_INCLUDE_DIR
    NAMES cmocka.h
    PATHS ${CMOCKA_ROOT_DIR}/include
)

find_library(CMOCKA_LIBRARY
    NAMES cmocka cmocka_shared
    PATHS ${CMOCKA_ROOT_DIR}/include
)

if(CMOCKA_LIBRARY)
    set(CMOCKA_LIBRARIES
    ${CMOCKA_LIBRARIES}
    ${CMOCKA_LIBRARY}
)
elseif(NOT CMOCKA_LIBRARY)
    # This is an alternative in case CMocka is not installed. I have not actually tested this, but is taken from https://github.com/OlivierLDff/cmocka-cmake-example/blob/master/cmake/FetchCMocka.cmake
    include(FetchContent)

    FetchContent_Declare(
    cmocka
    GIT_REPOSITORY https://git.cryptomilk.org/projects/cmocka.git
    GIT_TAG        cmocka-1.1.5
    GIT_SHALLOW    1
    )

    set(WITH_STATIC_LIB ON CACHE BOOL "CMocka: Build with a static library" FORCE)
    set(WITH_CMOCKERY_SUPPORT OFF CACHE BOOL "CMocka: Install a cmockery header" FORCE)
    set(WITH_EXAMPLES OFF CACHE BOOL "CMocka: Build examples" FORCE)
    set(UNIT_TESTING OFF CACHE BOOL "CMocka: Build with unit testing" FORCE)
    set(PICKY_DEVELOPER OFF CACHE BOOL "CMocka: Build with picky developer flags" FORCE)

    FetchContent_MakeAvailable(cmocka)

endif(CMOCKA_LIBRARY)

list(APPEND CMAKE_MODULE_PATH 
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
)