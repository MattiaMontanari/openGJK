# Scalar openGJK unit tests (CMocka)

message(STATUS "[opengjk_scalar] Building unit tests with CMocka")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/LoadCMocka.cmake)
include(AddCMockaTest)

set(PASSING_UNIT_TESTS
    subalg_1simplex_test
    subalg_2simplex_test
    subalg_3simplex_test
    HFFtest_test
    test_simple
    leak
)

foreach(_UNIT_TEST ${PASSING_UNIT_TESTS})
    add_cmocka_test(${_UNIT_TEST}
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${_UNIT_TEST}.c
        COMPILE_OPTIONS -D_GNU_SOURCE -I${CMOCKA_INCLUDE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/../include/
        LINK_LIBRARIES opengjk_scalar_static ${CMOCKA_LIBRARIES} m)

    add_cmocka_test_environment(${_UNIT_TEST})

    set_tests_properties(${_UNIT_TEST} PROPERTIES WILL_FAIL 0)
endforeach()

