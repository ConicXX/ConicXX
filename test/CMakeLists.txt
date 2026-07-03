find_package(GTest 1.14 QUIET)
if(NOT GTest_FOUND)
  message(STATUS "conicxx: system GTest not found, fetching googletest via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
else()
  message(STATUS "conicxx: using system GTest")
endif()

include(GoogleTest)

add_executable(conicxx_tests
  cones/test_zero_cone.cpp
  cones/test_nonnegative_cone.cpp
  cones/test_second_order_cone.cpp
  cones/test_cone_set.cpp
  convert/test_convert.cpp
  kkt/test_sparsity_map.cpp
  kkt/test_kkt_system.cpp
  kkt/test_equilibration.cpp
  solver/test_solve_qp.cpp
  solver/test_solve_lp.cpp
  solver/test_solve_socp.cpp
  solver/test_solve_friction_qp_soc.cpp
  solver/test_solve_dense_input.cpp
  solver/test_infeasible.cpp
  solver/test_update_reuse.cpp
  version/test_version.cpp
)

target_link_libraries(conicxx_tests PRIVATE conicxx::conicxx GTest::gtest_main)
target_include_directories(conicxx_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
conicxx_set_warnings(conicxx_tests)

gtest_discover_tests(conicxx_tests)
