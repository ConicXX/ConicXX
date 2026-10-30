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

# Tests reach into internal headers (conicxx/cones/, conicxx/kkt/, conicxx/ipm/); they live under
# src/conicxx like the public ones, so the conicxx target's include path already covers them. The
# test directories mirror that layout; api/ covers the public helpers.
add_executable(conicxx_tests
  api/test_convert.cpp
  api/test_version.cpp
  cones/test_cone_set.cpp
  cones/test_nonnegative_cone.cpp
  cones/test_second_order_cone.cpp
  cones/test_zero_cone.cpp
  kkt/test_kkt_system.cpp
  kkt/test_quasidefinite_ldlt.cpp
  kkt/test_sparsity_map.cpp
  solver/test_alloc_free_hot_path.cpp
  solver/test_compliant_rows.cpp
  solver/test_coulomb.cpp
  solver/test_equilibration.cpp
  solver/test_infeasible.cpp
  solver/test_ipm_safeguards.cpp
  solver/test_solve_apex_contacts.cpp
  solver/test_solve_dense_input.cpp
  solver/test_solve_equality_robustness.cpp
  solver/test_solve_friction_qp_soc.cpp
  solver/test_solve_lp.cpp
  solver/test_solve_qp.cpp
  solver/test_solve_socp.cpp
  solver/test_termination_unscaled.cpp
  solver/test_update_reuse.cpp
  solver/test_warm_start.cpp
)
target_link_libraries(conicxx_tests PRIVATE conicxx::conicxx GTest::gtest_main)
target_include_directories(conicxx_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
conicxx_set_warnings(conicxx_tests)

gtest_discover_tests(conicxx_tests)
