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

# Tests reach into internal headers (KktSystem, QuasiDefiniteLdlt, cones); those live in
# src/conicxx next to the public ones, so the conicxx target's include path already covers them.
add_executable(conicxx_tests
  test_alloc_free_hot_path.cpp
  test_compliant_rows.cpp
  test_cone_set.cpp
  test_convert.cpp
  test_equilibration.cpp
  test_infeasible.cpp
  test_ipm_safeguards.cpp
  test_kkt_system.cpp
  test_nonnegative_cone.cpp
  test_quasidefinite_ldlt.cpp
  test_second_order_cone.cpp
  test_solve_apex_contacts.cpp
  test_solve_dense_input.cpp
  test_solve_equality_robustness.cpp
  test_solve_friction_qp_soc.cpp
  test_solve_lp.cpp
  test_solve_qp.cpp
  test_solve_socp.cpp
  test_sparsity_map.cpp
  test_termination_unscaled.cpp
  test_update_reuse.cpp
  test_version.cpp
  test_warm_start.cpp
  test_zero_cone.cpp
)
target_link_libraries(conicxx_tests PRIVATE conicxx::conicxx GTest::gtest_main)
target_include_directories(conicxx_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
conicxx_set_warnings(conicxx_tests)

gtest_discover_tests(conicxx_tests)
