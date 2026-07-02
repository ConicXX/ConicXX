add_executable(conicxx_benchmarks
  problem_generators.cpp
  run_benchmarks.cpp
)
target_link_libraries(conicxx_benchmarks PRIVATE conicxx::conicxx)
target_include_directories(conicxx_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
conicxx_set_warnings(conicxx_benchmarks)

if(CONICXX_BUILD_TESTS AND BUILD_TESTING)
  # Registered as a ctest so `ctest` surfaces a hard regression signal
  # (nonzero exit if any instance fails to converge) in addition to the
  # detailed report printed when the binary is run directly. Labeled
  # separately so it can be excluded from fast dev loops with
  # `ctest -LE benchmark`.
  add_test(NAME conicxx_benchmarks COMMAND conicxx_benchmarks)
  set_tests_properties(conicxx_benchmarks PROPERTIES LABELS "benchmark")
endif()
