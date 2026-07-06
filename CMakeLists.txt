cmake_minimum_required(VERSION 3.21)
project(conicxx VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# --- Options ---
option(CONICXX_BUILD_SHARED "Build conicxx as a shared library instead of static" OFF)
option(CONICXX_BUILD_TESTS "Build gtest-based unit tests" ${PROJECT_IS_TOP_LEVEL})
option(CONICXX_BUILD_EXAMPLES "Build example programs" ${PROJECT_IS_TOP_LEVEL})
option(CONICXX_BUILD_BENCHMARKS "Build the benchmark suite" ${PROJECT_IS_TOP_LEVEL})
option(CONICXX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(CONICXX_INSTALL "Generate install/export rules" ${PROJECT_IS_TOP_LEVEL})
option(CONICXX_WITH_QDLDL "Build with the QDLDL sparse LDL^T backend (github.com/osqp/qdldl)" ON)

# --- Dependencies: Eigen (header-only), find_package first, FetchContent fallback ---
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "conicxx: system Eigen3 not found, fetching Eigen 3.4.0 via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING_SAVED ${BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(Eigen3)
  set(BUILD_TESTING ${BUILD_TESTING_SAVED} CACHE BOOL "" FORCE)
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
  endif()
else()
  message(STATUS "conicxx: using system Eigen3 (${Eigen3_VERSION})")
endif()

# --- Dependencies: QDLDL (optional sparse LDL^T backend), same find/fetch pattern as Eigen ---
if(CONICXX_WITH_QDLDL)
  find_package(qdldl QUIET)
  if(NOT qdldl_FOUND)
    message(STATUS "conicxx: system qdldl not found, fetching qdldl v0.1.9 via FetchContent")
    include(FetchContent)
    # FORCE (not qdldl's own option() defaults): a sibling dependency that already vendors qdldl
    # and caches QDLDL_BUILD_STATIC_LIB itself (e.g. an older copy bundled with another solver)
    # makes qdldl's own `option(QDLDL_BUILD_STATIC_LIB ON)` a silent no-op -- option() never
    # overrides an already-cached variable -- so qdldlstatic doesn't get built and the
    # qdldl::qdldl alias below fails. Forcing it here the same way as the others avoids that.
    set(QDLDL_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
    set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(QDLDL_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
    set(QDLDL_UNITTESTS OFF CACHE BOOL "" FORCE)
    set(QDLDL_LONG OFF CACHE BOOL "" FORCE)  # QDLDL_int == int, matching conicxx::Index
    FetchContent_Declare(
      qdldl
      GIT_REPOSITORY https://github.com/osqp/qdldl.git
      GIT_TAG v0.1.9
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(qdldl)
    if(NOT TARGET qdldl::qdldl)
      add_library(qdldl::qdldl ALIAS qdldlstatic)
    endif()
  else()
    message(STATUS "conicxx: using system qdldl")
  endif()
endif()

include(cmake/CompilerWarnings.cmake)

# --- Version header ---
# Single source of truth for the version number is the project() call above;
# everything else (this header, the shared-library SOVERSION, and the
# installed CMake package-version file below) derives from it.
configure_file(
  include/conicxx/version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/conicxx/version.h
  @ONLY
)

# --- Library target ---
set(CONICXX_SOURCES
  src/conicxx/cones/zero_cone.cpp
  src/conicxx/cones/nonnegative_cone.cpp
  src/conicxx/cones/second_order_cone.cpp
  src/conicxx/cones/cone_set.cpp
  src/conicxx/kkt/sparsity_map.cpp
  src/conicxx/kkt/kkt_system.cpp
  src/conicxx/kkt/equilibration.cpp
  src/conicxx/convert.cpp
  src/conicxx/solution.cpp
  src/conicxx/solver.cpp
  src/conicxx/solver_impl.cpp
  src/conicxx/version.cpp
  src/conicxx/backend_comparison.cpp
)

if(CONICXX_WITH_QDLDL)
  list(APPEND CONICXX_SOURCES src/conicxx/kkt/qdldl_ldlt.cpp src/conicxx/kkt/regularized_ldlt.cpp)
endif()

if(CONICXX_BUILD_SHARED)
  add_library(conicxx SHARED ${CONICXX_SOURCES})
else()
  add_library(conicxx STATIC ${CONICXX_SOURCES})
endif()
add_library(conicxx::conicxx ALIAS conicxx)

target_include_directories(conicxx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(conicxx PUBLIC Eigen3::Eigen)
if(CONICXX_WITH_QDLDL)
  target_link_libraries(conicxx PRIVATE qdldl::qdldl)
  # PUBLIC (not PRIVATE): kkt_system.h is a private header, but the test target reaches into
  # src/ and includes it directly (see test/CMakeLists.txt), so it must see this macro exactly
  # like conicxx's own sources do -- otherwise KktSystem (which conditionally has a QdldlLdlt
  # member) has two different sizes/layouts across translation units, which is undefined
  # behavior and corrupts the stack (verified: this exact mismatch reliably crashed every
  # KktSystem test with a stack-buffer-overflow before being caught here). This does not leak
  # QDLDL into the public API -- no public header under include/conicxx/ is affected.
  target_compile_definitions(conicxx PUBLIC CONICXX_HAVE_QDLDL)
endif()
target_compile_features(conicxx PUBLIC cxx_std_17)
conicxx_set_warnings(conicxx)

set_target_properties(conicxx PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  EXPORT_NAME conicxx
)

# --- Tests ---
if(CONICXX_BUILD_TESTS AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

# --- Examples ---
if(CONICXX_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# --- Benchmarks ---
if(CONICXX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# --- Install / export rules ---
if(CONICXX_INSTALL)
  install(TARGETS conicxx EXPORT conicxxTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(DIRECTORY include/conicxx DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "*.in" EXCLUDE)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/conicxx/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/conicxx)

  install(EXPORT conicxxTargets
    FILE conicxx-targets.cmake
    NAMESPACE conicxx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conicxx
  )

  configure_package_config_file(
    cmake/conicxx-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/conicxx-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conicxx
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/conicxx-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/conicxx-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/conicxx-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conicxx
  )
endif()
