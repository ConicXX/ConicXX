cmake_minimum_required(VERSION 3.21)
project(coulomb_exp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_subdirectory(../.. conicxx_build EXCLUDE_FROM_ALL)  # ConicXX root
foreach(t replay fixpoint)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} conicxx::conicxx)
endforeach()
