add_executable(friction_contact_demo friction_contact_demo.cpp)
target_link_libraries(friction_contact_demo PRIVATE conicxx::conicxx)

foreach(example quickstart infeasibility compliant_rows)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE conicxx::conicxx)
endforeach()
