function(conicxx_set_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wshadow>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
  )
  if(CONICXX_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>
      $<$<CXX_COMPILER_ID:MSVC>:/WX>
    )
  endif()
endfunction()
