add_library(ninfer_third_party_spdlog STATIC
  src/async.cpp
  src/bundled_fmtlib_format.cpp
  src/cfg.cpp
  src/color_sinks.cpp
  src/file_sinks.cpp
  src/spdlog.cpp
  src/stdout_sinks.cpp)
add_library(spdlog::spdlog ALIAS ninfer_third_party_spdlog)

target_compile_features(ninfer_third_party_spdlog PUBLIC cxx_std_11)
target_compile_definitions(ninfer_third_party_spdlog PUBLIC SPDLOG_COMPILED_LIB)
target_include_directories(ninfer_third_party_spdlog SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ninfer_third_party_spdlog PUBLIC Threads::Threads)

# MSVC compiles as the system ANSI code page unless told otherwise, and the fmt library
# bundled inside spdlog static_asserts that the compiler is in UTF-8 mode. The flag has to
# reach every translation unit that includes the fmt headers, not just this library's own
# sources, so it is PUBLIC.
if(MSVC)
  target_compile_options(ninfer_third_party_spdlog PUBLIC /utf-8)
endif()
