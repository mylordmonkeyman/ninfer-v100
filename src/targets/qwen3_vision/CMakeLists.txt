add_library(ninfer_qwen3_vision STATIC
  impl/encoder.cpp)
ninfer_internal_includes(ninfer_qwen3_vision)
target_include_directories(ninfer_qwen3_vision PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(ninfer_qwen3_vision PUBLIC
  ninfer_core
  ninfer_ops)
