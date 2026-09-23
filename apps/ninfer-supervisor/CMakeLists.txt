if(NOT WIN32)
  return()
endif()

# WIN32 links this for the Windows subsystem so no console window appears when
# the supervisor autostarts at login. It still has a main() rather than a
# WinMain(), so the entry point is set back explicitly; main() adopts the
# parent terminal when the binary is run by hand (see attach_parent_console_if_any).
add_executable(ninfer-supervisor WIN32
  main.cpp
  engine_child.cpp
  collector.cpp
  server.cpp
  tray.cpp)
target_include_directories(ninfer-supervisor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-supervisor PRIVATE CUDA::cudart)
target_link_options(ninfer-supervisor PRIVATE
  "LINKER:/ENTRY:mainCRTStartup"
  "LINKER:/DEFAULTLIB:dxgi"
  "LINKER:/DEFAULTLIB:pdh"
  "LINKER:/DEFAULTLIB:ole32"
  "LINKER:/DEFAULTLIB:oleaut32"
  "LINKER:/DEFAULTLIB:uuid"
  "LINKER:/DEFAULTLIB:shell32"
  "LINKER:/DEFAULTLIB:user32"
  "LINKER:/DEFAULTLIB:gdi32"
  "LINKER:/DEFAULTLIB:advapi32"
  "LINKER:/DEFAULTLIB:ws2_32"
  "LINKER:/DEFAULTLIB:crypt32")
target_compile_definitions(ninfer-supervisor PRIVATE
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  _WIN32_WINNT=0x0A00)
