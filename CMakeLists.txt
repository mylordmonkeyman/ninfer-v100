cmake_minimum_required(VERSION 3.28)

# NInfer has two explicit CUDA build contracts. Keep architecture selection
# single-valued: target-private source ownership and host dispatch depend on one
# coherent ISA contract per binary.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architecture to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a" AND
   NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a or 70; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

set(NINFER_VOLTA_BUILD OFF)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  set(NINFER_VOLTA_BUILD ON)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# __CUDA_ARCH__ exists only during device compilation. Host-side target dispatch
# therefore uses this build definition as the authoritative SM70 signal.
if(NINFER_VOLTA_BUILD)
  add_compile_definitions(NINFER_VOLTA_BUILD=1)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(NINFER_VOLTA_BUILD)
  # CUDA 12.8 is the pinned Volta toolchain. CUDA 13 removed sm_70 support, and
  # accepting a different 12.x release would make the compatibility audit
  # non-reproducible.
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8 OR
     NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9)
    message(FATAL_ERROR
      "The sm_70 Volta build requires CUDA 12.8.x; found CUDA compiler "
      "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
else()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
    message(FATAL_ERROR
      "The sm_120a build requires CUDA 13.1 or newer; found CUDA compiler "
      "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
endif()

# CUDA 13's CCCL headers require MSVC's standard-conforming preprocessor.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  # windows.h defines max/min as macros that break std::max/std::min.
  add_compile_definitions(NOMINMAX)
endif()

# Media (vision) decode and acquisition need FFMPEG and libcurl. Those are not
# part of the default Windows toolchain, so the text-only build
# (NINFER_BUILD_MEDIA=OFF) compiles API-compatible stubs instead and rejects
# vision requests at runtime.
option(NINFER_BUILD_MEDIA "Build FFMPEG/libcurl media decode and acquisition" ON)

# ninfer_serve and prompt_input link ninfer_media_acquire unconditionally, so
# the target must exist whenever apps/tests build; NINFER_BUILD_MEDIA only
# selects the real (libcurl) vs stub implementation.
set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(NINFER_BUILD_MEDIA)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFMPEG IMPORTED_TARGET
      libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
    if(NINFER_BUILD_MEDIA_ACQUIRE)
      pkg_check_modules(LIBCURL IMPORTED_TARGET libcurl>=7.85)
    endif()
  endif()

  if(NOT TARGET PkgConfig::FFMPEG)
    set(FFMPEG_ROOT "P:/third_party/ffmpeg/ffmpeg-master-latest-win64-gpl-shared" CACHE PATH "FFmpeg root directory")
    find_path(FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h
      HINTS ${FFMPEG_ROOT}/include ENV FFMPEG_INCLUDE_DIR)
    find_library(AVFORMAT_LIBRARY NAMES avformat libavformat
      HINTS ${FFMPEG_ROOT}/lib ENV AVFORMAT_LIBRARY)
    find_library(AVCODEC_LIBRARY NAMES avcodec libavcodec
      HINTS ${FFMPEG_ROOT}/lib ENV AVCODEC_LIBRARY)
    find_library(AVUTIL_LIBRARY NAMES avutil libavutil
      HINTS ${FFMPEG_ROOT}/lib ENV AVUTIL_LIBRARY)
    find_library(SWSCALE_LIBRARY NAMES swscale libswscale
      HINTS ${FFMPEG_ROOT}/lib ENV SWSCALE_LIBRARY)

    if(FFMPEG_INCLUDE_DIR AND AVFORMAT_LIBRARY AND AVCODEC_LIBRARY AND AVUTIL_LIBRARY AND SWSCALE_LIBRARY)
      add_library(PkgConfig::FFMPEG INTERFACE IMPORTED)
      target_include_directories(PkgConfig::FFMPEG INTERFACE ${FFMPEG_INCLUDE_DIR})
      target_link_libraries(PkgConfig::FFMPEG INTERFACE
        ${AVFORMAT_LIBRARY} ${AVCODEC_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})
      message(STATUS "Found FFmpeg (direct): ${FFMPEG_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "Could NOT find FFmpeg (libavformat, libavcodec, libavutil, libswscale). Set FFMPEG_ROOT or FFMPEG_INCLUDE_DIR.")
    endif()
  endif()

  if(NINFER_BUILD_MEDIA_ACQUIRE AND NOT TARGET PkgConfig::LIBCURL)
    set(CURL_ROOT "P:/third_party/curl-inst" CACHE PATH "libcurl root directory")
    find_path(CURL_INCLUDE_DIR NAMES curl/curl.h
      HINTS ${CURL_ROOT}/include ENV CURL_INCLUDE_DIR)
    find_library(CURL_LIBRARY NAMES libcurl_imp curl libcurl
      HINTS ${CURL_ROOT}/lib ENV CURL_LIBRARY)

    if(CURL_INCLUDE_DIR AND CURL_LIBRARY)
      add_library(PkgConfig::LIBCURL INTERFACE IMPORTED)
      target_include_directories(PkgConfig::LIBCURL INTERFACE ${CURL_INCLUDE_DIR})
      target_link_libraries(PkgConfig::LIBCURL INTERFACE ${CURL_LIBRARY})
      message(STATUS "Found libcurl (direct): ${CURL_INCLUDE_DIR} -> ${CURL_LIBRARY}")
    else()
      find_package(CURL REQUIRED)
      add_library(PkgConfig::LIBCURL INTERFACE IMPORTED)
      target_link_libraries(PkgConfig::LIBCURL INTERFACE CURL::libcurl)
    endif()
  endif()
endif()
find_package(Threads REQUIRED)
add_subdirectory(third_party/xgrammar)
add_compile_definitions(NINFER_BUILD_MEDIA_ENABLED=$<BOOL:${NINFER_BUILD_MEDIA}>)

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  # Product operational logging is an offline, repository-pinned dependency. Core-only builds do
  # not configure or link it.
  add_subdirectory(third_party/spdlog)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
option(NINFER_BUILD_FLASH_NEXT_TOOLS "Build Qwen3.8-Flash-Next target-private tools" OFF)
if(NINFER_BUILD_FLASH_NEXT_TOOLS OR BUILD_TESTING)
  add_subdirectory(tools)
endif()
