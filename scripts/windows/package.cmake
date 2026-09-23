cmake_minimum_required(VERSION 3.28)
if(POLICY CMP0207)
  cmake_policy(SET CMP0207 NEW)
endif()
if(NOT DEFINED DESTINATION)
  message(FATAL_ERROR "Pass -DDESTINATION=<app directory>")
endif()
file(MAKE_DIRECTORY "${DESTINATION}")
file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES ${NINFER_EXECUTABLES}
  DIRECTORIES ${NINFER_RUNTIME_DIRS}
  RESOLVED_DEPENDENCIES_VAR dependencies
  UNRESOLVED_DEPENDENCIES_VAR unresolved
  PRE_EXCLUDE_REGEXES "^[Aa][Pp][Ii]-[Mm][Ss]-" "^[Ee][Xx][Tt]-[Mm][Ss]-"
  POST_EXCLUDE_REGEXES "[/\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\]")
if(unresolved)
  message(FATAL_ERROR "Missing runtime DLLs: ${unresolved}")
endif()
file(COPY ${NINFER_EXECUTABLES} ${dependencies} ${NINFER_MSVC_RUNTIME} DESTINATION "${DESTINATION}")
message(STATUS "Packaged Windows applications and runtime DLLs in ${DESTINATION}")
