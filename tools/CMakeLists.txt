option(NINFER_BUILD_FLASH_NEXT_REFERENCE "Build target-private Qwen3.8-Flash-Next reference tool" OFF)
if(NINFER_BUILD_FLASH_NEXT_REFERENCE OR BUILD_TESTING OR NINFER_BUILD_FLASH_NEXT_TOOLS)
  add_subdirectory(reference/qwen3_8_flash_next)
endif()
