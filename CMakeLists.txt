cmake_minimum_required(VERSION 3.20)
project(AdaptiveSpec LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE   "-O3")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3")

if(NOT APPLE)
  message(FATAL_ERROR "AdaptiveSpec's Metal backend requires macOS")
endif()

add_library(aspec_core INTERFACE)
target_include_directories(aspec_core INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Objective-C++ targets: ARC on, Metal + Foundation linked.
function(aspec_metal_exe name src)
  add_executable(${name} ${src})
  set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_link_libraries(${name} PRIVATE aspec_core "-framework Metal" "-framework Foundation")
endfunction()

aspec_metal_exe(metal_check src/metal_check.mm)
