cmake_minimum_required(VERSION 3.10)
project(CUDA_TEST LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include(CheckLanguage)
check_language(CUDA)

find_package (glog  REQUIRED)
include_directories(${GLOG_INCLUDE_DIR})

add_executable(test1 test1.cu)
target_link_libraries(test1 glog::glog)