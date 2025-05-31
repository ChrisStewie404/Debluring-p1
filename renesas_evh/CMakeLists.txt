project(renesas)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")


set(LLM_SUPPORT_AUDIO ON CACHE BOOL "set MNN options" FORCE)
set(LLM_SUPPORT_VISION ON CACHE BOOL "set MNN options" FORCE)
set(MNN_BUILD_AUDIO ON CACHE BOOL "set MNN options" FORCE)
set(MNN_BUILD_LLM ON CACHE BOOL "set MNN options" FORCE)
set(MNN_BUILD_OPENCV ON CACHE BOOL "set MNN options" FORCE)
set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "set MNN options" FORCE)
set(MNN_IMGCODECS ON CACHE BOOL "set MNN options" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "set MNN options" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "set MNN options" FORCE)

# find header files under these directories
include_directories(${CMAKE_CURRENT_LIST_DIR}/MNN_renesas/include
                    ${CMAKE_CURRENT_LIST_DIR}/MNN_renesas/)

# add a sub project, which contains another CMakeLists.txt
# whenever you add a subdirectory, every target declared in that directory is available.
add_subdirectory(MNN_renesas)

add_executable(main ./src/main.cpp)

target_link_libraries(main MNN MNN_Express MNNOpenCV) 

