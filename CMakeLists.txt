cmake_minimum_required(VERSION 3.22)
project(
    schwan
    VERSION 0.0.1
    LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# vcpkg for OpenCV
# -----------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# -----------------------------
# Dependencies
# -----------------------------
set(OpenCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/arm64-osx/share/opencv4")
set(quirc_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/installed/arm64-osx/share/quirc")

find_package(quirc CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

# -----------------------------
# ONNX Runtime (Homebrew)
# -----------------------------
set(ONNXRUNTIME_ROOT "/opt/homebrew/Cellar/onnxruntime/1.22.2_2")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)


# -----------------------------
# Executable
# -----------------------------
add_executable(schwan schwan/main.cpp)

target_link_libraries(schwan
    PRIVATE
        ${OpenCV_LIBS}
        onnxruntime
)

# -----------------------------
# Print info for debugging
# -----------------------------
message(STATUS "Using OpenCV from: ${OpenCV_LIBS}")
message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOT}")
