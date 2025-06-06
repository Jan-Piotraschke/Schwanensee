cmake_minimum_required(VERSION 3.22)
project(
    custom_ops
    VERSION 0.0.1
    LANGUAGES CXX
)

find_package(Torch REQUIRED)

add_executable(main src/main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
