cmake_minimum_required(VERSION 3.26)

project(vp_kd_trees)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(common)
add_subdirectory(kd)
add_subdirectory(vp)
add_subdirectory(comp)

file(COPY
    media
    dataset
    DESTINATION ${CMAKE_BINARY_DIR}
)
