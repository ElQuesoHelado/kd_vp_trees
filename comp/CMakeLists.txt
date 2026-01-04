cmake_minimum_required(VERSION 3.26)

project(comp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(comp main.cpp)

target_link_libraries(comp
  kd_tree_lib
  vp_tree_lib
  ${OpenCV_LIBS}
)

target_include_directories(comp PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/kd
  ${CMAKE_SOURCE_DIR}/vp
  common
)
