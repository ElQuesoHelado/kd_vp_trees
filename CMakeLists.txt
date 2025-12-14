cmake_minimum_required(VERSION 3.26)

project(vptree)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(vptree main.cpp vp_tree.cpp)

target_link_libraries(
  vptree
  ${OpenCV_LIBS}
)

target_include_directories(vptree PRIVATE ${OpenCV_INCLUDE_DIRS})
