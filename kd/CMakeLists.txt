add_library(kd_tree_lib INTERFACE)

target_include_directories(kd_tree_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kd_tree_lib
    INTERFACE common
)

add_executable(kd_tree_stats main.cpp)

target_link_libraries(kd_tree_stats
    PRIVATE kd_tree_lib
)
