add_library(vp_tree_lib STATIC vp_tree.cpp)

target_include_directories(vp_tree_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(vp_tree_lib
    PUBLIC common
)

add_executable(vp_tree_stats
    main.cpp
)

target_link_libraries(vp_tree_stats vp_tree_lib)
