cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(supervoxel_mi)

find_package(PCL 1.7 REQUIRED COMPONENTS common io visualization octree)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (supervoxel_mi supervoxel_mi.cpp supervoxel_cluster_search.cpp supervoxel_mapping.cpp) 
target_link_libraries (supervoxel_mi ${PCL_LIBRARIES} ${Boost_LIBRARIES})
