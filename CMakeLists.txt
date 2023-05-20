cmake_minimum_required(VERSION 3.0.0)
project(ofs)

find_package(ospray 2.0.0 REQUIRED)


add_executable(ofs main.cpp)
target_link_libraries(ofs ${client_target} ospray::ospray)
