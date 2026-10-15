cmake_minimum_required(VERSION 3.16)
project(exam1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

add_executable(task1 src/task1.cpp)
add_executable(task2 src/task2.cpp)
add_executable(task3 src/task3.cpp)
add_executable(task4 src/task4.cpp)