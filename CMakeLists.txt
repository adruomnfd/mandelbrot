cmake_minimum_required(VERSION 3.0.0)

project(mandelbrot)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -pthread")

add_executable(${PROJECT_NAME}  
mandelbrot.cpp
ext/glad/src/glad.c
)

add_subdirectory(ext/glfw)

include_directories(${PROJECT_NAME}  PRIVATE 
ext/
ext/glad/include
)

target_link_libraries(${PROJECT_NAME}  glfw)
