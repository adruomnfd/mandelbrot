SET(PROJECT_NAME mandelbrot)

cmake_minimum_required(VERSION 3.0.0)
project(${PROJECT_NAME} VERSION 0.1.0)
include(CTest)
enable_testing()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CPack)




FILE(GLOB SOURCES CONFIGURE_DEPENDS *.c *.cpp)
FILE(GLOB HEADERS *.h *.hpp)


SET(SUB_DIRS 
glfw
)

SET(SOURCES 
${SOURCES} 
glad/src/glad.c
)

SET(INCLUDE_DIRS 
./ 
glad/include 
glfw/include
)

SET(LINK_DIRS 
${GLFW_LIBRARIES}
)

SET(LINK_LIBS 
${GLFW_LIBRARIES}
)




add_executable(${PROJECT_NAME}  ${SOURCES} ${HEADERS})



add_subdirectory(${SUB_DIRS} build)
include_directories(${PROJECT_NAME}  PRIVATE ${INCLUDE_DIRS})
link_directories(${PROJECT_NAME}  PRIVATE glfw ${LINK_DIRS})
target_link_libraries(${PROJECT_NAME}  glfw ${LINK_LIBS})
