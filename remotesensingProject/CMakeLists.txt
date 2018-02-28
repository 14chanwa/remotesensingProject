cmake_minimum_required(VERSION 3.0)

#Configure project
project(RemoteSensingProject)
set(PROJECT_DESCRIPTION "Implements some depth map estimator from 3D light fields")

# Set C++ compiler flags.
set(CMAKE_CXX_FLAGS "")

# Specify library content :
# Recursively include all .cpp files
#file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES "src/*.cpp")
# Add directory containing the headers, and put all headers in a variable.
include_directories(include)
file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS "include/*.hpp")

find_package( OpenCV REQUIRED )
add_executable( main ./src/main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )

