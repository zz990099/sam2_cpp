cmake_minimum_required(VERSION 3.8)
project(sam2)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

set(source_file 
  src/sam2.cpp 
  src/sam2_track.cpp
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})   

target_link_libraries(${PROJECT_NAME} PUBLIC 
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()