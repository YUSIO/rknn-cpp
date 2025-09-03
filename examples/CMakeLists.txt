cmake_minimum_required(VERSION 3.16)
project(rknn_cpp_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库和头文件路径（相对于examples目录）
set(RKNN_CPP_ROOT "..")
include_directories(${RKNN_CPP_ROOT}/include)
link_directories(${RKNN_CPP_ROOT}/lib)

# 查找OpenCV（可选）
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()

# 构建basic_usage
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage rknn_cpp rknnrt)

# 如果有OpenCV，构建opencv_example
if(OpenCV_FOUND)
    add_executable(opencv_example opencv_example.cpp)
    target_link_libraries(opencv_example rknn_cpp rknnrt ${OpenCV_LIBS})
endif()

# 设置运行时库路径
set_target_properties(basic_usage PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${RKNN_CPP_ROOT}/lib"
)

if(OpenCV_FOUND)
    set_target_properties(opencv_example PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${RKNN_CPP_ROOT}/lib"
    )
endif()