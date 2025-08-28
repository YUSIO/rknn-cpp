cmake_minimum_required(VERSION 3.16)
project(rknn-cpp LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
# 设置RKNN库路径 (默认aarch64)
set(RKNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64)


# 直接设置RKNN库文件路径
set(RKNN_LIB ${RKNN_LIB_DIR}/librknnrt.so)

# 检查库文件是否存在
if(NOT EXISTS ${RKNN_LIB})
    message(FATAL_ERROR "RKNN library not found at: ${RKNN_LIB}")
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()
# 包含目录
include_directories(include)
include_directories(3rdparty/rknpu2/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件
set(SOURCES
    src/base/base_model_impl.cpp
    src/models/resnet_model.cpp
    src/models/yolov3_model.cpp
    src/utils/image_utils.cpp
)

# 创建库
add_library(rknn_cpp SHARED ${SOURCES})

# 链接库
target_link_libraries(rknn_cpp ${RKNN_LIB})
if(OpenCV_FOUND)
    target_link_libraries(rknn_cpp ${OpenCV_LIBS})
endif()

# 构建basic_usage示例
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage rknn_cpp)
set_target_properties(basic_usage PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)
add_executable(yolotest examples/yolov3.cpp)
target_link_libraries(yolotest rknn_cpp)
set_target_properties(yolotest PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 构建OpenCV示例
add_executable(opencv_example examples/opencv_example.cpp)
target_link_libraries(opencv_example rknn_cpp)
set_target_properties(opencv_example PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 显示配置信息
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "RKNN Library: ${RKNN_LIB}")
message(STATUS "OpenCV: ${OpenCV_FOUND}")
