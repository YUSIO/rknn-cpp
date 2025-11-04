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
    src/models/custom_model.cpp
)

# 创建库
add_library(rknn_cpp SHARED ${SOURCES})

# 链接库
target_link_libraries(rknn_cpp ${RKNN_LIB})
if(OpenCV_FOUND)
    target_link_libraries(rknn_cpp ${OpenCV_LIBS})
endif()
set_target_properties(rknn_cpp PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 构建OpenCV示例
add_executable(opencv_example examples/opencv_example.cpp)
target_link_libraries(opencv_example rknn_cpp)
set_target_properties(opencv_example PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# 显示配置信息
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "RKNN Library: ${RKNN_LIB}")
message(STATUS "OpenCV: ${OpenCV_FOUND}")

# ================ INSTALL 配置 ============

# 设置安装路径
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install prefix" FORCE)

# 1. 安装头文件
install(DIRECTORY include/
    DESTINATION include/
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 2. 安装RKNN头文件
install(DIRECTORY 3rdparty/rknpu2/include/
    DESTINATION include/
    FILES_MATCHING PATTERN "*.h"
)

# 3. 安装库文件
install(TARGETS rknn_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 4. 安装RKNN依赖库
install(FILES ${RKNN_LIB}
    DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

# 5. 安装可执行文件
install(TARGETS  opencv_example
    RUNTIME DESTINATION bin
)

# 6. 安装示例代码
install(DIRECTORY examples/
    DESTINATION examples
    FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h" PATTERN "CMakeLists.txt"
)
# 7. 安装模型文件
install(DIRECTORY models/
    DESTINATION models
)

# 8. 安装输入文件
install(DIRECTORY inputs/
    DESTINATION inputs
)

# 打印安装信息
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Install to: ${CMAKE_INSTALL_PREFIX}")