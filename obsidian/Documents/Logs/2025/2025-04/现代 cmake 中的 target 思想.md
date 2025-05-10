
现代CMake（通常指3.0及以上版本）的核心理念是围绕"target"（目标）构建的，这种思想彻底改变了CMake项目的组织方式。下面我将详细介绍这一思想的核心要素：

## 1. 从变量到目标的转变

### 传统CMake方法（不推荐）
```cmake
# 旧方式：基于变量
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
add_executable(myapp main.cpp)
target_link_libraries(myapp ${Boost_LIBRARIES})
```

### 现代CMake方法
```cmake
# 现代方式：基于目标
find_package(Boost REQUIRED)
add_executable(myapp main.cpp)
target_link_libraries(myapp Boost::boost)
```

## 2. 目标的本质

目标（Target）是现代CMake的基本构建单元，代表了构建系统中的一个实体：
- 可执行文件（通过`add_executable`创建）
- 库（通过`add_library`创建）
- 自定义目标（通过`add_custom_target`创建）
- 导入的目标（通过`add_library(... IMPORTED)`或`find_package`导入）
- 接口目标（通过`add_library(... INTERFACE)`创建，只有接口没有实现）

## 3. 目标属性与作用域

目标拥有属性，这些属性定义了目标的构建和使用方式：

### 属性作用域
- **PRIVATE**：仅用于目标自身构建，不会传递给依赖此目标的其他目标
- **INTERFACE**：不用于目标自身构建，但会传递给依赖此目标的其他目标
- **PUBLIC**：同时用于目标自身构建和传递给依赖此目标的其他目标

### 常用目标命令
```cmake
# 设置包含目录
target_include_directories(mylib 
    PUBLIC include
    PRIVATE src)

# 设置编译选项
target_compile_options(mylib 
    PRIVATE -Wall -Werror
    PUBLIC -std=c++17)

# 设置编译定义
target_compile_definitions(mylib
    PUBLIC MY_API_VERSION=2
    PRIVATE DEBUG_MODE)

# 设置链接库
target_link_libraries(mylib
    PUBLIC dependency1
    PRIVATE dependency2)
```

## 4. 依赖传递机制

现代CMake最强大的特性之一是自动处理依赖传递：

```cmake
# 库A依赖Boost
add_library(LibA ...)
target_link_libraries(LibA PUBLIC Boost::boost)

# 库B依赖库A
add_library(LibB ...)
target_link_libraries(LibB PUBLIC LibA)

# 应用程序依赖库B
add_executable(App ...)
target_link_libraries(App PRIVATE LibB)
```

在这个例子中：
- App自动获得LibB的公共依赖（包括LibA）
- App也自动获得LibA的公共依赖（包括Boost）
- 所有必要的包含路径、编译选项和链接库都会自动传递

## 5. 导入目标与导出目标

### 导入目标
```cmake
# 导入外部库
find_package(SomeLib REQUIRED)
target_link_libraries(myapp PRIVATE SomeLib::SomeLib)
```

### 导出目标
```cmake
# 导出自己的库供他人使用
install(TARGETS mylib
    EXPORT mylib-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include)

install(EXPORT mylib-targets
    FILE mylibTargets.cmake
    NAMESPACE MyLib::
    DESTINATION lib/cmake/mylib)
```

## 6. 目标思想的优势

1. **封装性**：每个目标封装自己的构建需求和使用需求
2. **模块化**：项目被组织为相互依赖的目标网络
3. **自动依赖传递**：无需手动管理传递性依赖
4. **可见性控制**：通过PRIVATE/INTERFACE/PUBLIC控制属性传递
5. **减少全局状态**：避免使用全局变量和命令
6. **更好的IDE集成**：目标映射到IDE中的项目结构

## 7. 实际应用示例

```cmake
# 创建一个库
add_library(geometry STATIC
    src/circle.cpp
    src/rectangle.cpp)

# 设置库的属性
target_include_directories(geometry
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(geometry
    PUBLIC cxx_std_17)

# 添加依赖
find_package(Eigen3 REQUIRED)
target_link_libraries(geometry
    PUBLIC Eigen3::Eigen)

# 创建使用该库的可执行文件
add_executable(geometry_app main.cpp)
target_link_libraries(geometry_app PRIVATE geometry)
```

通过这种基于目标的方法，CMake项目变得更加结构化、可维护，并且能够更好地与现代C++开发实践相结合。