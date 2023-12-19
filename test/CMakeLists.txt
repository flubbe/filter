cmake_minimum_required(VERSION 3.15)

project(filter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED CONFIG)

include_directories(../include)

add_executable(ukf src/ukf.cpp)
target_link_libraries(ukf fmt::fmt Eigen3::Eigen GTest::gtest GTest::gtest_main)
