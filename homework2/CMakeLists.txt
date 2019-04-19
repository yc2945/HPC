cmake_minimum_required(VERSION 3.13)
project(homework2)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

include_directories(.)

add_executable(homework2
        MMult1.cpp
        omp_bug2.c
        omp_bug3.c
        omp_bug4.c
        omp_bug5.c
        omp_bug6.c
        utils.h
        val_test01.cpp
        val_test01_solved.cpp
        val_test02.cpp
        val_test02_solved.cpp)
