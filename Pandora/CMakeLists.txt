cmake_minimum_required (VERSION 2.8.11)

project(Pandora)


set(CMAKE_C_COMPILER   "/opt/intel/bin/icc")
set(CMAKE_CXX_COMPILER "/opt/intel/bin/icpc")
set(MKLROOT "/opt/intel/mkl/")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -debug -qopt-prefetch -unroll-aggressive -m64 -xSSE4.2 -ansi-alias -no-prec-div -O3 -I${MKLROOT}/include/intel64/lp64 -qopenmp-link static -I${MKLROOT}/include -std=c++11")

find_package(OpenMP)
if(OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
endif()

include_directories(include)
include_directories(/opt/intel/include)

set(SOURCE_CODE
    src/main.cpp
    )

make_directory(test)
add_executable(test/testPandora ${SOURCE_CODE})


