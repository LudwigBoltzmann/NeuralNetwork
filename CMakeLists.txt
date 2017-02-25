cmake_minimum_required (VERSION 2.8.11)

project(NeuralNetwork)


set(CMAKE_C_COMPILER   "/opt/intel/bin/icc")
set(CMAKE_CXX_COMPILER "/opt/intel/bin/icpc")
set(MKLROOT "/opt/intel/mkl/")

#set(CMAKE_EXE_LINKER_FLAGS "-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /opt/intel/lib/intel64/libiomp5.a -lpthread -lm -ldl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -lm")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${MKLROOT}/include/intel64/lp64 -qopenmp-link static -I${MKLROOT}/include -std=c++11")

find_package(OpenMP)
if(OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ")
endif()

include_directories(.)
include_directories(/opt/intel/include)


set(SOURCE_CODE
    src/main.cpp
    )

add_executable(NeuralNetwork ${SOURCE_CODE})


