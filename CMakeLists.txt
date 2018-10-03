
cmake_minimum_required(VERSION 2.8)

project(MRI_RECON)
set(PROGNAME c13_recon_binary)

file(GLOB LOCAL_INCS ${CMAKE_SOURCE_DIR}/*.h)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CmakeModules/")

set(CMAKE_FIND_STATIC FIRST) 
set(BUILD_SHARED_LIBS OFF)
set(STATIC_LIBS ON)

ADD_DEFINITIONS(-DLINUX -DRECON_64BIT -D_FILE_OFFSET_BITS=64)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -Wall -m64 -O2 -fopenmp")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})
set(CMAKE_C_FLAGS "-g -Wno-deprecated")

set(LOCAL_LIBS fftw3f_omp fftw3f blitz m)

# Find Armadillo
include(FindARMADILLO)
include_directories( BEFORE ${ARMADILLO_INCLUDE_DIRS} )

#--------------------------------------
# Target Library 
#--------------------------------------

#Main Recon
add_library(RECONLIB STATIC ArrayTemplates.cpp clear.cpp threshold.cpp temporal_diff.cpp wavelet3D.cpp )
target_link_libraries(RECONLIB ${ARMADILLO_LIBRARIES} ${LOCAL_LIBS})

#--------------------------------------
# Support for Command Line Binary
#--------------------------------------
add_executable(c13_recon_binary recon.cxx ${LOCAL_INCS})
target_link_libraries(c13_recon_binary ${ARMADILLO_LIBRARIES} ${LOCAL_LIBS} RECONLIB)





