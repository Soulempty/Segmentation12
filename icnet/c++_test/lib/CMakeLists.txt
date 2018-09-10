cmake_minimum_required (VERSION 2.8)
 
SET (SRC_LIST seg_test.cpp)

include_directories ( /home/chao/ICNet-master/PSPNet/include  
    /usr/local/include  
    /usr/local/cuda/include  
    /usr/include 
    /home/chao/ICNet-master/PSPNet/cmake)  
 
add_library(seg_test SHARED ${SRC_LIST})


