 cmake_minimum_required (VERSION 2.8)  
  
project (seg_infer)  
  
add_executable(seg_infer seg_main.cpp)  
  
include_directories ( /home/chao/ICNet-master/PSPNet/include  
    /usr/local/include 
    /home/chao/ICNet-master/evaluation/seg_test/lib 
    /usr/local/cuda/include  
    /usr/include )  
  
target_link_libraries(seg_infer  
    /home/chao/ICNet-master/evaluation/seg_test/lib/libseg_test.so
    /home/chao/ICNet-master/PSPNet/build/lib/libcaffe.so  
    /usr/local/lib/libopencv_highgui.so   
    /usr/local/lib/libopencv_core.so   
    /usr/local/lib/libopencv_imgproc.so 
    /usr/local/lib/libopencv_imgcodecs.so
    /usr/local/lib/libopencv_shape.so
    /usr/local/lib/libopencv_videoio.so
    /usr/lib/x86_64-linux-gnu/libglog.so  
    /usr/lib/x86_64-linux-gnu/libboost_system.so  
    ) 
