#------------------------------------------------------------------------------
# Test face multitask algorithms
#------------------------------------------------------------------------------
project(bobetocalo_pami20)
cmake_minimum_required(VERSION 3.2)

include(${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/../../alignment/ert_simple/CMakeLists.txt)

#-- Find include files
set(face_multitask_bobetocalo_pami20_include
  ${faces_framework_include}
  ${face_alignment_ert_simple_include}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-genfiles
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
  ${CMAKE_CURRENT_LIST_DIR}/include
)

set(face_multitask_bobetocalo_pami20_src
  ${faces_framework_src}
  ${face_alignment_ert_simple_src}
  ${CMAKE_CURRENT_LIST_DIR}/src/FaceMultitaskMnnOr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/HonariChannelFeatures.cpp
)

set(face_multitask_bobetocalo_pami20_test
  ${CMAKE_CURRENT_LIST_DIR}/test/face_multitask_bobetocalo_pami20_test.cpp
)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/")

set(face_multitask_bobetocalo_pami20_libs
  ${faces_framework_libs}
  ${face_alignment_ert_simple_libs}
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
  dl
  pthread
)

#-- Setup CMake to run tests
enable_testing()

include_directories(${face_multitask_bobetocalo_pami20_include})

foreach(test ${face_multitask_bobetocalo_pami20_test})
  get_filename_component(test_name ${test} NAME_WE)
  add_executable(${test_name}
    ${face_multitask_bobetocalo_pami20_src}
    ${test}
  )
  target_link_libraries(${test_name} ${face_multitask_bobetocalo_pami20_libs})
  add_test(NAME ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} COMMAND ${test_name} --measure pupils --database cofw)
endforeach()
