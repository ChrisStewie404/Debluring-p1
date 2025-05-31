# Install script for directory: /home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN" TYPE FILE FILES
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/MNNDefine.h"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/Interpreter.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/HalideRuntime.h"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/Tensor.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/ErrorCode.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/ImageProcess.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/Matrix.h"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/Rect.h"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/MNNForwardType.h"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/AutoTime.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/MNNSharedContext.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN/expr" TYPE FILE FILES
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/Expr.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/ExprCreator.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/MathOp.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/Optimizer.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/Executor.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/Module.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/ExecutorScope.hpp"
    "/home/anllinux/MLsys/Debluring-p1/renesas_evh/MNN_renesas/include/MNN/expr/Scope.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/anllinux/MLsys/Debluring-p1/renesas_evh/build/MNN_renesas/libMNN.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/anllinux/MLsys/Debluring-p1/renesas_evh/build/MNN_renesas/express/cmake_install.cmake")
  include("/home/anllinux/MLsys/Debluring-p1/renesas_evh/build/MNN_renesas/tools/cv/cmake_install.cmake")
  include("/home/anllinux/MLsys/Debluring-p1/renesas_evh/build/MNN_renesas/tools/converter/cmake_install.cmake")
  include("/home/anllinux/MLsys/Debluring-p1/renesas_evh/build/MNN_renesas/tools/audio/cmake_install.cmake")

endif()

