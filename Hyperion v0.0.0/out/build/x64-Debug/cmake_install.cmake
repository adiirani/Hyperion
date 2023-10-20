# Install script for directory: C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/HyperionLib.lib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0" TYPE STATIC_LIBRARY FILES "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/out/build/x64-Debug/HyperionLib.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/ActivationFunction.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/LayerType.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/LossFunction.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/attentive.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/baseLayerTemplate.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/convo.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/dropout.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/embedding.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/fullConn.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/normalization.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/pooling.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/recurrent.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/residual.h;C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/PreProcessing.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0" TYPE FILE FILES
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/enumsAndLoss/headers/ActivationFunction.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/enumsAndLoss/headers/LayerType.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/enumsAndLoss/headers/LossFunction.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/attentive.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/baseLayerTemplate.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/convo.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/dropout.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/embedding.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/fullConn.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/normalization.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/pooling.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/recurrent.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/NeuralNetworks/layers/headers/residual.h"
    "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/PreProcessing/headers/PreProcessing.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/adi_i/Downloads/Hyperion-main/Hyperion-main/Hyperion v0.0.0/out/build/x64-Debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
