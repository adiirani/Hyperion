cmake_minimum_required(VERSION 3.10)

project(Hyperion VERSION 0.0.0)

# Include the headers directory for your library
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Define the source files for your library
set(LIB_SOURCES
    NeuralNetworks/enumsAndLoss/LossFunctions.cpp
    NeuralNetworks/layers/attentive.cpp
    NeuralNetworks/layers/baseLayerTemplate.cpp
    NeuralNetworks/layers/convo.cpp
    NeuralNetworks/layers/dropout.cpp
    NeuralNetworks/layers/embedding.cpp
    NeuralNetworks/layers/fullConn.cpp
    NeuralNetworks/layers/normalization.cpp
    NeuralNetworks/layers/pooling.cpp
    NeuralNetworks/layers/recurrent.cpp
    NeuralNetworks/layers/residual.cpp
    PreProcessing/PreProcessing.cpp
)

# Define the headers for your library
set(LIB_HEADERS
    NeuralNetworks/enumsAndLoss/headers/ActivationFunction.h
    NeuralNetworks/enumsAndLoss/headers/LayerType.h
    NeuralNetworks/enumsAndLoss/headers/LossFunction.h
    NeuralNetworks/layers/headers/attentive.h
    NeuralNetworks/layers/headers/baseLayerTemplate.h
    NeuralNetworks/layers/headers/convo.h
    NeuralNetworks/layers/headers/dropout.h
    NeuralNetworks/layers/headers/embedding.h
    NeuralNetworks/layers/headers/fullConn.h
    NeuralNetworks/layers/headers/normalization.h
    NeuralNetworks/layers/headers/pooling.h
    NeuralNetworks/layers/headers/recurrent.h
    NeuralNetworks/layers/headers/residual.h
    PreProcessing/headers/PreProcessing.h
)

# Create the library
add_library(HyperionLib ${LIB_SOURCES})

# Specify the location for installing the library and headers
install(TARGETS HyperionLib
    ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)

install(FILES ${LIB_HEADERS}
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)
