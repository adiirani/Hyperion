cmake_minimum_required(VERSION 3.10)

project(Hyperion VERSION 0.0.0)

set(HEADER_FILES
    headers/ActivationFunction.h
    headers/attentive.h
    headers/baseLayerTemplate.h
    headers/convo.h
    headers/dropout.h
    headers/embedding.h
    headers/fullConn.h
    headers/LayerType.h
    headers/LossFunction.h
    headers/neuralNet.h
    headers/normalization.h
    headers/pooling.h
    headers/PreProcessing.h
    headers/recurrent.h
    headers/residual.h
)

# Create a list of all source files
set(SOURCE_FILES
    src/attentive.cpp
    src/baseLayerTemplate.cpp
    src/convo.cpp
    src/dropout.cpp
    src/embedding.cpp
    src/fullConn.cpp
    src/LossFunctions.cpp
    src/neuralNet.cpp
    src/normalization.cpp
    src/pooling.cpp
    src/PreProcessing.cpp
    src/recurrent.cpp
    src/residual.cpp
)

# Create a library from the source and header files
add_library(YourLibrary STATIC ${SOURCE_FILES} ${HEADER_FILES})
