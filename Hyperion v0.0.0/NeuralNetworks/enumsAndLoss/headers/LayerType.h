#ifndef LAYERTYPE_H
#define LAYERTYPE_H

enum LayerType {
	FullConn, //Bog-standard Full Connect Layer (dense in TF but screw that they should have made it clear); used for anything
	Convo, //Convolutional layer
	Pooling, //Pooling layer
	Recurrent, //Recurrent layer
	Embedding,
	Normalization,
	Dropout,
	Attention,
	Residual
};

#endif