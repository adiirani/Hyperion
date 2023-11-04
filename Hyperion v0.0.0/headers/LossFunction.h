#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <vector>

enum LossFunction {
	MSE,
	MAE,
	Huber,
	CrossEntropy,
	CategoricalCrossEntropy,
	SparseCategoricalCrossEntropy,
	Hinge,
	Triplet,
	Poisson,
	KLDivergence,
	Contrastive,
	Wasserstein
};

double lossFunc(std::vector<double> predicted, std::vector<double> actual, LossFunction type);

#endif
