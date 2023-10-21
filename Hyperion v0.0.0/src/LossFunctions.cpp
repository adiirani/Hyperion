#include "../headers/LossFunction.h"
#include <cmath>


double lossFunc(std::vector<double> predicted, std::vector<double> actual, LossFunction type) {
	int n = predicted.size(); // Assuming predicted and actual vectors have the same length.

	double loss = 0.0;

	double delta = 1.0; // Delta parameter
	double margin = 1.0; // Margin parameter
	double margin_contrastive = 1.0; // Margin for contrastive loss
	double margin_triplet = 1.0; // Margin for triplet loss

	switch (type) {
	case MSE:
		for (int i = 0; i < n; i++) {
			double error = predicted[i] - actual[i];
			loss += error * error;
		}
		loss /= n; // Divide by the number of samples.
		break;

	case MAE:
		for (int i = 0; i < n; i++) {
			double error = predicted[i] - actual[i];
			loss += (error >= 0) ? error : -error;
		}
		loss /= n; // Divide by the number of samples.
		break;

	case Huber:
		for (int i = 0; i < n; i++) {
			double error = predicted[i] - actual[i];
			if (error <= delta && error >= -delta) {
				loss += 0.5 * error * error;
			}
			else {
				loss += delta * ((error >= 0) ? (error - 0.5 * delta) : (-error - 0.5 * delta));
			}
		}
		loss /= n;
		break;

	case CrossEntropy:
		for (int i = 0; i < n; i++) {
			double p = predicted[i] / static_cast<double>(actual[i]);
			loss += -actual[i] * std::log(p);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case CategoricalCrossEntropy:
		for (int i = 0; i < n; i++) {
			double p = predicted[i];
			loss += -actual[i] * std::log(p);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case SparseCategoricalCrossEntropy:
		for (int i = 0; i < n; i++) {
			double p = predicted[i];
			loss += -actual[i] * std::log(p);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case Hinge:
		for (int i = 0; i < n; i++) {
			double error = 1.0 - predicted[i] * actual[i];
			loss += (error >= margin) ? 0 : (margin - error);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case Poisson:
		for (int i = 0; i < n; i++) {
			double lambda = predicted[i];
			double x = actual[i];
			loss += lambda - x * std::log(lambda);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case KLDivergence:
		for (int i = 0; i < n; i++) {
			double p = predicted[i];
			double q = actual[i];
			loss += p * std::log(p / q);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case Contrastive:
		for (int i = 0; i < n; i++) {
			double distance = predicted[i];
			loss += (distance >= margin_contrastive) ? 0 : (margin_contrastive - distance) * (margin_contrastive - distance);
		}
		loss /= n; // Divide by the number of samples.
		break;

	case Triplet:
		for (int i = 0; i < n; i++) {
			double anchor = predicted[i];
			double positive = actual[i];
			double negative = 0.0; // Replace with the negative example.

			double d_pos = (anchor - positive + margin_triplet >= 0) ? (anchor - positive + margin_triplet) : 0;
			double d_neg = (negative - anchor + margin_triplet >= 0) ? (negative - anchor + margin_triplet) : 0;

			loss += d_pos + d_neg;
		}
		loss /= n; // Divide by the number of samples.
		break;

	case Wasserstein:
		for (int i = 0; i < n; i++) {
			double x = predicted[i];
			double y = actual[i];
			loss += (x >= y) ? (x - y) : (y - x);
		}
		loss /= n; // Divide by the number of samples.
		break;

	default:
		// MSE is default. If you don't like it, you should have specified a loss function.
		for (int i = 0; i < n; i++) {
			double error = predicted[i] - actual[i];
			loss += error * error;
		}
		loss /= n;
		break;
	}

	return loss;
}
