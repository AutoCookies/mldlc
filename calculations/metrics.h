#ifndef METRICS_H
#define METRICS_H

#include "../objects/matrix.h"

float meanAbsoluteError(const Matrix *y_true, const Matrix *y_pred);
float meanSquaredError(const Matrix *y_true, const Matrix *y_pred);
float rootMeanSquaredError(const Matrix *y_true, const Matrix *y_pred);
float binaryCrossEntropy(const Matrix *y_true, const Matrix *y_pred);
float rSquared(const Matrix *y_true, const Matrix *y_pred);
float adjustedRSquared(float r2, int n, int k);
float accuracyScore(const Matrix *y_true, const Matrix *y_pred);
float precisionScore(const Matrix *y_true, const Matrix *y_pred);
float recallScore(const Matrix *y_true, const Matrix *y_pred);
float f1Score(const Matrix *y_true, const Matrix *y_pred);
float calculateEntropy(const float *probs, int n_classes);
float giniImpurity(const float *probs, int n_classes);

#endif