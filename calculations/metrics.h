#ifndef METRICS_H
#define METRICS_H

#include "../objects/matrix.h"

float meanAbsoluteError(const Matrix* y_true, const Matrix* y_pred);
float meanSquaredError(const Matrix* y_true, const Matrix* y_pred);
float rootMeanSquaredError(const Matrix* y_true, const Matrix* y_pred);
float binaryCrossEntropy(const Matrix* y_true, const Matrix* y_pred);
float rSquared(const Matrix* y_true, const Matrix* y_pred);
float adjustedRSquared(float r2, int n, int k);

#endif