#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "../objects/matrix.h"

/**
 * @brief Scalar Sigmoid function: 1 / (1 + exp(-x))
 */
float sigmoid(float x);

/**
 * @brief Element-wise Matrix Sigmoid function.
 * Creates a new matrix with same dimensions as input.
 */
Matrix* matrixSigmoid(const Matrix* m);

/**
 * @brief Element-wise Matrix Sigmoid function (In-place).
 * Modifies the input matrix.
 */
void matrixSigmoidInplace(Matrix* m);

#endif
