#ifndef REGULARIZE_H
#define REGULARIZE_H

#include "../objects/matrix.h"

typedef enum {
    REG_NONE,
    REG_L1,
    REG_L2,
    REG_ELASTIC_NET
} RegularizationType;

/**
 * @brief Applies regularization gradient penalty to the weights gradient.
 * 
 * @param type Type of regularization (L1, L2, Elastic Net).
 * @param lambda Regularization strength.
 * @param alpha L1 ratio for Elastic Net (0.0 to 1.0).
 * @param weights The current model weights.
 * @param gradient The gradient to be updated with the penalty.
 */
void applyRegularizationGradient(RegularizationType type, float lambda, float alpha, const Matrix* weights, Matrix* gradient);

/**
 * @brief Computes the regularization loss/penalty for the total cost.
 * 
 * @param type Type of regularization.
 * @param lambda Regularization strength.
 * @param alpha L1 ratio for Elastic Net.
 * @param weights The model weights.
 * @return float The computed penalty.
 */
float computeRegularizationPenalty(RegularizationType type, float lambda, float alpha, const Matrix* weights);

#endif
