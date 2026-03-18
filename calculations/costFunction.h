#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include "../objects/matrix.h"

typedef enum {
    COST_MSE,
    COST_MAE,
    COST_RMSE,
    COST_BINARY_CROSS_ENTROPY
} CostType;

float computeCost(CostType type, const Matrix* y_true, const Matrix* y_pred);

#endif
