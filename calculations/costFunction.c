#include "costFunction.h"
#include "metrics.h"
#include <stdio.h>

float computeCost(CostType type, const Matrix* y_true, const Matrix* y_pred) {
    switch (type) {
        case COST_MSE:
            return meanSquaredError(y_true, y_pred);
        case COST_MAE:
            return meanAbsoluteError(y_true, y_pred);
        case COST_RMSE:
            return rootMeanSquaredError(y_true, y_pred);
        case COST_BINARY_CROSS_ENTROPY:
            return binaryCrossEntropy(y_true, y_pred);
        default:
            printf("Error: Unknown CostType\n");
            return -1.0f;
    }
}
