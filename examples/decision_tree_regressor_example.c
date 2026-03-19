#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../objects/matrix.h"
#include "../al/decisionTree.h"
#include "../calculations/costFunction.h"

int main() {
    printf("Decision Tree Regressor Example (Sine Wave)\n");
    printf("===========================================\n\n");

    // 1. Create synthetic dataset: y = sin(x)
    int n_samples = 50;
    Matrix* X = createMatrix(n_samples, 1, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_ptr = (float*)X->data;
    float* y_ptr = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        float x = (float)i / n_samples * 2.0f * 3.14159f;
        x_ptr[i] = x;
        y_ptr[i] = sinf(x);
    }

    // 2. Initialize and Train Regressor
    // Using MSE criterion, max_depth=4, min_samples_split=5, min_samples_leaf=2, max_features=-1, min_imp_dec=0.0, seed=42
    DecisionTree* reg = createDecisionTree(TREE_REGRESSION, CRITERION_MSE, 4, 5, 2, -1, 0.0f, 42);
    printf("Training Decision Tree Regressor...\n");
    trainDecisionTree(reg, X, y);

    // 3. Evaluate
    Matrix* y_pred = predictDecisionTree(reg, X);
    
    printf("\nEvaluation Metrics:\n");
    float mse = computeCost(COST_MSE, y, y_pred);
    printf("  Mean Squared Error: %.6f\n", mse);

    // Sample predictions
    printf("\nSample Predictions (Subset):\n");
    float* yp_data = (float*)y_pred->data;
    for (int i = 0; i < 10; i += 2) {
        printf("  x=%.2f: True=%.4f, Pred=%.4f\n", x_ptr[i], y_ptr[i], yp_data[i]);
    }

    // Cleanup
    freeMatrix(X); freeMatrix(y); freeMatrix(y_pred);
    freeDecisionTree(reg);

    return 0;
}
