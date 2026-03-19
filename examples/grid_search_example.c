#include <stdio.h>
#include <stdlib.h>
#include "model_selection.h"
#include "dataloader.h"

int main() {
    printf("GridSearchCV Example (Tuning SVM)\n");
    printf("=================================\n\n");

    // 1. Create synthetic binary classification dataset
    int n_samples = 40;
    int n_features = 2;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);
    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        x_data[i*2] = (float)rand() / RAND_MAX * 10 - 5;
        x_data[i*2+1] = (float)rand() / RAND_MAX * 10 - 5;
        y_data[i] = (x_data[i*2] + x_data[i*2+1] > 0) ? 1.0f : -1.0f;
    }

    // 2. Define Parameter Grid for SVM
    float C_vals[] = {0.1f, 1.0f, 10.0f};
    float lr_vals[] = {0.001f, 0.01f};
    int iter_vals[] = {1000, 2000};

    SVMParamGrid grid = {
        .C_values = C_vals, .n_C = 3,
        .lr_values = lr_vals, .n_lr = 2,
        .iter_values = iter_vals, .n_iter = 2
    };

    // 3. Run GridSearch with 5-fold CV
    printf("Running Grid Search with 5-fold cross-validation...\n");
    SVMGridResult result = gridSearchSVM(X, y, grid, 5);

    printf("\nBest Parameters Found:\n");
    printf("  C:             %.2f\n", result.best_C);
    printf("  Learning Rate: %.4f\n", result.best_lr);
    printf("  Iterations:    %d\n", result.best_iter);
    printf("  Best Score:    %.2f%%\n", result.best_score * 100);

    // Cleanup
    freeMatrix(X);
    freeMatrix(y);

    return 0;
}
