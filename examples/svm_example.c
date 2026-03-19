#include <stdio.h>
#include <stdlib.h>
#include "../objects/matrix.h"
#include "../al/svm.h"
#include "../calculations/metrics.h"

int main() {
    printf("Support Vector Machine (SVM) Example\n");
    printf("===================================\n\n");

    // 1. Create a synthetic linearly separable dataset
    // Features: [x1, x2]
    // Classes: +1 and -1
    int n_samples = 8;
    int n_features = 2;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    // Class +1: [2, 2], [2, 3], [3, 2], [3, 3]
    x_data[0] = 2.0f; x_data[1] = 2.0f; y_data[0] = 1.0f;
    x_data[2] = 2.0f; x_data[3] = 3.0f; y_data[1] = 1.0f;
    x_data[4] = 3.0f; x_data[5] = 2.0f; y_data[2] = 1.0f;
    x_data[6] = 3.0f; x_data[7] = 3.0f; y_data[3] = 1.0f;

    // Class -1: [-2, -2], [-2, -3], [-3, -2], [-3, -3]
    x_data[8] = -2.0f; x_data[9] = -2.0f; y_data[4] = -1.0f;
    x_data[10] = -2.0f; x_data[11] = -3.0f; y_data[5] = -1.0f;
    x_data[12] = -3.0f; x_data[13] = -2.0f; y_data[6] = -1.0f;
    x_data[14] = -3.0f; x_data[15] = -3.0f; y_data[7] = -1.0f;

    // 2. Initialize and Train SVM
    float C = 10.0f;
    float lr = 0.01f;
    int iter = 2000;
    SVM* model = createSVM(n_features, C, lr, iter);

    printf("Training SVM for %d iterations (C=%.1f)...\n", iter, C);
    trainSVM(model, X, y);

    // 3. Evaluation
    Matrix* y_pred = predictSVM(model, X);
    
    // Note: accuracyScore assumes labels match exactly (float comparison)
    // For SVM +1/-1, this works fine.
    float acc = accuracyScore(y, y_pred);
    printf("\nAccuracy on Training Set: %.2f%%\n", acc * 100.0f);

    printf("\nLearned Model:\n");
    float* w = (float*)model->weights->data;
    printf("  Weights: [%.4f, %.4f]\n", w[0], w[1]);
    printf("  Bias:    %.4f\n", model->bias);

    printf("\nDetailed Predictions:\n");
    float* yp_data = (float*)y_pred->data;
    for (int i = 0; i < n_samples; i++) {
        printf("  Sample %d: Features=[%.1f, %.1f], True=%.0f, Pred=%.0f\n",
               i, x_data[i*2], x_data[i*2+1], y_data[i], yp_data[i]);
    }

    // Cleanup
    freeMatrix(X);
    freeMatrix(y);
    freeMatrix(y_pred);
    freeSVM(model);

    return 0;
}
