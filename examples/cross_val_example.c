#include <stdio.h>
#include <stdlib.h>
#include "model_selection.h"
#include "dataloader.h"

int main() {
    printf("Standalone Cross-Validation Example\n");
    printf("===================================\n\n");

    // 1. Create synthetic binary classification dataset
    int n_samples = 100;
    int n_features = 4;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);
    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int j = 0; j < n_features; j++) {
            x_data[i * n_features + j] = (float)rand() / RAND_MAX * 10 - 5;
            sum += x_data[i * n_features + j];
        }
        y_data[i] = (sum > 0) ? 1.0f : -1.0f;
    }

    // 2. Perform 5-Fold Cross-Validation for SVM
    int cv_folds = 5;
    printf("Performing %d-Fold Cross-Validation for SVM (C=1.0, lr=0.01, iter=1000)...\n", cv_folds);
    float* scores = crossValScoreSVM(X, y, 1.0f, 0.01f, 1000, cv_folds);

    float sum_scores = 0.0f;
    for (int i = 0; i < cv_folds; i++) {
        printf("  Fold %d Accuracy: %.2f%%\n", i + 1, scores[i] * 100);
        sum_scores += scores[i];
    }
    printf("\n  Mean Accuracy: %.2f%%\n", (sum_scores / cv_folds) * 100);

    // 3. Perform 5-Fold Cross-Validation for LinearClassifier
    printf("\nPerforming %d-Fold Cross-Validation for LinearClassifier (lr=0.1, iter=500, lambda=0.01)...\n", cv_folds);
    float* scores_lin = crossValScoreLinearClassifier(X, y, 0.1f, 500, 0.01f, cv_folds);

    float sum_scores_lin = 0.0f;
    for (int i = 0; i < cv_folds; i++) {
        printf("  Fold %d Accuracy: %.2f%%\n", i + 1, scores_lin[i] * 100);
        sum_scores_lin += scores_lin[i];
    }
    printf("\n  Mean Accuracy: %.2f%%\n", (sum_scores_lin / cv_folds) * 100);

    // Cleanup
    free(scores);
    free(scores_lin);
    freeMatrix(X);
    freeMatrix(y);

    return 0;
}
