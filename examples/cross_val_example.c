#include <stdio.h>
#include <stdlib.h>
#include "model_selection.h"
#include "svm.h"
#include "linearAl.h"
#include "palloc.h"

typedef struct {
    float C;
    float lr;
    int iter;
} SVMParams;

typedef struct {
    float lr;
    int iter;
    float lambda;
} LinearClassifierParams;

static void* svm_create_adapter(int n_features, void* params) {
    SVMParams* p = (SVMParams*)params;
    return createSVM(n_features, p->C, p->lr, p->iter);
}

static void svm_train_adapter(void* model, const Matrix* X, const Matrix* y) {
    trainSVM((SVM*)model, X, y);
}

static Matrix* svm_predict_adapter(void* model, const Matrix* X) {
    return predictSVM((SVM*)model, X);
}

static void svm_free_adapter(void* model) {
    freeSVM((SVM*)model);
}

static void* linear_cls_create_adapter(int n_features, void* params) {
    LinearClassifierParams* p = (LinearClassifierParams*)params;
    return createLinearClassifier(n_features, p->lr, p->iter, REG_L2, p->lambda, 0.0f);
}

static void linear_cls_train_adapter(void* model, const Matrix* X, const Matrix* y) {
    trainLinearClassifier((LinearClassifier*)model, X, y);
}

static Matrix* linear_cls_predict_adapter(void* model, const Matrix* X) {
    return predictLinearClassifierClass((LinearClassifier*)model, X, 0.5f);
}

static void linear_cls_free_adapter(void* model) {
    freeLinearClassifier((LinearClassifier*)model);
}

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
    SVMParams svm_params = {1.0f, 0.01f, 1000};
    Estimator svm_estimator = {
        .create = svm_create_adapter,
        .train = svm_train_adapter,
        .predict = svm_predict_adapter,
        .free_model = svm_free_adapter
    };
    float* scores = cross_val_score(&svm_estimator, X, y, &svm_params, cv_folds, 1, 42);

    float sum_scores = 0.0f;
    for (int i = 0; i < cv_folds; i++) {
        printf("  Fold %d Accuracy: %.2f%%\n", i + 1, scores[i] * 100);
        sum_scores += scores[i];
    }
    printf("\n  Mean Accuracy: %.2f%%\n", (sum_scores / cv_folds) * 100);

    // 3. Perform 5-Fold Cross-Validation for LinearClassifier
    printf("\nPerforming %d-Fold Cross-Validation for LinearClassifier (lr=0.1, iter=500, lambda=0.01)...\n", cv_folds);
    LinearClassifierParams linear_params = {0.1f, 500, 0.01f};
    Estimator linear_estimator = {
        .create = linear_cls_create_adapter,
        .train = linear_cls_train_adapter,
        .predict = linear_cls_predict_adapter,
        .free_model = linear_cls_free_adapter
    };
    float* scores_lin = cross_val_score(&linear_estimator, X, y, &linear_params, cv_folds, 1, 42);

    float sum_scores_lin = 0.0f;
    for (int i = 0; i < cv_folds; i++) {
        printf("  Fold %d Accuracy: %.2f%%\n", i + 1, scores_lin[i] * 100);
        sum_scores_lin += scores_lin[i];
    }
    printf("\n  Mean Accuracy: %.2f%%\n", (sum_scores_lin / cv_folds) * 100);

    // Cleanup
    pa_free(scores);
    pa_free(scores_lin);
    freeMatrix(X);
    freeMatrix(y);

    return 0;
}
