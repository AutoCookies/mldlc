#include <stdio.h>
#include <stdlib.h>
#include "palloc.h"
#include "model_selection.h"
#include "dataloader.h"
#include "svm.h"
#include "linearAl.h"

/* --- SVM Adapter --- */
typedef struct {
    float C;
    float lr;
    int iter;
} SVMParams;

void* svm_create_adapter(int n_features, void* params) {
    SVMParams* p = (SVMParams*)params;
    return createSVM(n_features, p->C, p->lr, p->iter);
}

void svm_train_adapter(void* model, const Matrix* X, const Matrix* y) {
    trainSVM((SVM*)model, X, y);
}

Matrix* svm_predict_adapter(void* model, const Matrix* X) {
    return predictSVM((SVM*)model, X);
}

void svm_free_adapter(void* model) {
    freeSVM((SVM*)model);
}

/* --- Linear Classifier Adapter --- */
typedef struct {
    float lr;
    int iter;
    float lambda;
} LinearParams;

void* lin_create_adapter(int n_features, void* params) {
    LinearParams* p = (LinearParams*)params;
    return createLinearClassifier(n_features, p->lr, p->iter, REG_L2, p->lambda, 0.0f);
}

void lin_train_adapter(void* model, const Matrix* X, const Matrix* y) {
    trainLinearClassifier((LinearClassifier*)model, X, y);
}

Matrix* lin_predict_adapter(void* model, const Matrix* X) {
    return predictLinearClassifierClass((LinearClassifier*)model, X, 0.5f);
}

void lin_free_adapter(void* model) {
    freeLinearClassifier((LinearClassifier*)model);
}

int main() {
    printf("Adaptive GridSearchCV Example\n");
    printf("=============================\n\n");

    // 1. Dataset Generation
    int n_samples = 60, n_features = 2;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);
    for (int i = 0; i < n_samples; i++) {
        float x1 = (float)rand() / RAND_MAX * 10 - 5;
        float x2 = (float)rand() / RAND_MAX * 10 - 5;
        ((float*)X->data)[i*2] = x1;
        ((float*)X->data)[i*2+1] = x2;
        ((float*)y->data)[i] = (x1 + x2 > 0) ? 1.0f : -1.0f;
    }

    // 2. Setup Generic SVM Estimator
    Estimator svm_est = {
        .create = svm_create_adapter,
        .train = svm_train_adapter,
        .predict = svm_predict_adapter,
        .free_model = svm_free_adapter
    };

    SVMParams svm_p1 = {1.0f, 0.01f, 1000};
    SVMParams svm_p2 = {10.0f, 0.01f, 1000};
    void* svm_grid[] = {&svm_p1, &svm_p2};

    printf("Running Generic Grid Search for SVM...\n");
    GridSearchResult res_svm = grid_search(&svm_est, X, y, svm_grid, 2, 3);
    printf("Best SVM score: %.2f%% (C = %.1f)\n", res_svm.best_score * 100, ((SVMParams*)res_svm.best_params)->C);

    // 3. Setup Generic Linear Estimator
    Estimator lin_est = {
        .create = lin_create_adapter,
        .train = lin_train_adapter,
        .predict = lin_predict_adapter,
        .free_model = lin_free_adapter
    };

    LinearParams lin_p1 = {0.1f, 500, 0.001f};
    LinearParams lin_p2 = {0.01f, 1000, 0.01f};
    void* lin_grid[] = {&lin_p1, &lin_p2};

    // For linear classifier, map targets from {-1, 1} to {0, 1} as it expects logistic inputs
    for(int i=0; i<n_samples; i++) if (((float*)y->data)[i] == -1.0f) ((float*)y->data)[i] = 0.0f;

    printf("\nRunning Generic Grid Search for Linear Classifier...\n");
    GridSearchResult res_lin = grid_search(&lin_est, X, y, lin_grid, 2, 3);
    printf("Best Linear score: %.2f%% (lr = %.2f)\n", res_lin.best_score * 100, ((LinearParams*)res_lin.best_params)->lr);

    freeMatrix(X); freeMatrix(y);
    return 0;
}
