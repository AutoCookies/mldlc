#include <stdio.h>
#include <stdlib.h>
#include "model_selection.h"
#include "svm.h"

typedef struct {
    float C;
    float lr;
    int iter;
} SVMParams;

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
    SVMParams p1 = {0.1f, 0.001f, 1000};
    SVMParams p2 = {0.1f, 0.01f, 2000};
    SVMParams p3 = {1.0f, 0.001f, 1000};
    SVMParams p4 = {1.0f, 0.01f, 2000};
    SVMParams p5 = {10.0f, 0.001f, 1000};
    SVMParams p6 = {10.0f, 0.01f, 2000};
    void* params_grid[] = {&p1, &p2, &p3, &p4, &p5, &p6};
    int n_candidates = 6;

    // 3. Run GridSearch with 5-fold CV
    printf("Running Grid Search with 5-fold cross-validation...\n");
    Estimator svm_estimator = {
        .create = svm_create_adapter,
        .train = svm_train_adapter,
        .predict = svm_predict_adapter,
        .free_model = svm_free_adapter
    };
    GridSearchResult result = grid_search(&svm_estimator, X, y, params_grid, n_candidates, 5);
    SVMParams* best = (SVMParams*)result.best_params;

    printf("\nBest Parameters Found:\n");
    printf("  C:             %.2f\n", best->C);
    printf("  Learning Rate: %.4f\n", best->lr);
    printf("  Iterations:    %d\n", best->iter);
    printf("  Best Score:    %.2f%%\n", result.best_score * 100);

    // Cleanup
    freeMatrix(X);
    freeMatrix(y);

    return 0;
}
