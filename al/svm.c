#include "svm.h"
#include <stdlib.h>
#include <stdio.h>
#include "palloc.h"

SVM* createSVM(int input_size, float C, float lr, int iter) {
    SVM* model = (SVM*)pa_malloc(sizeof(SVM));
    if (model == NULL) return NULL;

    model->weights = createMatrix(input_size, 1, DTYPE_FLOAT32);
    // Initialize weights to zero
    float* w_data = (float*)model->weights->data;
    for (int i = 0; i < input_size; i++) w_data[i] = 0.0f;

    model->bias = 0.0f;
    model->C = C;
    model->learning_rate = lr;
    model->iterations = iter;

    return model;
}

void trainSVM(SVM* model, const Matrix* X, const Matrix* y) {
    if (X->rows != y->rows) return;
    int n_samples = X->rows;
    int n_features = X->cols;
    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;
    float* w_data = (float*)model->weights->data;

    for (int iter = 0; iter < model->iterations; iter++) {
        for (int i = 0; i < n_samples; i++) {
            // Calculate decision function: f(x) = w^T * x + b
            float f_x = 0.0f;
            for (int j = 0; j < n_features; j++) {
                f_x += w_data[j] * x_data[i * n_features + j];
            }
            f_x += model->bias;

            // Update rule for Hinge Loss: max(0, 1 - y_i * f_x)
            float condition = y_data[i] * f_x;

            if (condition >= 1.0f) {
                // Correct classification, only penalize weights (L2)
                for (int j = 0; j < n_features; j++) {
                    w_data[j] -= model->learning_rate * (2.0f * (1.0f / model->iterations) * w_data[j]);
                }
            } else {
                // Misclassified or within margin, penalized both weights and bias
                for (int j = 0; j < n_features; j++) {
                    w_data[j] -= model->learning_rate * (2.0f * (1.0f / model->iterations) * w_data[j] - model->C * y_data[i] * x_data[i * n_features + j]);
                }
                model->bias += model->learning_rate * model->C * y_data[i];
            }
        }
    }
}

Matrix* predictSVM(const SVM* model, const Matrix* X) {
    Matrix* preds = createMatrix(X->rows, 1, DTYPE_FLOAT32);
    float* x_data = (float*)X->data;
    float* p_data = (float*)preds->data;
    float* w_data = (float*)model->weights->data;
    int n_features = X->cols;

    for (int i = 0; i < X->rows; i++) {
        float f_x = 0.0f;
        for (int j = 0; j < n_features; j++) {
            f_x += w_data[j] * x_data[i * n_features + j];
        }
        f_x += model->bias;
        p_data[i] = (f_x >= 0.0f) ? 1.0f : -1.0f;
    }

    return preds;
}

void freeSVM(SVM* model) {
    if (model == NULL) return;
    freeMatrix(model->weights);
    pa_free(model);
}
