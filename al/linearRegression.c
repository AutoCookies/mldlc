#include "linearRegression.h"
#include "../calculations/matrixcalc.h"
#include "../calculations/metrics.h"
#include <stdlib.h>
#include <stdio.h>

LinearRegression* createLinearRegression(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha) {
    LinearRegression* model = (LinearRegression*)malloc(sizeof(LinearRegression));
    if (model == NULL) return NULL;

    model->weights = createMatrix(input_size, 1, DTYPE_FLOAT32);
    if (model->weights == NULL) {
        free(model);
        return NULL;
    }

    float* w_data = (float*)model->weights->data;
    for (int i = 0; i < input_size; i++) {
        w_data[i] = 0.0f;
    }

    model->bias = 0.0f;
    model->learning_rate = lr;
    model->iterations = iter;
    model->reg_type = reg_type;
    model->lambda = lambda;
    model->alpha = alpha;

    return model;
}

void freeLinearRegression(LinearRegression* model) {
    if (model == NULL) return;
    if (model->weights != NULL) freeMatrix(model->weights);
    free(model);
}

Matrix* predictLinearRegression(const LinearRegression* model, const Matrix* X) {
    Matrix* product = matrixMultiplication(X, model->weights);
    if (product == NULL) return NULL;

    float* data = (float*)product->data;
    int total = product->rows * product->cols;
    for (int i = 0; i < total; i++) {
        data[i] += model->bias;
    }

    return product;
}

void gradientDescentStep(LinearRegression *model, const Matrix *X, const Matrix *y) {
    int n = X->rows;
    Matrix* y_pred = predictLinearRegression(model, X);
    Matrix* error = matrixSubtract(y_pred, y);
    Matrix* XT = matrixTranspose(X);
    Matrix* grad_w = matrixMultiplication(XT, error);

    float factor = (2.0f / n) * model->learning_rate;

    // Apply main gradient and regularization penalty
    applyRegularizationGradient(model->reg_type, model->lambda, model->alpha, model->weights, grad_w);

    float* w_data = (float*)model->weights->data;
    float* gw_data = (float*)grad_w->data;
    for (int i = 0; i < model->weights->rows; i++) {
        w_data[i] -= factor * gw_data[i];
    }

    float* err_data = (float*)error->data;
    float sum_err = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_err += err_data[i];
    }
    model->bias -= factor * sum_err;

    freeMatrix(y_pred);
    freeMatrix(error);
    freeMatrix(XT);
    freeMatrix(grad_w);
}

void trainLinearRegression(LinearRegression* model, const Matrix* X, const Matrix* y) {
    for (int i = 0; i < model->iterations; i++) {
        gradientDescentStep(model, X, y);
    }
}