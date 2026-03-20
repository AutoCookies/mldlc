#include "logRegression.h"
#include "../calculations/matrixcalc.h"
#include "../calculations/activations.h"
#include "../calculations/metrics.h"
#include <stdlib.h>
#include <stdio.h>
#include "palloc.h"

LogisticRegression* createLogisticRegression(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha) {
    LogisticRegression* model = (LogisticRegression*)pa_malloc(sizeof(LogisticRegression));
    if (model == NULL) return NULL;

    model->weights = createMatrix(input_size, 1, DTYPE_FLOAT32);
    if (model->weights == NULL) {
        pa_free(model);
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

void freeLogisticRegression(LogisticRegression* model) {
    if (model == NULL) return;
    if (model->weights != NULL) freeMatrix(model->weights);
    pa_free(model);
}

Matrix* predictLogisticRegression(const LogisticRegression* model, const Matrix* X) {
    // z = XW + b
    Matrix* product = matrixMultiplication(X, model->weights);
    if (product == NULL) return NULL;

    float* data = (float*)product->data;
    int total = product->rows * product->cols;
    for (int i = 0; i < total; i++) {
        data[i] += model->bias;
    }

    // p = sigmoid(z)
    matrixSigmoidInplace(product);

    return product;
}

Matrix* predictLogisticRegressionClass(const LogisticRegression* model, const Matrix* X, float threshold) {
    Matrix* probs = predictLogisticRegression(model, X);
    if (probs == NULL) return NULL;

    int total = probs->rows * probs->cols;
    float* data = (float*)probs->data;
    for (int i = 0; i < total; i++) {
        data[i] = (data[i] >= threshold) ? 1.0f : 0.0f;
    }

    return probs;
}

void gradientDescentStepLog(LogisticRegression *model, const Matrix *X, const Matrix *y) {
    int n = X->rows;
    Matrix* p = predictLogisticRegression(model, X);
    Matrix* error = matrixSubtract(p, y); // p - y
    Matrix* XT = matrixTranspose(X);
    Matrix* grad_w = matrixMultiplication(XT, error);

    float factor = (1.0f / n) * model->learning_rate;

    // Apply regularization penalty to gradient
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

    freeMatrix(p);
    freeMatrix(error);
    freeMatrix(XT);
    freeMatrix(grad_w);
}

void trainLogisticRegression(LogisticRegression* model, const Matrix* X, const Matrix* y) {
    for (int i = 0; i < model->iterations; i++) {
        gradientDescentStepLog(model, X, y);
    }
}
