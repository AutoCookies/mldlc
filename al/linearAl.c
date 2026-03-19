#include "linearAl.h"
#include "../calculations/matrixcalc.h"
#include "../calculations/activations.h"
#include "../calculations/metrics.h"
#include <stdlib.h>
#include <stdio.h>

/* --- Helper for Initialization --- */
static void initializeWeights(Matrix* weights) {
    float* w_data = (float*)weights->data;
    for (int i = 0; i < weights->rows * weights->cols; i++) {
        w_data[i] = 0.0f;
    }
}

/* --- Linear Regressor Implementation --- */

LinearRegressor* createLinearRegressor(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha) {
    LinearRegressor* model = (LinearRegressor*)malloc(sizeof(LinearRegressor));
    if (!model) return NULL;
    model->weights = createMatrix(input_size, 1, DTYPE_FLOAT32);
    initializeWeights(model->weights);
    model->bias = 0.0f;
    model->learning_rate = lr;
    model->iterations = iter;
    model->reg_type = reg_type;
    model->lambda = lambda;
    model->alpha = alpha;
    return model;
}

void freeLinearRegressor(LinearRegressor* model) {
    if (model) {
        freeMatrix(model->weights);
        free(model);
    }
}

Matrix* predictLinearRegressor(const LinearRegressor* model, const Matrix* X) {
    Matrix* res = matrixMultiplication(X, model->weights);
    if (!res) return NULL;
    float* data = (float*)res->data;
    for (int i = 0; i < res->rows * res->cols; i++) data[i] += model->bias;
    return res;
}

void trainLinearRegressor(LinearRegressor* model, const Matrix* X, const Matrix* y) {
    int n = X->rows;
    float factor = (2.0f / n) * model->learning_rate;
    Matrix* XT = matrixTranspose(X);

    for (int i = 0; i < model->iterations; i++) {
        Matrix* y_pred = predictLinearRegressor(model, X);
        Matrix* error = matrixSubtract(y_pred, y);
        Matrix* grad_w = matrixMultiplication(XT, error);

        applyRegularizationGradient(model->reg_type, model->lambda, model->alpha, model->weights, grad_w);

        float* w_data = (float*)model->weights->data;
        float* gw_data = (float*)grad_w->data;
        for (int j = 0; j < model->weights->rows; j++) w_data[j] -= factor * gw_data[j];

        float* err_data = (float*)error->data;
        float sum_err = 0.0f;
        for (int j = 0; j < n; j++) sum_err += err_data[j];
        model->bias -= factor * sum_err;

        freeMatrix(y_pred);
        freeMatrix(error);
        freeMatrix(grad_w);
    }
    freeMatrix(XT);
}

/* --- Linear Classifier Implementation --- */

LinearClassifier* createLinearClassifier(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha) {
    LinearClassifier* model = (LinearClassifier*)malloc(sizeof(LinearClassifier));
    if (!model) return NULL;
    model->weights = createMatrix(input_size, 1, DTYPE_FLOAT32);
    initializeWeights(model->weights);
    model->bias = 0.0f;
    model->learning_rate = lr;
    model->iterations = iter;
    model->reg_type = reg_type;
    model->lambda = lambda;
    model->alpha = alpha;
    return model;
}

void freeLinearClassifier(LinearClassifier* model) {
    if (model) {
        freeMatrix(model->weights);
        free(model);
    }
}

Matrix* predictLinearClassifier(const LinearClassifier* model, const Matrix* X) {
    Matrix* res = matrixMultiplication(X, model->weights);
    if (!res) return NULL;
    float* data = (float*)res->data;
    for (int i = 0; i < res->rows * res->cols; i++) data[i] += model->bias;
    matrixSigmoidInplace(res);
    return res;
}

Matrix* predictLinearClassifierClass(const LinearClassifier* model, const Matrix* X, float threshold) {
    Matrix* probs = predictLinearClassifier(model, X);
    if (!probs) return NULL;
    float* data = (float*)probs->data;
    for (int i = 0; i < probs->rows * probs->cols; i++) data[i] = (data[i] >= threshold) ? 1.0f : 0.0f;
    return probs;
}

void trainLinearClassifier(LinearClassifier* model, const Matrix* X, const Matrix* y) {
    int n = X->rows;
    float factor = (1.0f / n) * model->learning_rate;
    Matrix* XT = matrixTranspose(X);

    for (int i = 0; i < model->iterations; i++) {
        Matrix* p = predictLinearClassifier(model, X);
        Matrix* error = matrixSubtract(p, y);
        Matrix* grad_w = matrixMultiplication(XT, error);

        applyRegularizationGradient(model->reg_type, model->lambda, model->alpha, model->weights, grad_w);

        float* w_data = (float*)model->weights->data;
        float* gw_data = (float*)grad_w->data;
        for (int j = 0; j < model->weights->rows; j++) w_data[j] -= factor * gw_data[j];

        float* err_data = (float*)error->data;
        float sum_err = 0.0f;
        for (int j = 0; j < n; j++) sum_err += err_data[j];
        model->bias -= factor * sum_err;

        freeMatrix(p);
        freeMatrix(error);
        freeMatrix(grad_w);
    }
    freeMatrix(XT);
}
