#include <stdio.h>
#include <stdlib.h>
#include "../al/linearRegression.h"
#include "../calculations/metrics.h"
#include "../objects/matrix.h"

void testSimpleLinearRegression() {
    printf("--- Simple Linear Regression (y = 2x + 1) ---\n");
    // 1. Create synthetic dataset for y = 2x + 1
    int n_samples = 5;
    int n_features = 1;

    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        x_data[i] = (float)(i + 1);
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }

    // 2. Initialize Model
    float learning_rate = 0.01f;
    int iterations = 1000;
    LinearRegression* model = createLinearRegression(n_features, learning_rate, iterations, REG_NONE, 0.0f, 0.0f);
    
    // 3. Train Model
    printf("Training model for %d iterations...\n", iterations);
    trainLinearRegression(model, X, y);

    // 4. Print Results
    float weight = ((float*)model->weights->data)[0];
    float bias = model->bias;
    printf("Learned parameters:\n");
    printf("Weight: %.4f (Expected: 2.0)\n", weight);
    printf("Bias:   %.4f (Expected: 1.0)\n", bias);

    // 5. Predict and Evaluate
    Matrix* y_pred = predictLinearRegression(model, X);
    float mse = meanSquaredError(y, y_pred);
    printf("Mean Squared Error: %.6f\n\n", mse);

    // Cleanup
    freeMatrix(X);
    freeMatrix(y);
    freeMatrix(y_pred);
    freeLinearRegression(model);
}

void testMultipleLinearRegression() {
    printf("--- Multiple Linear Regression (y = 2x1 + 3x2 + 5) ---\n");
    // 1. Create synthetic dataset
    // x1 x2 | y
    // 1  1  | 10
    // 2  1  | 12
    // 1  2  | 13
    // 2  2  | 15
    // 3  3  | 20
    int n_samples = 5;
    int n_features = 2;

    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    // Sample 0: (1, 1) -> 10
    x_data[0] = 1.0f; x_data[1] = 1.0f; y_data[0] = 10.0f;
    // Sample 1: (2, 1) -> 12
    x_data[2] = 2.0f; x_data[3] = 1.0f; y_data[1] = 12.0f;
    // Sample 2: (1, 2) -> 13
    x_data[4] = 1.0f; x_data[5] = 2.0f; y_data[2] = 13.0f;
    // Sample 3: (2, 2) -> 15
    x_data[6] = 2.0f; x_data[7] = 2.0f; y_data[3] = 15.0f;
    // Sample 4: (3, 3) -> 20
    x_data[8] = 3.0f; x_data[9] = 3.0f; y_data[4] = 20.0f;

    // 2. Initialize Model
    float learning_rate = 0.01f;
    int iterations = 2000; // More features might need more iterations
    LinearRegression* model = createLinearRegression(n_features, learning_rate, iterations, REG_NONE, 0.0f, 0.0f);

    // 3. Train Model
    printf("Training model for %d iterations...\n", iterations);
    trainLinearRegression(model, X, y);

    // 4. Print Results
    float w1 = ((float*)model->weights->data)[0];
    float w2 = ((float*)model->weights->data)[1];
    float bias = model->bias;
    printf("Learned parameters:\n");
    printf("Weight 1: %.4f (Expected: 2.0)\n", w1);
    printf("Weight 2: %.4f (Expected: 3.0)\n", w2);
    printf("Bias:     %.4f (Expected: 5.0)\n", bias);

    // 5. Predict and Evaluate
    Matrix* y_pred = predictLinearRegression(model, X);
    float mse = meanSquaredError(y, y_pred);
    printf("Mean Squared Error: %.6f\n", mse);

    // 6. Predict for x = (10, 5) -> y = 2(10) + 3(5) + 5 = 40
    Matrix* X_new = createMatrix(1, 2, DTYPE_FLOAT32);
    ((float*)X_new->data)[0] = 10.0f;
    ((float*)X_new->data)[1] = 5.0f;
    Matrix* y_new_pred = predictLinearRegression(model, X_new);
    printf("Prediction for (10.0, 5.0): %.4f (Expected: 40.0)\n\n", ((float*)y_new_pred->data)[0]);

    // Cleanup
    freeMatrix(X);
    freeMatrix(y);
    freeMatrix(y_pred);
    freeMatrix(X_new);
    freeMatrix(y_new_pred);
    freeLinearRegression(model);
}

int main() {
    testSimpleLinearRegression();
    testMultipleLinearRegression();
    return 0;
}
