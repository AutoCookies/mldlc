#include <stdio.h>
#include <stdlib.h>
#include "../objects/matrix.h"
#include "../calculations/matrixcalc.h"
#include "../calculations/metrics.h"
#include "../calculations/preprocessing.h"
#include "../calculations/costFunction.h"
#include "../al/linearRegression.h"
#include "../loader/dataloader.h"

int main() {
    // 1. Load data from CSV
    const char* filename = "data/IRIS.csv";
    printf("Loading dataset from %s...\n", filename);
    Matrix* full_data = loadCSV(filename, 1);
    if (full_data == NULL) {
        printf("Failed to load data\n");
        return 1;
    }

    int n_samples = full_data->rows;
    int n_all_cols = full_data->cols;
    printf("Loaded %d samples with %d columns\n", n_samples, n_all_cols);

    // 2. Split into Features (X) and Target (y)
    // Features: sepal_length, sepal_width, petal_length (cols 0, 1, 2)
    // Target: petal_width (col 3)
    int n_features = 3;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* full_ptr = (float*)full_data->data;
    float* x_ptr = (float*)X->data;
    float* y_ptr = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            x_ptr[i * n_features + j] = full_ptr[i * n_all_cols + j];
        }
        y_ptr[i] = full_ptr[i * n_all_cols + 3];
    }

    // 3. Preprocess Features (Standardization)
    printf("Standardizing features...\n");
    standardize(X);

    // 4. Initialize and Train Model
    float learning_rate = 0.1f;
    int iterations = 1000;
    float lambda = 0.001f; // Regularization strength
    printf("Training Linear Regression with L2 Regularization (LR = %.2f, Iterations = %d, Lambda = %.4f)...\n", 
           learning_rate, iterations, lambda);
    LinearRegression* model = createLinearRegression(n_features, learning_rate, iterations, REG_L2, lambda, 0.0f);
    
    Matrix* y_pred_initial = predictLinearRegression(model, X);
    float initial_cost = computeCost(COST_MSE, y, y_pred_initial);
    printf("Initial Cost (MSE): %.6f\n", initial_cost);
    freeMatrix(y_pred_initial);

    trainLinearRegression(model, X, y);

    Matrix* y_pred_final = predictLinearRegression(model, X);
    float final_cost = computeCost(COST_MSE, y, y_pred_final);
    printf("Final Cost (MSE):   %.6f\n\n", final_cost);

    // 5. Output Results
    printf("Learned Parameters:\n");
    float* weights = (float*)model->weights->data;
    printf("  Weights: [%.4f, %.4f, %.4f]\n", weights[0], weights[1], weights[2]);
    printf("  Bias:    %.4f\n\n", model->bias);

    // 6. Predict all
    Matrix* y_pred = y_pred_final;
    float* yp_ptr = (float*)y_pred->data;

    // 7. Detailed Arithmetic Breakdown (Sample 0)
    printf("--- Detailed Calculation for Sample 0 ---\n");
    float* sample0_x = &x_ptr[0]; // First sample features (standardized)
    float sample0_y_true = y_ptr[0];
    float sample0_y_pred = yp_ptr[0];

    printf("Standardized Features (X): [%.4f, %.4f, %.4f]\n", sample0_x[0], sample0_x[1], sample0_x[2]);
    printf("Learned Weights (W):      [%.4f, %.4f, %.4f]\n", weights[0], weights[1], weights[2]);
    printf("Learned Bias (b):         %.4f\n", model->bias);

    printf("\nCalculation steps (y = x1*w1 + x2*w2 + x3*w3 + b):\n");
    float step1 = sample0_x[0] * weights[0];
    float step2 = sample0_x[1] * weights[1];
    float step3 = sample0_x[2] * weights[2];
    float manual_sum = step1 + step2 + step3 + model->bias;

    printf("  (%.4f * %.4f) + (%.4f * %.4f) + (%.4f * %.4f) + %.4f\n", 
           sample0_x[0], weights[0], sample0_x[1], weights[1], sample0_x[2], weights[2], model->bias);
    printf("  = %.4f + %.4f + %.4f + %.4f\n", step1, step2, step3, model->bias);
    printf("  = %.4f\n", manual_sum);
    
    printf("\nVerification:\n");
    printf("  Manual Sum:       %.4f\n", manual_sum);
    printf("  Model Prediction: %.4f\n", sample0_y_pred);
    printf("  True Label (y):   %.4f\n\n", sample0_y_true);

    // 8. Evaluate
    printf("Overall Evaluation Metrics:\n");
    float mse = computeCost(COST_MSE, y, y_pred);
    float r2 = rSquared(y, y_pred);
    float adj_r2 = adjustedRSquared(r2, n_samples, n_features);
    printf("  Mean Squared Error: %.6f\n", mse);
    printf("  R-Squared:          %.6f\n", r2);
    printf("  Adjusted R-Squared: %.6f\n", adj_r2);

    // Sample Predictions (first 5)
    printf("\nPredictions Subset (First 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("  Sample %d: True = %.2f, Pred = %.4f\n", i, y_ptr[i], yp_ptr[i]);
    }

    // Cleanup
    freeMatrix(full_data);
    freeMatrix(X);
    freeMatrix(y);
    freeMatrix(y_pred);
    freeLinearRegression(model);

    return 0;
}
