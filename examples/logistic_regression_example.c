#include <stdio.h>
#include <stdlib.h>
#include "../objects/matrix.h"
#include "../calculations/metrics.h"
#include "../calculations/costFunction.h"
#include "../al/linearAl.h"

int main() {
    printf("Linear Classification Example (Logistic Regression)\n");
    printf("==================================================\n\n");

    // 1. Create a simple synthetic dataset
    // Features: [x1, x2]
    // Task: Classify points based on x1 + x2 > 0
    int n_samples = 6;
    int n_features = 2;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    // Sample 0: [-2, -2] -> 0
    x_data[0] = -2.0f; x_data[1] = -2.0f; y_data[0] = 0.0f;
    // Sample 1: [-1, -1] -> 0
    x_data[2] = -1.0f; x_data[3] = -1.0f; y_data[1] = 0.0f;
    // Sample 2: [0, -1]  -> 0
    x_data[4] = 0.0f;  x_data[5] = -1.0f; y_data[2] = 0.0f;
    // Sample 3: [1, 1]   -> 1
    x_data[6] = 1.0f;  x_data[7] = 1.0f;  y_data[3] = 1.0f;
    // Sample 4: [2, 2]   -> 1
    x_data[8] = 2.0f;  x_data[9] = 2.0f;  y_data[4] = 1.0f;
    // Sample 5: [0, 1]   -> 1
    x_data[10] = 0.0f; x_data[11] = 1.0f; y_data[5] = 1.0f;

    // 2. Initialize and Train Model
    float lr = 0.1f;
    int iter = 1000;
    LinearClassifier* model = createLinearClassifier(n_features, lr, iter, REG_NONE, 0.0f, 0.0f);

    printf("Initial Loss (BCE): %.6f\n", computeCost(COST_BINARY_CROSS_ENTROPY, y, predictLinearClassifier(model, X)));

    printf("Training for %d iterations...\n", iter);
    trainLinearClassifier(model, X, y);

    Matrix* final_probs = predictLinearClassifier(model, X);
    printf("Final Loss (BCE):   %.6f\n\n", computeCost(COST_BINARY_CROSS_ENTROPY, y, final_probs));

    // 3. Results and Evaluation
    printf("Learned Parameters:\n");
    float* w = (float*)model->weights->data;
    printf("  Weights: [%.4f, %.4f]\n", w[0], w[1]);
    printf("  Bias:    %.4f\n\n", model->bias);

    printf("Predictions:\n");
    Matrix* final_classes = predictLinearClassifierClass(model, X, 0.5f);
    float* p_data = (float*)final_probs->data;
    float* c_data = (float*)final_classes->data;

    for (int i = 0; i < n_samples; i++) {
        printf("  Sample %d: Features=[%.1f, %.1f], True=%.0f, Prob=%.4f, Class=%.0f\n",
               i, x_data[i*2], x_data[i*2+1], y_data[i], p_data[i], c_data[i]);
    }

    // Classification Report
    printf("\nClassification Report:\n");
    printf("  Accuracy:  %.4f\n", accuracyScore(y, final_classes));
    printf("  Precision: %.4f\n", precisionScore(y, final_classes));
    printf("  Recall:    %.4f\n", recallScore(y, final_classes));
    printf("  F1 Score:  %.4f\n", f1Score(y, final_classes));

    // Cleanup
    freeMatrix(X);
    freeMatrix(y);
    freeMatrix(final_probs);
    freeMatrix(final_classes);
    freeLinearClassifier(model);

    return 0;
}
