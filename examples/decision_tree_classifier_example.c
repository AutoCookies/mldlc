#include <stdio.h>
#include <stdlib.h>
#include "../loader/dataloader.h"
#include "../al/decisionTree.h"
#include "../calculations/metrics.h"

int main() {
    printf("Decision Tree Classifier Example (Synthetic Numeric Data)\n");
    printf("========================================================\n\n");

    // 1. Create synthetic numeric dataset (2 features, 3 classes)
    int n_samples = 30;
    int n_features = 2;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_ptr = (float*)X->data;
    float* y_ptr = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        // Class 0: x1 < 2, x2 < 2
        // Class 1: x1 > 2, x2 < 2
        // Class 2: x2 > 2
        if (i < 10) {
            x_ptr[i*2] = 1.0f; x_ptr[i*2+1] = 1.0f; y_ptr[i] = 0.0f;
        } else if (i < 20) {
            x_ptr[i*2] = 3.0f; x_ptr[i*2+1] = 1.0f; y_ptr[i] = 1.0f;
        } else {
            x_ptr[i*2] = 2.0f; x_ptr[i*2+1] = 4.0f; y_ptr[i] = 2.0f;
        }
    }

    // 2. Train-Test Split
    Matrix *X_train, *X_test, *y_train, *y_test;
    train_test_split(X, y, 0.2f, 1, 42, &X_train, &X_test, &y_train, &y_test);

    // 3. Train Classifier
    // Using Gini criterion, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=-1, min_imp_dec=0.0, seed=42
    DecisionTree* clf = createDecisionTree(TREE_CLASSIFICATION, CRITERION_GINI, 5, 2, 1, -1, 0.0f, 42);
    printf("Training Decision Tree Classifier...\n");
    trainDecisionTree(clf, X_train, y_train);

    // 4. Evaluate
    Matrix* y_pred = predictDecisionTree(clf, X_test);
    
    printf("\nEvaluation Metrics:\n");
    printf("  Accuracy:  %.4f\n", accuracyScore(y_test, y_pred));
    printf("  F1 Score:  %.4f\n", f1Score(y_test, y_pred));

    // Sample predictions
    printf("\nSample Predictions (Test Set):\n");
    float* yp_data = (float*)y_pred->data;
    float* yt_data = (float*)y_test->data;
    for (int i = 0; i < (y_test->rows < 5 ? y_test->rows : 5); i++) {
        printf("  Sample %d: True=%.0f, Pred=%.0f\n", i, yt_data[i], yp_data[i]);
    }

    // Cleanup
    freeMatrix(X); freeMatrix(y);
    freeMatrix(X_train); freeMatrix(X_test); freeMatrix(y_train); freeMatrix(y_test);
    freeMatrix(y_pred);
    freeDecisionTree(clf);

    return 0;
}
