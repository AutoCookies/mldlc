#include <stdio.h>
#include <stdlib.h>
#include "../loader/dataloader.h"
#include "../objects/matrix.h"

void printMatrixSimple(const Matrix* m, const char* label) {
    printf("%s (%dx%d):\n", label, m->rows, m->cols);
    float* data = (float*)m->data;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.1f ", data[i * m->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("Testing train_test_split...\n\n");

    int n_samples = 10;
    int n_features = 2;
    Matrix* X = createMatrix(n_samples, n_features, DTYPE_FLOAT32);
    Matrix* y = createMatrix(n_samples, 1, DTYPE_FLOAT32);

    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    for (int i = 0; i < n_samples; i++) {
        x_data[i * 2] = (float)i;
        x_data[i * 2 + 1] = (float)i * 10.0f;
        y_data[i] = (float)i;
    }

    printMatrixSimple(X, "Original X");
    printMatrixSimple(y, "Original y");

    Matrix *X_train, *X_test, *y_train, *y_test;

    // Test 1: Split without shuffle
    printf("--- Test 1: Split without shuffle (test_size=0.2) ---\n");
    train_test_split(X, y, 0.2f, 0, 42, &X_train, &X_test, &y_train, &y_test);
    
    printMatrixSimple(X_train, "X_train");
    printMatrixSimple(X_test, "X_test");
    printMatrixSimple(y_train, "y_train");
    printMatrixSimple(y_test, "y_test");

    freeMatrix(X_train); freeMatrix(X_test); freeMatrix(y_train); freeMatrix(y_test);

    // Test 2: Split with shuffle
    printf("--- Test 2: Split with shuffle (test_size=0.3, seed=42) ---\n");
    train_test_split(X, y, 0.3f, 1, 42, &X_train, &X_test, &y_train, &y_test);

    printMatrixSimple(X_train, "X_train (shuffled)");
    printMatrixSimple(X_test, "X_test (shuffled)");
    printMatrixSimple(y_train, "y_train (shuffled)");
    printMatrixSimple(y_test, "y_test (shuffled)");

    freeMatrix(X_train); freeMatrix(X_test); freeMatrix(y_train); freeMatrix(y_test);

    freeMatrix(X); freeMatrix(y);

    return 0;
}
