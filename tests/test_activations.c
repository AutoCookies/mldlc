#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../calculations/activations.h"
#include "../objects/matrix.h"

void printMatrixSimple(const Matrix* m) {
    float* data = (float*)m->data;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.4f ", data[i * m->cols + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("Testing Sigmoid Activations...\n\n");

    // 1. Test scalar sigmoid
    printf("Scalar Tests:\n");
    printf("  sigmoid(0.0)  = %.4f (Expected: 0.5000)\n", sigmoid(0.0f));
    printf("  sigmoid(2.0)  = %.4f (Expected: 0.8808)\n", sigmoid(2.0f));
    printf("  sigmoid(-2.0) = %.4f (Expected: 0.1192)\n", sigmoid(-2.0f));

    // 2. Test matrix sigmoid
    printf("\nMatrix Tests:\n");
    Matrix* m = createMatrix(1, 3, DTYPE_FLOAT32);
    float* d = (float*)m->data;
    d[0] = -2.0f; d[1] = 0.0f; d[2] = 2.0f;

    printf("Input Matrix:\n");
    printMatrixSimple(m);

    Matrix* sig_m = matrixSigmoid(m);
    printf("After matrixSigmoid:\n");
    printMatrixSimple(sig_m);

    matrixSigmoidInplace(m);
    printf("After matrixSigmoidInplace:\n");
    printMatrixSimple(m);

    freeMatrix(m);
    freeMatrix(sig_m);

    return 0;
}
