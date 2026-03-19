#include "activations.h"
#include <math.h>
#include <stdlib.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

Matrix* matrixSigmoid(const Matrix* m) {
    if (m == NULL || m->dtype != DTYPE_FLOAT32) return NULL;

    Matrix* result = createMatrix(m->rows, m->cols, m->dtype);
    if (result == NULL) return NULL;

    int total = m->rows * m->cols;
    float* src_data = (float*)m->data;
    float* dst_data = (float*)result->data;

    for (int i = 0; i < total; i++) {
        dst_data[i] = sigmoid(src_data[i]);
    }

    return result;
}

void matrixSigmoidInplace(Matrix* m) {
    if (m == NULL || m->dtype != DTYPE_FLOAT32) return;

    int total = m->rows * m->cols;
    float* data = (float*)m->data;

    for (int i = 0; i < total; i++) {
        data[i] = sigmoid(data[i]);
    }
}
