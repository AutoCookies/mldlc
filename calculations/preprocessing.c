#include "preprocessing.h"
#include <math.h>

void standardize(Matrix* X) {
    if (X->dtype != DTYPE_FLOAT32) return;

    int rows = X->rows;
    int cols = X->cols;
    float* data = (float*)X->data;

    for (int j = 0; j < cols; j++) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            sum += data[i * cols + j];
        }
        float mean = sum / rows;

        float sq_sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            float diff = data[i * cols + j] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / rows);

        for (int i = 0; i < rows; i++) {
            if (std_dev > 0.000001f) {
                data[i * cols + j] = (data[i * cols + j] - mean) / std_dev;
            }
        }
    }
}

void minMaxScale(Matrix* X) {
    if (X->dtype != DTYPE_FLOAT32) return;

    int rows = X->rows;
    int cols = X->cols;
    float* data = (float*)X->data;

    for (int j = 0; j < cols; j++) {
        float min_val = data[0 * cols + j];
        float max_val = data[0 * cols + j];

        for (int i = 1; i < rows; i++) {
            float val = data[i * cols + j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        float range = max_val - min_val;
        for (int i = 0; i < rows; i++) {
            if (range > 0.000001f) {
                data[i * cols + j] = (data[i * cols + j] - min_val) / range;
            }
        }
    }
}   