#include "matrix.h"
#include <stdlib.h>
#include <stdint.h>

Matrix* createMatrix(int rows, int cols, DataType dtype) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix == NULL) return NULL;

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->dtype = dtype;

    size_t total_elements = (size_t)rows * cols;

    size_t element_size = 0;
    switch (dtype) {
        case DTYPE_FLOAT32: element_size = sizeof(float); break;
        case DTYPE_INT32:   element_size = sizeof(int32_t); break;
        case DTYPE_INT8:    element_size = sizeof(int8_t); break;
    }

    matrix->data = malloc(total_elements * element_size);

    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }

    return matrix;
}

void freeMatrix(Matrix* matrix) {
    if (matrix == NULL) return;
    if (matrix->data != NULL) {
        free(matrix->data);
    }
    free(matrix);
}
