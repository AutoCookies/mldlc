#ifndef MATRIX_H
#define MATRIX_H

#include "dtype.h"

typedef struct {
    int rows;
    int cols;
    DataType dtype;
    void* data;
} Matrix;

Matrix* createMatrix(int rows, int cols, DataType dtype);
void freeMatrix(Matrix* m);

#endif
