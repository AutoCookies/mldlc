#ifndef MATRIXCALC_H
#define MATRIXCALC_H

#include "../objects/matrix.h"
#include "../objects/dtype.h"

Matrix *matrixAdd(const Matrix *m1, const Matrix *m2);
Matrix *matrixSubtract(const Matrix *m1, const Matrix *m2);
Matrix *matrixMultiplication(const Matrix *m1, const Matrix *m2);
Matrix *matrixDivision(const Matrix *m1, const Matrix *m2);

void *getMatrixData(const Matrix *m);
DataType getMatrixType(const Matrix *m);

#endif