#include <stdio.h>
#include "../objects/matrix.h"
#include "../calculations/matrixcalc.h"
#include <stdlib.h>
#include <stdint.h>

void printMatrix(const Matrix *m)
{
    if (m == NULL)
    {
        printf("Matrix is NULL\n");
        return;
    }

    printf("Matrix (%d x %d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            if (m->dtype == DTYPE_FLOAT32)
            {
                printf("%.2f ", ((float *)m->data)[i * m->cols + j]);
            }
            else if (m->dtype == DTYPE_INT32)
            {
                printf("%d ", ((int32_t *)m->data)[i * m->cols + j]);
            }
            else if (m->dtype == DTYPE_INT8)
            {
                printf("%d ", ((int8_t *)m->data)[i * m->cols + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    Matrix *m1 = createMatrix(2, 2, DTYPE_FLOAT32);
    Matrix *m2 = createMatrix(2, 2, DTYPE_FLOAT32);

    if (m1 == NULL || m2 == NULL)
        return 1;

    float *d1 = (float *)m1->data;
    float *d2 = (float *)m2->data;

    d1[0] = 1.0f;
    d1[1] = 2.0f;
    d1[2] = 3.0f;
    d1[3] = 4.0f;

    d2[0] = 5.0f;
    d2[1] = 6.0f;
    d2[2] = 7.0f;
    d2[3] = 8.0f;

    printf("Matrix 1:\n");
    printMatrix(m1);
    printf("Matrix 2:\n");
    printMatrix(m2);

    Matrix *m_sum = matrixAdd(m1, m2);
    printf("Sum (m1 + m2):\n");
    printMatrix(m_sum);

    Matrix *m_sub = matrixSubtract(m1, m2);
    printf("Subtraction (m1 - m2):\n");
    printMatrix(m_sub);

    Matrix *m_mul = matrixMultiplication(m1, m2);
    printf("Multiplication (m1 * m2):\n");
    printMatrix(m_mul);

    Matrix *m_div = matrixDivision(m1, m2);
    printf("Division (m1 / m2):\n");
    printMatrix(m_div);

    Matrix *m_trans = matrixTranspose(m1);
    printf("Transpose of m1:\n");
    printMatrix(m_trans);

    Matrix *m3 = createMatrix(2, 3, DTYPE_INT32);
    Matrix *m4 = createMatrix(3, 2, DTYPE_INT32);

    if (m3 != NULL && m4 != NULL)
    {
        int32_t *d3 = (int32_t *)m3->data;
        int32_t *d4 = (int32_t *)m4->data;

        for (int i = 0; i < 6; i++)
        {
            d3[i] = i + 1;
            d4[i] = i + 1;
        }

        printf("Matrix 3 (Int32):\n");
        printMatrix(m3);
        printf("Matrix 4 (Int32):\n");
        printMatrix(m4);

        Matrix *m_mul_int = matrixMultiplication(m3, m4);
        printf("Multiplication (m3 * m4) (Int32):\n");
        printMatrix(m_mul_int);

        if (m_mul_int != NULL)
        {
            freeMatrix(m_mul_int);
        }
    }

    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(m3);
    freeMatrix(m4);

    if (m_sum)
        freeMatrix(m_sum);
    if (m_sub)
        freeMatrix(m_sub);
    if (m_mul)
        freeMatrix(m_mul);
    if (m_div)
        freeMatrix(m_div);
    if (m_trans)
        freeMatrix(m_trans);

    return 0;
}