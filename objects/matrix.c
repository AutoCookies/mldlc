#include "matrix.h"
#include "palloc.h"
#include <stdint.h>

Matrix *createMatrix(int rows, int cols, DataType dtype)
{
    Matrix *matrix = (Matrix *)pa_malloc(sizeof(Matrix));
    if (matrix == NULL)
        return NULL;

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->dtype = dtype;

    size_t total_elements = (size_t)rows * cols;

    size_t element_size = 0;
    switch (dtype)
    {
    case DTYPE_FLOAT32:
        element_size = sizeof(float);
        break;
    case DTYPE_INT32:
        element_size = sizeof(int32_t);
        break;
    case DTYPE_INT8:
        element_size = sizeof(int8_t);
        break;
    }

    matrix->data = pa_malloc(total_elements * element_size);

    if (matrix->data == NULL)
    {
        pa_free(matrix);
        return NULL;
    }

    return matrix;
}

Matrix *matrixTranspose(const Matrix *m)
{
    if (m == NULL)
        return NULL;

    Matrix *result = createMatrix(m->cols, m->rows, m->dtype);
    if (result == NULL)
        return NULL;

    int R = m->rows;
    int C = m->cols;

    if (m->dtype == DTYPE_FLOAT32)
    {
        float *src = (float *)m->data;
        float *dst = (float *)result->data;

        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                dst[j * R + i] = src[i * C + j];
            }
        }
    }
    else if (m->dtype == DTYPE_INT32)
    {
        int32_t *src = (int32_t *)m->data;
        int32_t *dst = (int32_t *)result->data;

        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                dst[j * R + i] = src[i * C + j];
            }
        }
    }
    else if (m->dtype == DTYPE_INT8)
    {
        int8_t *src = (int8_t *)m->data;
        int8_t *dst = (int8_t *)result->data;

        for (int i = 0; i < R; i++)
        {
            for (int j = 0; j < C; j++)
            {
                dst[j * R + i] = src[i * C + j];
            }
        }
    }

    return result;
}

void freeMatrix(Matrix *matrix)
{
    if (matrix == NULL)
        return;
    if (matrix->data != NULL)
    {
        pa_free(matrix->data);
    }
    pa_free(matrix);
}
