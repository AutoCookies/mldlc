#include "matrixcalc.h"
#include <stdlib.h>
#include <stdint.h>

Matrix *matrixAdd(const Matrix *m1, const Matrix *m2)
{
    if ((m1->cols != m2->cols) || (m1->rows != m2->rows) || (m1->dtype != m2->dtype))
        return NULL;

    Matrix *result = createMatrix(m1->rows, m1->cols, m1->dtype);
    if (result == NULL)
        return NULL;

    int total_elements = m1->rows * m1->cols;

    if (m1->dtype == DTYPE_FLOAT32)
    {
        float *d1 = (float *)m1->data;
        float *d2 = (float *)m2->data;
        float *res_data = (float *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            res_data[i] = d1[i] + d2[i];
        }
    }
    else if (m1->dtype == DTYPE_INT32)
    {
        int32_t *d1 = (int32_t *)m1->data;
        int32_t *d2 = (int32_t *)m2->data;
        int32_t *res_data = (int32_t *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            res_data[i] = d1[i] + d2[i];
        }
    }
    else if (m1->dtype == DTYPE_INT8)
    {
        int8_t *d1 = (int8_t *)m1->data;
        int8_t *d2 = (int8_t *)m2->data;
        int8_t *res_data = (int8_t *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            res_data[i] = d1[i] + d2[i];
        }
    }

    return result;
}

Matrix *matrixSubtract(const Matrix *m1, const Matrix *m2)
{
    if ((m1->cols != m2->cols) || (m1->rows != m2->rows) || (m1->dtype != m2->dtype))
        return NULL;

    Matrix *result = createMatrix(m1->rows, m1->cols, m1->dtype);
    if (result == NULL)
        return NULL;

    int total_elements = m1->rows * m1->cols;

    if (m1->dtype == DTYPE_FLOAT32)
    {
        float *d1 = (float *)m1->data;
        float *d2 = (float *)m2->data;
        float *res_data = (float *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            res_data[i] = d1[i] - d2[i];
        }
    }
    else if (m1->dtype == DTYPE_INT32)
    {
        int32_t *d1 = (int32_t *)m1->data;
        int32_t *d2 = (int32_t *)m2->data;
        int32_t *res_data = (int32_t *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            res_data[i] = d1[i] - d2[i];
        }
    }
    else if (m1->dtype == DTYPE_INT8)
    {
        int8_t *d1 = (int8_t *)m1->data;
        int8_t *d2 = (int8_t *)m2->data;
        int8_t *res_data = (int8_t *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            res_data[i] = d1[i] - d2[i];
        }
    }

    return result;
}

Matrix *matrixMultiplication(const Matrix *m1, const Matrix *m2)
{
    if ((m1->cols != m2->rows) || (m1->dtype != m2->dtype))
        return NULL;

    Matrix *result = createMatrix(m1->rows, m2->cols, m1->dtype);
    if (result == NULL)
        return NULL;

    int M = m1->rows;
    int K = m1->cols;
    int N = m2->cols;

    if (m1->dtype == DTYPE_FLOAT32)
    {
        float *d1 = (float *)m1->data;
        float *d2 = (float *)m2->data;
        float *res_data = (float *)result->data;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    sum += d1[i * K + k] * d2[k * N + j];
                }
                res_data[i * N + j] = sum;
            }
        }
    }
    else if (m1->dtype == DTYPE_INT32)
    {
        int32_t *d1 = (int32_t *)m1->data;
        int32_t *d2 = (int32_t *)m2->data;
        int32_t *res_data = (int32_t *)result->data;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int32_t sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += d1[i * K + k] * d2[k * N + j];
                }
                res_data[i * N + j] = sum;
            }
        }
    }

    return result;
}

Matrix *matrixDivision(const Matrix *m1, const Matrix *m2)
{
    if ((m1->rows != m2->rows) || (m1->cols != m2->cols) || (m1->dtype != m2->dtype))
        return NULL;

    Matrix *result = createMatrix(m1->rows, m1->cols, m1->dtype);
    if (result == NULL)
        return NULL;

    int total_elements = m1->rows * m1->cols;

    if (m1->dtype == DTYPE_FLOAT32)
    {
        float *d1 = (float *)m1->data;
        float *d2 = (float *)m2->data;
        float *res_data = (float *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            if (d2[i] != 0.0f)
            {
                res_data[i] = d1[i] / d2[i];
            }
            else
            {
                res_data[i] = 0.0f; // Hoặc xử lý NAN/INF tùy bạn
            }
        }
    }
    else if (m1->dtype == DTYPE_INT32)
    {
        int32_t *d1 = (int32_t *)m1->data;
        int32_t *d2 = (int32_t *)m2->data;
        int32_t *res_data = (int32_t *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            if (d2[i] != 0)
            {
                res_data[i] = d1[i] / d2[i];
            }
            else
            {
                res_data[i] = 0;
            }
        }
    }
    else if (m1->dtype == DTYPE_INT8)
    {
        int8_t *d1 = (int8_t *)m1->data;
        int8_t *d2 = (int8_t *)m2->data;
        int8_t *res_data = (int8_t *)result->data;
        for (int i = 0; i < total_elements; i++)
        {
            if (d2[i] != 0)
            {
                res_data[i] = d1[i] / d2[i];
            }
            else
            {
                res_data[i] = 0;
            }
        }
    }

    return result;
}