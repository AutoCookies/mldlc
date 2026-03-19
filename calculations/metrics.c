#include "metrics.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

float meanAbsoluteError(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols || y_true->dtype != y_pred->dtype)
        return -1.0f;

    int n = y_true->rows * y_true->cols;
    float sum = 0.0f;

    if (y_true->dtype == DTYPE_FLOAT32)
    {
        float *t = (float *)y_true->data;
        float *p = (float *)y_pred->data;
        for (int i = 0; i < n; i++)
        {
            sum += fabsf(t[i] - p[i]);
        }
    }
    else if (y_true->dtype == DTYPE_INT32)
    {
        int32_t *t = (int32_t *)y_true->data;
        int32_t *p = (int32_t *)y_pred->data;
        for (int i = 0; i < n; i++)
        {
            sum += (float)abs(t[i] - p[i]);
        }
    }

    return sum / n;
}

float meanSquaredError(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols || y_true->dtype != y_pred->dtype)
        return -1.0f;

    int n = y_true->rows * y_true->cols;
    float sum = 0.0f;

    if (y_true->dtype == DTYPE_FLOAT32)
    {
        float *t = (float *)y_true->data;
        float *p = (float *)y_pred->data;
        for (int i = 0; i < n; i++)
        {
            float diff = t[i] - p[i];
            sum += diff * diff;
        }
    }
    else if (y_true->dtype == DTYPE_INT32)
    {
        int32_t *t = (int32_t *)y_true->data;
        int32_t *p = (int32_t *)y_pred->data;
        for (int i = 0; i < n; i++)
        {
            int32_t diff = t[i] - p[i];
            sum += (float)(diff * diff);
        }
    }

    return sum / n;
}

float rootMeanSquaredError(const Matrix *y_true, const Matrix *y_pred)
{
    float mse = meanSquaredError(y_true, y_pred);
    if (mse < 0)
        return -1.0f;
    return sqrtf(mse);
}

float binaryCrossEntropy(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols || y_true->dtype != y_pred->dtype)
        return -1.0f;
    if (y_true->dtype != DTYPE_FLOAT32)
        return -1.0f;

    int n = y_true->rows * y_true->cols;
    float sum = 0.0f;
    float *t = (float *)y_true->data;
    float *p = (float *)y_pred->data;
    float epsilon = 1e-15f; // To avoid log(0)

    for (int i = 0; i < n; i++)
    {
        float pred = p[i];
        if (pred < epsilon)
            pred = epsilon;
        if (pred > 1.0f - epsilon)
            pred = 1.0f - epsilon;

        sum += -(t[i] * logf(pred) + (1.0f - t[i]) * logf(1.0f - pred));
    }

    return sum / n;
}

float rSquared(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols || y_true->dtype != y_pred->dtype)
        return -1.0f;
    if (y_true->dtype != DTYPE_FLOAT32)
        return -1.0f;

    int n = y_true->rows * y_true->cols;
    float *t = (float *)y_true->data;
    float *p = (float *)y_pred->data;

    float rss = 0.0f; // Residual Sum of Squares
    float tss = 0.0f; // Total Sum of Squares
    float y_mean = 0.0f;

    for (int i = 0; i < n; i++)
    {
        y_mean += t[i];
    }
    y_mean /= (float)n;

    for (int i = 0; i < n; i++)
    {
        float res = t[i] - p[i];
        rss += res * res;
        float err = t[i] - y_mean;
        tss += err * err;
    }

    if (tss == 0.0f)
        return 1.0f; // Perfect fit if no variance? or 0? Usually 1 if p matches exactly.
    return 1.0f - (rss / tss);
}

float adjustedRSquared(float r2, int n, int k)
{
    if (n - k - 1 <= 0)
        return r2; // Avoid division by zero or negative
    return 1.0f - (((1.0f - r2) * (n - 1)) / (n - k - 1));
}

float accuracyScore(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols)
        return 0.0f;
    int total = y_true->rows * y_true->cols;
    float *t = (float *)y_true->data;
    float *p = (float *)y_pred->data;
    int correct = 0;
    for (int i = 0; i < total; i++)
    {
        if (t[i] == p[i])
            correct++;
    }
    return (float)correct / total;
}

float precisionScore(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols)
        return 0.0f;
    int total = y_true->rows * y_true->cols;
    float *t = (float *)y_true->data;
    float *p = (float *)y_pred->data;
    int tp = 0, fp = 0;
    for (int i = 0; i < total; i++)
    {
        if (p[i] == 1.0f)
        {
            if (t[i] == 1.0f)
                tp++;
            else
                fp++;
        }
    }
    if (tp + fp == 0)
        return 0.0f;
    return (float)tp / (tp + fp);
}

float recallScore(const Matrix *y_true, const Matrix *y_pred)
{
    if (y_true->rows != y_pred->rows || y_true->cols != y_pred->cols)
        return 0.0f;
    int total = y_true->rows * y_true->cols;
    float *t = (float *)y_true->data;
    float *p = (float *)y_pred->data;
    int tp = 0, fn = 0;
    for (int i = 0; i < total; i++)
    {
        if (t[i] == 1.0f)
        {
            if (p[i] == 1.0f)
                tp++;
            else
                fn++;
        }
    }
    if (tp + fn == 0)
        return 0.0f;
    return (float)tp / (tp + fn);
}

float f1Score(const Matrix *y_true, const Matrix *y_pred)
{
    float precision = precisionScore(y_true, y_pred);
    float recall = recallScore(y_true, y_pred);
    if (precision + recall == 0)
        return 0.0f;
    return 2.0f * (precision * recall) / (precision + recall);
}

float calculateEntropy(const float *probs, int n_classes)
{
    float entropy = 0.0f;
    for (int i = 0; i < n_classes; i++)
    {
        if (probs[i] > 0.0f)
        {
            entropy -= probs[i] * (logf(probs[i]) / logf(2.0f));
        }
    }
    return entropy;
}

float giniImpurity(const float *probs, int n_classes)
{
    float gini = 1.0f;
    for (int i = 0; i < n_classes; i++)
    {
        gini -= probs[i] * probs[i];
    }
    return gini;
}