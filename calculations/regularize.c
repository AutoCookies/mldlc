#include "regularize.h"
#include <math.h>

void applyRegularizationGradient(RegularizationType type, float lambda, float alpha, const Matrix* weights, Matrix* gradient) {
    if (type == REG_NONE || lambda <= 0.0f) return;

    int n = weights->rows * weights->cols;
    float* w = (float*)weights->data;
    float* g = (float*)gradient->data;

    for (int i = 0; i < n; i++) {
        float penalty = 0.0f;
        if (type == REG_L1) {
            penalty = lambda * (w[i] > 0 ? 1.0f : (w[i] < 0 ? -1.0f : 0.0f));
        } else if (type == REG_L2) {
            penalty = 2.0f * lambda * w[i];
        } else if (type == REG_ELASTIC_NET) {
            float l1 = alpha * lambda * (w[i] > 0 ? 1.0f : (w[i] < 0 ? -1.0f : 0.0f));
            float l2 = (1.0f - alpha) * lambda * 2.0f * w[i];
            penalty = l1 + l2;
        }
        g[i] += penalty;
    }
}

float computeRegularizationPenalty(RegularizationType type, float lambda, float alpha, const Matrix* weights) {
    if (type == REG_NONE || lambda <= 0.0f) return 0.0f;

    int n = weights->rows * weights->cols;
    float* w = (float*)weights->data;
    float l1_sum = 0.0f;
    float l2_sum = 0.0f;

    for (int i = 0; i < n; i++) {
        if (type == REG_L1 || type == REG_ELASTIC_NET) {
            l1_sum += fabsf(w[i]);
        }
        if (type == REG_L2 || type == REG_ELASTIC_NET) {
            l2_sum += w[i] * w[i];
        }
    }

    if (type == REG_L1) return lambda * l1_sum;
    if (type == REG_L2) return lambda * l2_sum;
    if (type == REG_ELASTIC_NET) {
        return (alpha * lambda * l1_sum) + ((1.0f - alpha) * lambda * l2_sum);
    }

    return 0.0f;
}
