#include "criteria.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

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

float calculateInformationGain(float parent_entropy, const float *child_entropies, const int *child_sizes, int n_children, int total_samples)
{
    float weighted_child_entropy = 0.0f;

    for (int i = 0; i < n_children; i++)
    {
        float weight = (float)child_sizes[i] / total_samples;
        weighted_child_entropy += weight * child_entropies[i];
    }

    return parent_entropy - weighted_child_entropy;
}