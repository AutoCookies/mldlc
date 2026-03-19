#ifdef CRITERIA_H
#define CRITERIA_H

float calculateEntropy(const float *probs, int n_classes);
float giniImpurity(const float *probs, int n_classes);
float calculateInformationGain(float parent_entropy, const float *child_entropies, const int *child_sizes, int n_children, int total_samples);

#endif