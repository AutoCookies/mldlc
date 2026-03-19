#include "decisionTree.h"
#include "../calculations/metrics.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

DecisionTree* createDecisionTree(TreeType type, SplitCriterion criterion, 
                                 int max_depth, int min_samples_split, 
                                 int min_samples_leaf, int max_features,
                                 float min_impurity_decrease, int random_state) {
    DecisionTree* tree = (DecisionTree*)malloc(sizeof(DecisionTree));
    if (tree == NULL) return NULL;
    tree->root = NULL;
    tree->type = type;
    tree->criterion = criterion;
    tree->max_depth = max_depth;
    tree->min_samples_split = min_samples_split;
    tree->min_samples_leaf = min_samples_leaf;
    tree->max_features = max_features;
    tree->min_impurity_decrease = min_impurity_decrease;
    tree->random_state = random_state;
    return tree;
}

static void freeNode(DecisionNode* node) {
    if (node == NULL) return;
    freeNode(node->left);
    freeNode(node->right);
    free(node);
}

void freeDecisionTree(DecisionTree* tree) {
    if (tree == NULL) return;
    freeNode(tree->root);
    free(tree);
}

// Internal structure for data subset
typedef struct {
    int* indices;
    int count;
} Subset;

static float calculateNodeValue(const DecisionTree* tree, const Matrix* y, Subset* subset) {
    float* y_data = (float*)y->data;
    if (tree->type == TREE_REGRESSION) {
        float sum = 0.0f;
        for (int i = 0; i < subset->count; i++) {
            sum += y_data[subset->indices[i]];
        }
        return sum / subset->count;
    } else {
        // Classification: Most frequent class
        // This is a simplified version assuming classes are 0, 1, ...
        // In a real implementation we'd need a more robust way to find the mode.
        // For now, let's find the max value to know the range.
        int max_class = 0;
        for (int i = 0; i < subset->count; i++) {
            if ((int)y_data[subset->indices[i]] > max_class) 
                max_class = (int)y_data[subset->indices[i]];
        }
        
        int* counts = (int*)calloc(max_class + 1, sizeof(int));
        for (int i = 0; i < subset->count; i++) {
            counts[(int)y_data[subset->indices[i]]]++;
        }

        int best_class = 0;
        int max_count = -1;
        for (int i = 0; i <= max_class; i++) {
            if (counts[i] > max_count) {
                max_count = counts[i];
                best_class = i;
            }
        }
        free(counts);
        return (float)best_class;
    }
}

static float calculateCriterion(const DecisionTree* tree, const Matrix* y, Subset* subset) {
    if (subset->count == 0) return 0.0f;
    float* y_data = (float*)y->data;

    if (tree->type == TREE_REGRESSION) {
        // MSE calculation for regression
        float mean = 0.0f;
        for (int i = 0; i < subset->count; i++) mean += y_data[subset->indices[i]];
        mean /= subset->count;
        
        float mse = 0.0f;
        for (int i = 0; i < subset->count; i++) {
            float diff = y_data[subset->indices[i]] - mean;
            mse += diff * diff;
        }
        return mse / subset->count;
    } else {
        // Classification: Entropy or Gini
        int max_class = 0;
        for (int i = 0; i < subset->count; i++) {
            if ((int)y_data[subset->indices[i]] > max_class) 
                max_class = (int)y_data[subset->indices[i]];
        }
        
        float* probs = (float*)calloc(max_class + 1, sizeof(float));
        for (int i = 0; i < subset->count; i++) {
            probs[(int)y_data[subset->indices[i]]]++;
        }
        for (int i = 0; i <= max_class; i++) {
            probs[i] /= subset->count;
        }

        float result = 0.0f;
        if (tree->criterion == CRITERION_ENTROPY) {
            result = calculateEntropy(probs, max_class + 1);
        } else {
            result = giniImpurity(probs, max_class + 1);
        }
        free(probs);
        return result;
    }
}

static DecisionNode* buildTreeRecursive(const DecisionTree* tree, const Matrix* X, const Matrix* y, Subset* subset, int depth) {
    if (subset->count == 0) return NULL;

    DecisionNode* node = (DecisionNode*)malloc(sizeof(DecisionNode));
    node->left = NULL;
    node->right = NULL;
    node->is_leaf = 0;
    node->value = calculateNodeValue(tree, y, subset);

    // Stop conditions
    if (depth >= tree->max_depth || subset->count < tree->min_samples_split) {
        node->is_leaf = 1;
        return node;
    }

    float current_score = calculateCriterion(tree, y, subset);
    if (current_score == 0.0f) {
        node->is_leaf = 1;
        return node;
    }

    float best_gain = -1.0f;
    int best_feat = -1;
    float best_thresh = 0.0f;

    float* x_data = (float*)X->data;
    float* y_data = (float*)y->data;

    // Feature selection (max_features)
    int n_all_features = X->cols;
    int n_features_to_check = (tree->max_features > 0 && tree->max_features < n_all_features) ? tree->max_features : n_all_features;
    int* feature_indices = (int*)malloc(n_all_features * sizeof(int));
    for (int i = 0; i < n_all_features; i++) feature_indices[i] = i;

    if (n_features_to_check < n_all_features) {
        srand(tree->random_state + depth); // seed with depth for variation
        for (int i = n_all_features - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = feature_indices[i];
            feature_indices[i] = feature_indices[j];
            feature_indices[j] = temp;
        }
    }

    // Find best split
    for (int k = 0; k < n_features_to_check; k++) {
        int f = feature_indices[k];
        for (int i = 0; i < subset->count; i++) {
            float threshold = x_data[subset->indices[i] * X->cols + f];
            
            Subset left_sub, right_sub;
            left_sub.indices = (int*)malloc(subset->count * sizeof(int));
            right_sub.indices = (int*)malloc(subset->count * sizeof(int));
            left_sub.count = 0;
            right_sub.count = 0;

            for (int j = 0; j < subset->count; j++) {
                if (x_data[subset->indices[j] * X->cols + f] <= threshold) {
                    left_sub.indices[left_sub.count++] = subset->indices[j];
                } else {
                    right_sub.indices[right_sub.count++] = subset->indices[j];
                }
            }

            if (left_sub.count >= tree->min_samples_leaf && right_sub.count >= tree->min_samples_leaf) {
                float left_score = calculateCriterion(tree, y, &left_sub);
                float right_score = calculateCriterion(tree, y, &right_sub);
                float weighted_score = (left_score * left_sub.count + right_score * right_sub.count) / subset->count;
                float gain = current_score - weighted_score;

                if (gain > best_gain && gain >= tree->min_impurity_decrease) {
                    best_gain = gain;
                    best_feat = f;
                    best_thresh = threshold;
                }
            }
            free(left_sub.indices);
            free(right_sub.indices);
        }
    }
    free(feature_indices);

    if (best_feat == -1) {
        node->is_leaf = 1;
        return node;
    }

    node->feature_index = best_feat;
    node->threshold = best_thresh;

    // Split again to recurse
    Subset left_sub, right_sub;
    left_sub.indices = (int*)malloc(subset->count * sizeof(int));
    right_sub.indices = (int*)malloc(subset->count * sizeof(int));
    left_sub.count = 0;
    right_sub.count = 0;
    for (int i = 0; i < subset->count; i++) {
        if (x_data[subset->indices[i] * X->cols + best_feat] <= best_thresh) {
            left_sub.indices[left_sub.count++] = subset->indices[i];
        } else {
            right_sub.indices[right_sub.count++] = subset->indices[i];
        }
    }

    node->left = buildTreeRecursive(tree, X, y, &left_sub, depth + 1);
    node->right = buildTreeRecursive(tree, X, y, &right_sub, depth + 1);

    free(left_sub.indices);
    free(right_sub.indices);

    return node;
}

void trainDecisionTree(DecisionTree* tree, const Matrix* X, const Matrix* y) {
    Subset full_set;
    full_set.count = X->rows;
    full_set.indices = (int*)malloc(full_set.count * sizeof(int));
    for (int i = 0; i < full_set.count; i++) full_set.indices[i] = i;

    tree->root = buildTreeRecursive(tree, X, y, &full_set, 0);
    free(full_set.indices);
}

static float predictSample(const DecisionNode* node, const float* sample, int n_cols) {
    if (node->is_leaf) return node->value;
    if (sample[node->feature_index] <= node->threshold) {
        return predictSample(node->left, sample, n_cols);
    } else {
        return predictSample(node->right, sample, n_cols);
    }
}

Matrix* predictDecisionTree(const DecisionTree* tree, const Matrix* X) {
    if (tree->root == NULL) return NULL;
    Matrix* predictions = createMatrix(X->rows, 1, DTYPE_FLOAT32);
    float* x_data = (float*)X->data;
    float* p_data = (float*)predictions->data;

    for (int i = 0; i < X->rows; i++) {
        p_data[i] = predictSample(tree->root, x_data + i * X->cols, X->cols);
    }
    return predictions;
}