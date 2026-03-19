#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "../objects/matrix.h"

typedef enum {
    TREE_CLASSIFICATION,
    TREE_REGRESSION
} TreeType;

typedef enum {
    CRITERION_ENTROPY,
    CRITERION_GINI,
    CRITERION_MSE
} SplitCriterion;

typedef struct DecisionNode {
    int feature_index;      // Index of feature to split on
    float threshold;        // Threshold value for split
    float value;            // Value if leaf node (class or regression target)
    int is_leaf;            // Flag for leaf node
    struct DecisionNode* left;
    struct DecisionNode* right;
} DecisionNode;

typedef struct {
    DecisionNode* root;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;
    float min_impurity_decrease;
    int random_state;
    TreeType type;
    SplitCriterion criterion;
} DecisionTree;

/**
 * @brief Creates a new Decision Tree with enhanced parameters.
 */
DecisionTree* createDecisionTree(TreeType type, SplitCriterion criterion, 
                                 int max_depth, int min_samples_split, 
                                 int min_samples_leaf, int max_features,
                                 float min_impurity_decrease, int random_state);

/**
 * @brief Trains the Decision Tree.
 */
void trainDecisionTree(DecisionTree* tree, const Matrix* X, const Matrix* y);

/**
 * @brief Predicts using the Decision Tree.
 */
Matrix* predictDecisionTree(const DecisionTree* tree, const Matrix* X);

/**
 * @brief Frees the Decision Tree.
 */
void freeDecisionTree(DecisionTree* tree);

#endif
