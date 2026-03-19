#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include "../objects/matrix.h"

/**
 * @brief Generic interface for machine learning models.
 */
typedef struct {
    /**
     * @brief Creates a new model instance.
     * @param n_features Number of input features.
     * @param params Opaque pointer to model-specific parameters (e.g., SVMParam).
     */
    void* (*create)(int n_features, void* params);

    /**
     * @brief Trains the model.
     */
    void (*train)(void* model, const Matrix* X, const Matrix* y);

    /**
     * @brief Predicts class labels or probabilities.
     */
    Matrix* (*predict)(void* model, const Matrix* X);

    /**
     * @brief Frees the model instance.
     */
    void (*free_model)(void* model);
} Estimator;

/**
 * @brief Standalone cross-validation scoring.
 * @return float Array of scores (one per fold).
 */
float* cross_val_score(const Estimator* estimator, const Matrix* X, const Matrix* y, void* params, int cv, int shuffle, int random_state);

/**
 * @brief Results of GridSearchCV.
 */
typedef struct {
    void* best_params;    // Pointer to the best configuration in the grid
    float best_score;     // Average cross-validated score of the best configuration
    int best_index;       // Index of the best configuration in the parameter grid
} GridSearchResult;

/**
 * @brief Exhaustive hyperparameter search.
 * @param params_grid Array of model-specific parameter structs to try.
 * @param n_candidates Number of configurations in params_grid.
 */
GridSearchResult grid_search(const Estimator* estimator, const Matrix* X, const Matrix* y, 
                            void** params_grid, int n_candidates, int cv);

#endif
