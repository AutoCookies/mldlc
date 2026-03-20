#include "model_selection.h"
#include "../loader/dataloader.h"
#include "../calculations/metrics.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "palloc.h"

/* Helper to create subset matrices from indices */
static void createSubsets(const Matrix* X, const Matrix* y, const int* indices, int count, Matrix** subX, Matrix** suby) {
    *subX = createMatrix(count, X->cols, X->dtype);
    *suby = createMatrix(count, y->cols, y->dtype);
    int feat_size = X->cols * sizeof(float);
    int target_size = y->cols * sizeof(float);
    for (int i = 0; i < count; i++) {
        int idx = indices[i];
        memcpy((float*)(*subX)->data + i * X->cols, (float*)X->data + idx * X->cols, feat_size);
        memcpy((float*)(*suby)->data + i * y->cols, (float*)y->data + idx * y->cols, target_size);
    }
}

float* cross_val_score(const Estimator* est, const Matrix* X, const Matrix* y, void* params, int cv, int shuffle, int random_state) {
    if (!est || !X || !y) return NULL;
    
    float* scores = (float*)pa_malloc(cv * sizeof(float));
    int n_samples = X->rows;

    for (int f = 0; f < cv; f++) {
        KFoldSplit* split = get_k_fold_indices(n_samples, cv, f, shuffle, random_state);
        Matrix *trainX, *trainy, *valX, *valy;
        createSubsets(X, y, split->train_indices, split->train_count, &trainX, &trainy);
        createSubsets(X, y, split->val_indices, split->val_count, &valX, &valy);

        void* model = est->create(X->cols, params);
        est->train(model, trainX, trainy);
        Matrix* y_pred = est->predict(model, valX);
        
        // Scoring: Using accuracy as a default for now. 
        // In a "full" implementation, 'score' would also be a function pointer in the estimator.
        scores[f] = accuracyScore(valy, y_pred);

        freeMatrix(trainX); freeMatrix(trainy);
        freeMatrix(valX); freeMatrix(valy);
        freeMatrix(y_pred);
        est->free_model(model);
        free_k_fold_split(split);
    }
    return scores;
}

GridSearchResult grid_search(const Estimator* est, const Matrix* X, const Matrix* y, void** params_grid, int n_candidates, int cv) {
    GridSearchResult result = {NULL, -1.0f, -1};

    for (int i = 0; i < n_candidates; i++) {
        float* scores = cross_val_score(est, X, y, params_grid[i], cv, 1, 42);
        
        float sum = 0.0f;
        for (int j = 0; j < cv; j++) sum += scores[j];
        float avg = sum / cv;

        if (avg > result.best_score) {
            result.best_score = avg;
            result.best_params = params_grid[i];
            result.best_index = i;
        }
        pa_free(scores);
    }
    return result;
}
