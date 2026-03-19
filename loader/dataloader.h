#ifndef DATALOADER_H
#define DATALOADER_H

#include "../objects/matrix.h"

Matrix* loadCSV(const char* filename, int has_header);

void train_test_split(const Matrix* X, const Matrix* y, float test_size, int shuffle, int random_state, 
                      Matrix** X_train, Matrix** X_test, Matrix** y_train, Matrix** y_test);

typedef struct {
    int* train_indices;
    int* val_indices;
    int train_count;
    int val_count;
} KFoldSplit;

KFoldSplit* get_k_fold_indices(int n_samples, int k, int fold_idx, int shuffle, int random_state);
void free_k_fold_split(KFoldSplit* split);

#endif