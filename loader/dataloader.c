#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "palloc.h"

Matrix* loadCSV(const char* filename, int has_header) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;

    char line[1024];
    int rows = 0;
    int cols = 0;

    if (fgets(line, sizeof(line), file)) {
        char* tmp = pa_strdup(line);
        char* token = strtok(tmp, ",");
        while (token) {
            cols++;
            token = strtok(NULL, ",");
        }
        pa_free(tmp);
        if (!has_header) rows++;
    }

    while (fgets(line, sizeof(line), file)) rows++;

    Matrix* matrix = createMatrix(rows, cols, DTYPE_FLOAT32);
    float* data = (float*)matrix->data;

    rewind(file);
    if (has_header) fgets(line, sizeof(line), file);

    int r = 0;
    while (fgets(line, sizeof(line), file) && r < rows) {
        char* token = strtok(line, ",");
        int c = 0;
        while (token && c < cols) {
            data[r * cols + c] = (float)atof(token);
            token = strtok(NULL, ",");
            c++;
        }
        r++;
    }
    fclose(file);
    return matrix;
}

void train_test_split(const Matrix* X, const Matrix* y, float test_size, int shuffle, int random_state, 
                      Matrix** X_train, Matrix** X_test, Matrix** y_train, Matrix** y_test) {
    if (X == NULL || y == NULL) return;
    if (X->rows != y->rows) {
        printf("Error: X and y must have the same number of rows.\n");
        return;
    }

    int n_samples = X->rows;
    int n_test = (int)(n_samples * test_size);
    int n_train = n_samples - n_test;

    *X_train = createMatrix(n_train, X->cols, X->dtype);
    *X_test = createMatrix(n_test, X->cols, X->dtype);
    *y_train = createMatrix(n_train, y->cols, y->dtype);
    *y_test = createMatrix(n_test, y->cols, y->dtype);

    int* indices = (int*)pa_malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) indices[i] = i;

    if (shuffle) {
        srand(random_state);
        for (int i = n_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    int feat_size = X->cols * sizeof(float);
    int target_size = y->cols * sizeof(float);

    for (int i = 0; i < n_train; i++) {
        int idx = indices[i];
        memcpy((float*)(*X_train)->data + i * X->cols, (float*)X->data + idx * X->cols, feat_size);
        memcpy((float*)(*y_train)->data + i * y->cols, (float*)y->data + idx * y->cols, target_size);
    }

    for (int i = 0; i < n_test; i++) {
        int idx = indices[n_train + i];
        memcpy((float*)(*X_test)->data + i * X->cols, (float*)X->data + idx * X->cols, feat_size);
        memcpy((float*)(*y_test)->data + i * y->cols, (float*)y->data + idx * y->cols, target_size);
    }

    pa_free(indices);
}

KFoldSplit* get_k_fold_indices(int n_samples, int k, int fold_idx, int shuffle, int random_state) {
    if (fold_idx < 0 || fold_idx >= k) return NULL;

    KFoldSplit* split = (KFoldSplit*)pa_malloc(sizeof(KFoldSplit));
    int* indices = (int*)pa_malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i++) indices[i] = i;

    if (shuffle) {
        srand(random_state);
        for (int i = n_samples - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    int fold_size = n_samples / k;
    int val_start = fold_idx * fold_size;
    int val_end = (fold_idx == k - 1) ? n_samples : (fold_idx + 1) * fold_size;
    
    split->val_count = val_end - val_start;
    split->train_count = n_samples - split->val_count;
    split->val_indices = (int*)pa_malloc(split->val_count * sizeof(int));
    split->train_indices = (int*)pa_malloc(split->train_count * sizeof(int));

    int val_ptr = 0, train_ptr = 0;
    for (int i = 0; i < n_samples; i++) {
        if (i >= val_start && i < val_end) {
            split->val_indices[val_ptr++] = indices[i];
        } else {
            split->train_indices[train_ptr++] = indices[i];
        }
    }

    pa_free(indices);
    return split;
}

void free_k_fold_split(KFoldSplit* split) {
    if (split) {
        free(split->train_indices);
        free(split->val_indices);
        free(split);
    }
}