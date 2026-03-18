#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix* loadCSV(const char* filename, int has_header) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;

    char line[1024];
    int rows = 0;
    int cols = 0;

    if (fgets(line, sizeof(line), file)) {
        char* tmp = strdup(line);
        char* token = strtok(tmp, ",");
        while (token) {
            cols++;
            token = strtok(NULL, ",");
        }
        free(tmp);
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