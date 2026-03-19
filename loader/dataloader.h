#ifndef DATALOADER_H
#define DATALOADER_H

#include "../objects/matrix.h"

Matrix* loadCSV(const char* filename, int has_header);

void train_test_split(const Matrix* X, const Matrix* y, float test_size, int shuffle, int random_state, 
                      Matrix** X_train, Matrix** X_test, Matrix** y_train, Matrix** y_test);

#endif