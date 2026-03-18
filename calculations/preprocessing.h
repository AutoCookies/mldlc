#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "../objects/matrix.h"

void standardize(Matrix* X);
void minMaxScale(Matrix* X);

#endif