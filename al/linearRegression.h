#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "../objects/matrix.h"
#include "../calculations/regularize.h"

typedef struct {
    Matrix* weights;    
    float bias;      
    float learning_rate;
    int iterations;    
    RegularizationType reg_type;
    float lambda;
    float alpha;  
} LinearRegression;

LinearRegression* createLinearRegression(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha);
void freeLinearRegression(LinearRegression* model);
Matrix* predictLinearRegression(const LinearRegression* model, const Matrix* X);
void gradientDescentStep(LinearRegression *model, const Matrix *X, const Matrix *y);
void trainLinearRegression(LinearRegression* model, const Matrix* X, const Matrix* y);

#endif