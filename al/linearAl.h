#ifndef LINEARAL_H
#define LINEARAL_H

#include "../objects/matrix.h"
#include "../calculations/regularize.h"

/* --- Linear Regression (Regression) --- */

typedef struct {
    Matrix* weights;    
    float bias;      
    float learning_rate;
    int iterations;    
    RegularizationType reg_type;
    float lambda;
    float alpha;  
} LinearRegressor;

LinearRegressor* createLinearRegressor(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha);
void freeLinearRegressor(LinearRegressor* model);
Matrix* predictLinearRegressor(const LinearRegressor* model, const Matrix* X);
void trainLinearRegressor(LinearRegressor* model, const Matrix* X, const Matrix* y);

/* --- Linear Classification (Logistic Regression) --- */

typedef struct {
    Matrix* weights;    
    float bias;      
    float learning_rate;
    int iterations;    
    RegularizationType reg_type;
    float lambda;
    float alpha;
} LinearClassifier;

LinearClassifier* createLinearClassifier(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha);
void freeLinearClassifier(LinearClassifier* model);
Matrix* predictLinearClassifier(const LinearClassifier* model, const Matrix* X);
Matrix* predictLinearClassifierClass(const LinearClassifier* model, const Matrix* X, float threshold);
void trainLinearClassifier(LinearClassifier* model, const Matrix* X, const Matrix* y);

#endif
