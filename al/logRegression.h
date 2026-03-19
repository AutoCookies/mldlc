#ifndef LOGREGRESSION_H
#define LOGREGRESSION_H

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
} LogisticRegression;

/**
 * @brief Creates a Logistic Regression model.
 */
LogisticRegression* createLogisticRegression(int input_size, float lr, int iter, RegularizationType reg_type, float lambda, float alpha);

/**
 * @brief Frees the Logistic Regression model.
 */
void freeLogisticRegression(LogisticRegression* model);

/**
 * @brief Predicts class probabilities (0 to 1).
 * Returns a matrix of shape (n_samples, 1).
 */
Matrix* predictLogisticRegression(const LogisticRegression* model, const Matrix* X);

/**
 * @brief Predicts binary classes (0 or 1) based on a threshold (default 0.5).
 */
Matrix* predictLogisticRegressionClass(const LogisticRegression* model, const Matrix* X, float threshold);

/**
 * @brief Trains the Logistic Regression model using Gradient Descent.
 */
void trainLogisticRegression(LogisticRegression* model, const Matrix* X, const Matrix* y);

#endif
