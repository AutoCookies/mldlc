#ifndef SVM_H
#define SVM_H

#include "../objects/matrix.h"

typedef struct {
    Matrix* weights;
    float bias;
    float C;            // Regularization parameter
    float learning_rate;
    int iterations;
} SVM;

/**
 * @brief Creates a new SVM model.
 * 
 * @param input_size Number of features.
 * @param C Regularization parameter (trade-off between margin and error).
 * @param lr Learning rate.
 * @param iter Number of training iterations.
 * @return SVM* Pointer to the created SVM model.
 */
SVM* createSVM(int input_size, float C, float lr, int iter);

/**
 * @brief Trains the SVM model using Primal Gradient Descent on Hinge Loss.
 * 
 * @param model Pointer to the SVM model.
 * @param X Training features (Input Matrix).
 * @param y Training labels (Input Matrix, expected values -1 or 1).
 */
void trainSVM(SVM* model, const Matrix* X, const Matrix* y);

/**
 * @brief Predicts class labels for a given set of features.
 * 
 * @param model Pointer to the SVM model.
 * @param X Input features.
 * @return Matrix* Predicted labels (-1 or 1).
 */
Matrix* predictSVM(const SVM* model, const Matrix* X);

/**
 * @brief Frees the SVM model.
 */
void freeSVM(SVM* model);

#endif
