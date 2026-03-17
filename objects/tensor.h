#ifndef TENSOR_H
#define TENSOR_H

#include "dtype.h"

// Define tensor struct
typedef struct {
  int rank;
  int* shape;
  DataType dtype;
  void* data;
} Tensor;

Tensor* createTensor(int rank, int* shape, DataType dtype);
void freeTensor(Tensor* tensor);

#endif
