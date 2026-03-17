#include "tensor.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

Tensor* createTensor(int rank, int* shape, DataType dtype) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) return NULL;

    tensor->rank = rank;
    tensor->dtype = dtype;

    tensor->shape = (int*)malloc(rank * sizeof(int));
    size_t total_elements = 1;
    for (int i = 0; i < rank; i++) {
        tensor->shape[i] = shape[i];
        total_elements *= shape[i];
    }

    size_t element_size = 0;
    switch (dtype) {
        case DTYPE_FLOAT32: element_size = sizeof(float); break;
        case DTYPE_INT8:    element_size = sizeof(int8_t); break;
        case DTYPE_INT32:   element_size = sizeof(int32_t); break;
    }

    tensor->data = malloc(total_elements * element_size);

    return tensor;
}

void freeTensor(Tensor* tensor) {
    if (tensor == NULL) return;
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}
