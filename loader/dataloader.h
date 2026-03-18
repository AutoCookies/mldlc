#ifndef DATALOADER_H
#define DATALOADER_H

#include "../objects/matrix.h"

Matrix* loadCSV(const char* filename, int has_header);

#endif