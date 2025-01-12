#ifndef CTORCH_CNN_FUNCTIONAL_H
#define CTORCH_CNN_FUNCTIONAL_H

#include "ctorch.h"
#include "ctorch.cnn.h"
#include <iostream>

// all default to dim=0, works for linear models
void ReLU(CTorch::CTensor*& ct);
void exp(CTorch::CTensor*& ct);
void softmax(CTorch::CTensor*& ct);
void sigmoid(CTorch::CTensor*& ct);

#endif // CTORCH_CNN_FUNCTIONAL_H