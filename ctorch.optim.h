#ifndef CTORCH_OPTIM_H
#define CTORCH_OPTIM_H

#include "ctorch.h"
#include <iostream>
#include <vector>

class Optim {
    public:
        class COptimizer {
            public:
                virtual ~COptimizer();
                virtual void zero_grad() = 0;
                virtual void step() = 0;
        };
        class CAdamW : public COptimizer {
            CTorch::CTensor* params;
            CTorch::CTensor* grads;
            float lr;
            std::tuple<float, float> beta;
            float eps;
            public:
                CAdamW(std::vector<CTorch::CTensor*> params, float lr=0.001, std::tuple<float, float> beta=std::tuple<float, float>(0.9, 0.999), float eps=1e-08);
                ~CAdamW();
                void zero_grad() override;
                void step() override;
        };
};

#endif // CTORCH_OPTIM_H